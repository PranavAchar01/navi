//
// ServerLink.swift
//
// WebSocket bridge to the meta-sight server. Sends glasses camera frames, motion
// batches and questions; receives spoken answers, belt state and safety status.
//
// Wire protocol (shared with the browser client):
//   frame  -> binary: Float64 LE timestamp (ms since epoch) + UInt32 LE seq + JPEG bytes
//   motion -> {"type":"motion","s":[[t, ax, ay, az, gx, gy, gz], ...]} (gyro omitted when absent)
//   ask    -> {"type":"ask","text":"..."}
//   answer <- {"type":"answer","text":"..."}
//   belt   <- {"type":"belt","duty":[8],"modes":[8],"lost":b,"found":b,"latency_ms":n|null}
//   safety <- {"type":"safety","path":m|null,"guide":{...}|null,"found":b}
//

import Foundation
import Observation
import os

/// Latest belt state reported by the server.
struct BeltState: Equatable, Sendable {
  static let motorCount = 8

  var duty: [Int] = Array(repeating: 0, count: motorCount)
  var modes: [String] = Array(repeating: "off", count: motorCount)
  var lost: Bool = false
  var found: Bool = false
  var latencyMs: Double?
}

/// Where the server is currently guiding the wearer.
struct GuideTarget: Equatable, Sendable, Decodable {
  let label: String
  let dist: Double?
  /// Degrees, 0 = straight ahead, positive = to the right.
  let bearing: Double?
}

/// Thread-safe gate that keeps at most one camera frame in flight and hands out
/// sequence numbers. Read from the SDK's frame callback, off the main actor.
final class FrameGate: Sendable {
  private struct State {
    var open = false
    var busy = false
    var seq: UInt32 = 0
  }

  private let state = OSAllocatedUnfairLock(initialState: State())

  /// Claims the single in-flight slot. Returns the frame's sequence number, or
  /// nil when the link is down or a send is still pending (the frame is dropped).
  func tryAcquire() -> UInt32? {
    state.withLock { s in
      guard s.open, !s.busy else { return nil }
      s.busy = true
      let seq = s.seq
      s.seq &+= 1
      return seq
    }
  }

  func release() {
    state.withLock { $0.busy = false }
  }

  func setOpen(_ open: Bool) {
    state.withLock {
      $0.open = open
      $0.busy = false
    }
  }
}

@Observable
@MainActor
final class ServerLink {
  enum Connection: Equatable {
    case disconnected
    case connecting
    case connected

    var text: String {
      switch self {
      case .disconnected: return "offline"
      case .connecting: return "connecting"
      case .connected: return "connected"
      }
    }
  }

  static let defaultURL = "ws://Pranavs-Mac-Mini.local:8445/ws"
  private static let urlDefaultsKey = "metasight.serverURL"
  private static let reconnectInterval: Duration = .milliseconds(1500)
  private static let motionFlushInterval: Duration = .milliseconds(100)

  // MARK: - Observed state

  /// Server URL as typed by the user. Persisted; applied with `applyURL()`.
  var urlString: String
  private(set) var connection: Connection = .disconnected
  private(set) var belt = BeltState()
  private(set) var guide: GuideTarget?
  private(set) var statusLine: String = "waiting for server"
  private(set) var lastAnswer: String?

  // MARK: - Frame path (read off the main actor)

  nonisolated let frameGate = FrameGate()

  // MARK: - Private

  @ObservationIgnored private let urlSession = URLSession(configuration: .default)
  @ObservationIgnored private var socket: URLSessionWebSocketTask?
  @ObservationIgnored private var supervisor: Task<Void, Never>?
  @ObservationIgnored private var motionFlusher: Task<Void, Never>?
  @ObservationIgnored private var motionBuffer: [[Double]] = []
  @ObservationIgnored private let speaker = AnswerSpeaker()

  init(defaults: UserDefaults = .standard) {
    urlString = defaults.string(forKey: Self.urlDefaultsKey) ?? Self.defaultURL
  }

  // MARK: - Lifecycle

  /// Starts the connect/reconnect loop and the motion batcher. Idempotent.
  func start() {
    guard supervisor == nil else { return }
    supervisor = Task { [weak self] in
      while !Task.isCancelled {
        self?.connectIfNeeded()
        try? await Task.sleep(for: Self.reconnectInterval)
      }
    }
    motionFlusher = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: Self.motionFlushInterval)
        self?.flushMotion()
      }
    }
  }

  func stop() {
    supervisor?.cancel()
    supervisor = nil
    motionFlusher?.cancel()
    motionFlusher = nil
    closeSocket()
  }

  /// Persists the edited URL and reconnects to it right away.
  func applyURL() {
    urlString = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
    UserDefaults.standard.set(urlString, forKey: Self.urlDefaultsKey)
    closeSocket()
    connectIfNeeded()
  }

  // MARK: - Outgoing

  /// Sends one encoded frame packet. Callers must hold the `frameGate` slot;
  /// it is released when the send completes (or fails).
  func sendFrame(_ packet: Data) {
    guard let socket, connection == .connected else {
      frameGate.release()
      return
    }
    let gate = frameGate
    socket.send(.data(packet)) { _ in gate.release() }
  }

  /// Queues one motion sample, [t, ax, ay, az] or [t, ax, ay, az, gx, gy, gz];
  /// batches go out every ~100 ms.
  func addMotion(_ sample: [Double]) {
    guard connection == .connected else { return }
    motionBuffer.append(sample)
  }

  func ask(_ text: String) {
    let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !question.isEmpty else { return }
    sendJSON(["type": "ask", "text": question])
  }

  /// Builds the binary frame message: Float64 LE ms timestamp + UInt32 LE seq + JPEG.
  nonisolated static func framePacket(jpeg: Data, seq: UInt32, timestampMs: Double) -> Data {
    var packet = Data(capacity: 12 + jpeg.count)
    withUnsafeBytes(of: timestampMs.bitPattern.littleEndian) { packet.append(contentsOf: $0) }
    withUnsafeBytes(of: seq.littleEndian) { packet.append(contentsOf: $0) }
    packet.append(jpeg)
    return packet
  }

  // MARK: - Connection

  private func connectIfNeeded() {
    guard socket == nil else { return }
    guard let url = URL(string: urlString), let scheme = url.scheme?.lowercased(),
      scheme == "ws" || scheme == "wss", url.host != nil
    else {
      connection = .disconnected
      statusLine = "invalid server URL"
      return
    }

    let task = urlSession.webSocketTask(with: url)
    task.maximumMessageSize = 4 * 1024 * 1024
    socket = task
    connection = .connecting
    task.resume()

    // A pong confirms the handshake finished.
    task.sendPing { [weak self] error in
      Task { @MainActor in self?.handlePing(on: task, error: error) }
    }
    receiveLoop(on: task)
  }

  private func handlePing(on task: URLSessionWebSocketTask, error: Error?) {
    guard task === socket else { return }
    if error != nil {
      dropSocket(task)
    } else {
      markConnected()
    }
  }

  private func markConnected() {
    guard connection != .connected else { return }
    connection = .connected
    frameGate.setOpen(true)
    statusLine = "connected"
  }

  private func receiveLoop(on task: URLSessionWebSocketTask) {
    Task { [weak self] in
      do {
        while true {
          let message = try await task.receive()
          guard let self, task === self.socket else { return }
          self.markConnected()
          if case .string(let text) = message {
            self.handle(text)
          }
          // Binary messages (echoed frames) are ignored.
        }
      } catch {
        self?.dropSocket(task)
      }
    }
  }

  private func dropSocket(_ task: URLSessionWebSocketTask) {
    guard task === socket else { return }
    task.cancel(with: .goingAway, reason: nil)
    socket = nil
    connection = .disconnected
    frameGate.setOpen(false)
    motionBuffer.removeAll()
    statusLine = "server offline, retrying"
  }

  private func closeSocket() {
    guard let socket else { return }
    dropSocket(socket)
  }

  private func flushMotion() {
    guard !motionBuffer.isEmpty else { return }
    let batch = motionBuffer
    motionBuffer.removeAll(keepingCapacity: true)
    sendJSON(["type": "motion", "s": batch])
  }

  private func sendJSON(_ object: [String: Any]) {
    guard let socket, connection == .connected,
      let data = try? JSONSerialization.data(withJSONObject: object),
      let text = String(data: data, encoding: .utf8)
    else { return }
    socket.send(.string(text)) { _ in }
  }

  // MARK: - Incoming

  private struct Envelope: Decodable {
    let type: String
  }

  private struct AnswerMessage: Decodable {
    let text: String
  }

  private struct BeltMessage: Decodable {
    let duty: [Int]
    let modes: [String]
    let lost: Bool?
    let found: Bool?
    let latency_ms: Double?
  }

  private struct SafetyMessage: Decodable {
    let path: Double?
    let guide: GuideTarget?
    let found: Bool?
  }

  private func handle(_ text: String) {
    let data = Data(text.utf8)
    let decoder = JSONDecoder()
    guard let envelope = try? decoder.decode(Envelope.self, from: data) else { return }
    switch envelope.type {
    case "answer":
      guard let answer = try? decoder.decode(AnswerMessage.self, from: data) else { return }
      lastAnswer = answer.text
      speaker.speak(answer.text)
    case "belt":
      guard let msg = try? decoder.decode(BeltMessage.self, from: data) else { return }
      belt = BeltState(
        duty: Self.padded(msg.duty, with: 0),
        modes: Self.padded(msg.modes, with: "off"),
        lost: msg.lost ?? false,
        found: msg.found ?? false,
        latencyMs: msg.latency_ms
      )
    case "safety":
      guard let msg = try? decoder.decode(SafetyMessage.self, from: data) else { return }
      guide = msg.guide
      statusLine = Self.statusLine(path: msg.path, guide: msg.guide, found: msg.found ?? false)
    default:
      break
    }
  }

  private static func padded<T>(_ values: [T], with filler: T) -> [T] {
    let head = Array(values.prefix(BeltState.motorCount))
    return head + Array(repeating: filler, count: BeltState.motorCount - head.count)
  }

  /// "path 1.2 m · guiding to water bottle 2.1 m"
  static func statusLine(path: Double?, guide: GuideTarget?, found: Bool) -> String {
    var parts: [String] = []
    if let path {
      parts.append(String(format: "path %.1f m", path))
    } else {
      parts.append("path clear")
    }
    if let guide {
      if let dist = guide.dist {
        parts.append(String(format: "guiding to %@ %.1f m", guide.label, dist))
      } else {
        parts.append("guiding to \(guide.label)")
      }
    }
    if found {
      parts.append("found it")
    }
    return parts.joined(separator: " · ")
  }
}
