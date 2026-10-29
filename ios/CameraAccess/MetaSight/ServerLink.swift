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
//   camera -> {"type":"camera","source":"iphone","hfov":deg,"width":w,"height":h,"walk":id|null}
//   arpose -> {"type":"arpose","seq":n,"ts":ms,"m":[16 floats],"track":"..."} (just before its frame)
//   answer <- {"type":"answer","text":"...","label":s|null,"pin":{"near":[...],...}|null}
//   belt   <- {"type":"belt","duty":[8],"modes":[8],"lost":b,"found":b,"latency_ms":n|null}
//   beltcmd <- {"type":"beltcmd","pkt":"<base64 of the 26-byte belt command>"}  relayed to the belt over BLE
//   beltaudio <- {"type":"beltaudio","cmd":[[mode,duty] x 8],"flags":n}  haptic Braille pulses, heard
//   safety <- {"type":"safety","sectors":[8],"cues":[{"mode":...}],"path":m|null,"guide":{...}|null,"found":b}
//

import Foundation
import Observation
import os
#if canImport(UIKit)
  import UIKit
#endif

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
  /// True when this fix is the camera seeing it, false when it comes from the map; nil when
  /// the server does not say.
  var inView: Bool? = nil

  enum CodingKeys: String, CodingKey {
    case label, dist, bearing
    case inView = "in_view"
  }
}

/// The latest answer, with the item it is about and what it was seen near, when the
/// server named one.
struct AnswerNote: Equatable, Sendable {
  let text: String
  let label: String?
  let near: String?
  /// When the item was last seen, if the server had a pin for it.
  let seen: Date?
  let at: Date
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
  private(set) var statusLine: String = "Waiting for server"
  private(set) var lastAnswer: String?
  private(set) var answerNote: AnswerNote?
  /// Sector distances from the latest safety message, as forwarded to the watch.
  private(set) var proximity: ProximityReading?
  /// The obstacle warning the wearer sees on the lens, the belt chip and this card:
  /// the cued warning, held and smoothed by `WarningFilter`.
  private(set) var warning: ProximityWarning?

  // MARK: - Frame path (read off the main actor)

  nonisolated let frameGate = FrameGate()

  // MARK: - Private

  @ObservationIgnored private let urlSession = URLSession(configuration: .default)
  @ObservationIgnored private var socket: URLSessionWebSocketTask?
  @ObservationIgnored private var supervisor: Task<Void, Never>?
  @ObservationIgnored private var motionFlusher: Task<Void, Never>?
  @ObservationIgnored private var motionBuffer: [[Double]] = []
  @ObservationIgnored private var warningFilter = WarningFilter()
  @ObservationIgnored private let speaker = AnswerSpeaker()
  /// Spatial beeps for the proximity bands, through the active output (the glasses).
  @ObservationIgnored let beeper = SpatialBeeper()
  /// "Find my X" on the phone's haptics, from the guide distance.
  let finder = FindHaptics()
  /// The belt's motors (and haptic Braille pulses) as stereo tones, for testing without a belt.
  @ObservationIgnored let beltAudio = BeltAudio()
  /// The belt's Bluetooth link: the server's belt commands, relayed to the ESP32.
  let beltBLE = BeltBLE()
  /// Sent every time the socket connects (and right away when set), so a reconnect
  /// restores what the server needs to know about this camera.
  @ObservationIgnored private var hello: [String: Any]?

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
    send(json: ["type": "ask", "text": question])
  }

  /// Sets the message re-sent on every connect and sends it now.
  func announce(_ message: [String: Any]?) {
    hello = message
    if let message { send(json: message) }
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
      statusLine = "Invalid server address"
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
    statusLine = "Connected"
    if let hello { send(json: hello) }
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
    statusLine = "Server offline, retrying"
  }

  private func closeSocket() {
    guard let socket else { return }
    dropSocket(socket)
  }

  private func flushMotion() {
    guard !motionBuffer.isEmpty else { return }
    let batch = motionBuffer
    motionBuffer.removeAll(keepingCapacity: true)
    send(json: ["type": "motion", "s": batch])
  }

  func send(json object: [String: Any]) {
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
    struct Pin: Decodable {
      let near: [String]?
      let ts: Double?
    }
    let text: String
    let label: String?
    let pin: Pin?
  }

  private struct BeltMessage: Decodable {
    let duty: [Int]
    let modes: [String]
    let lost: Bool?
    let found: Bool?
    let latency_ms: Double?
  }

  private struct BeltCommandMessage: Decodable {
    let pkt: String
  }

  private struct BeltAudioMessage: Decodable {
    let cmd: [[Int]]
  }

  private struct SafetyMessage: Decodable {
    struct Cue: Decodable {
      let mode: String
    }
    let sectors: [Double?]?
    let cues: [Cue]?
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
      answerNote = AnswerNote(
        text: answer.text, label: answer.label, near: answer.pin?.near?.first,
        seen: answer.pin?.ts.map { Date(timeIntervalSince1970: $0) }, at: Date())
      speaker.speak(answer.text)
      #if canImport(UIKit)
        // VoiceOver forwards announcements to a paired Bluetooth Braille display.
        UIAccessibility.post(notification: .announcement, argument: answer.text)
      #endif
    case "belt":
      guard let msg = try? decoder.decode(BeltMessage.self, from: data) else { return }
      belt = BeltState(
        duty: Self.padded(msg.duty, with: 0),
        modes: Self.padded(msg.modes, with: "off"),
        lost: msg.lost ?? false,
        found: msg.found ?? false,
        latencyMs: msg.latency_ms
      )
      beltAudio.update(duty: msg.duty, modes: msg.modes)
    case "beltcmd":
      guard let msg = try? decoder.decode(BeltCommandMessage.self, from: data),
        let packet = Data(base64Encoded: msg.pkt)
      else { return }
      beltBLE.write(packet)
    case "beltaudio":
      guard let msg = try? decoder.decode(BeltAudioMessage.self, from: data) else { return }
      beltAudio.updateBraille(cmd: msg.cmd)
    case "safety":
      guard let msg = try? decoder.decode(SafetyMessage.self, from: data) else { return }
      guide = msg.guide
      finder.update(guide: msg.guide, found: msg.found ?? false)
      let reading = msg.sectors.flatMap { sectors in
        msg.cues.map { Self.proximity(sectors: sectors, modes: $0.map(\.mode), path: msg.path) }
      }
      warning = warningFilter.update(reading?.cueWarning, now: Date())
      statusLine = Self.statusLine(
        warning: warning, guide: Self.liveGuide(msg.guide, answer: answerNote), found: msg.found ?? false)
      if var reading {
        proximity = reading
        reading.find = finder.buzzing ? finder.dist : nil
        WatchLink.shared.send(reading)
        beeper.update(reading)
      }
    default:
      break
    }
  }

  /// The front sector also counts the walking path, as the belt's planner does.
  static func proximity(sectors: [Double?], modes: [String], path: Double?) -> ProximityReading {
    var distances = sectors
    if let path, !distances.isEmpty {
      distances[0] = min(distances[0] ?? .infinity, path)
    }
    return ProximityReading(distances: distances, modes: modes)
  }

  private static func padded<T>(_ values: [T], with filler: T) -> [T] {
    let head = Array(values.prefix(BeltState.motorCount))
    return head + Array(repeating: filler, count: BeltState.motorCount - head.count)
  }

  /// The guide worth naming on the status line: one with a fix, or one whose answer is
  /// still up (the lens's 12 s). A search the server never finishes would otherwise say
  /// "Finding laptop" for the rest of the walk.
  static func liveGuide(_ guide: GuideTarget?, answer: AnswerNote?, now: Date = Date()) -> GuideTarget? {
    guard let guide else { return nil }
    if guide.dist != nil || guide.bearing != nil { return guide }
    guard let answer, answer.label == guide.label,
      now.timeIntervalSince(answer.at) < DisplayCard.answerSeconds
    else { return nil }
    return guide
  }

  /// "Obstacle ahead, 1.5 m · Finding water bottle, 2.1 m", in the wearer's words. Built
  /// from the same cued warning as the lens and the belt chip, so the three never disagree;
  /// "Clear" only when nothing within the 2 m cue range was cued.
  static func statusLine(warning: ProximityWarning?, guide: GuideTarget?, found: Bool) -> String {
    var parts: [String] = []
    switch warning {
    case nil: parts.append("Clear ahead")
    case let w? where w.band == .stop: parts.append("Stop · obstacle \(w.directionName.lowercased())")
    case let w?: parts.append("Obstacle \(w.directionName.lowercased()), \(DisplayCard.distanceText(w.distance))")
    }
    if let guide {
      if let dist = guide.dist {
        parts.append(String(format: "Finding %@, %.1f m", guide.label, dist))
      } else {
        parts.append("Finding \(guide.label)")
      }
    }
    if found {
      parts.append("Found it")
    }
    return parts.joined(separator: " · ")
  }
}

// MARK: - The warning the wearer sees

// Shared by the lens, the belt chip and the phone card so all three show the same zone
// and the same number. The server's cues flicker with depth noise: a zone is only raised
// once its cue has held for a moment (less for Stop), it falls back one zone at a time,
// and the number moves at most once a second inside a zone.

struct WarningFilter {
  /// How long a cue must hold before its zone is shown: Stop almost at once, a far
  /// obstacle only once it is clearly there.
  static func rise(_ band: ProximityBand) -> TimeInterval {
    switch band {
    case .stop: 0.15
    case .fast: 0.3
    case .slow, .clear: 0.5
    }
  }

  /// How long the cue must stay below the shown zone before it drops one step. Stop lets
  /// go sooner: it rises in 0.15 s if the obstacle comes back, and a held Stop on a noisy
  /// stream turns the lens red for half the walk.
  static func fall(_ band: ProximityBand) -> TimeInterval { band == .stop ? 0.3 : 0.6 }
  /// The number inside a zone changes at most this often.
  static let numberHold: TimeInterval = 1
  /// Messages arrive about every 60 ms; a hold that is due within a millisecond is due.
  private static let slack: TimeInterval = 0.001

  private(set) var shown: ProximityWarning?
  /// When the raw cue first reached each band, and has stayed at or above it since.
  private var heldSince: [ProximityBand: Date] = [:]
  private var belowSince: Date?
  private var numberAt = Date.distantPast

  /// Feeds one safety message's cued warning; returns the warning to show.
  @discardableResult
  mutating func update(_ raw: ProximityWarning?, now: Date) -> ProximityWarning? {
    let rawBand = raw?.band ?? .clear
    for band in [ProximityBand.slow, .fast, .stop] {
      if rawBand >= band {
        heldSince[band] = heldSince[band] ?? now
      } else {
        heldSince[band] = nil
      }
    }
    let ready = [ProximityBand.stop, .fast, .slow].first {
      guard let since = heldSince[$0] else { return false }
      return now.timeIntervalSince(since) >= Self.rise($0) - Self.slack
    } ?? .clear
    var band = shown?.band ?? .clear

    if ready > band {
      band = ready
      belowSince = nil
    } else if rawBand < band {
      let since = belowSince ?? now
      belowSince = since
      if now.timeIntervalSince(since) >= Self.fall(band) - Self.slack {
        band = max(ready, ProximityBand(rawValue: band.rawValue - 1) ?? .clear)
        belowSince = rawBand < band ? now : nil
      }
    } else {
      belowSince = nil
    }

    guard band > .clear else {
      shown = nil
      return nil
    }
    let zoneChanged = band != shown?.band
    if let raw, raw.band >= band, zoneChanged || now.timeIntervalSince(numberAt) >= Self.numberHold - Self.slack {
      // A fresh reading for this zone: take its sector and number.
      if zoneChanged || DisplayCard.distanceText(raw.distance) != DisplayCard.distanceText(shown?.distance)
        || raw.sector != shown?.sector
      {
        numberAt = now
      }
      shown = ProximityWarning(sector: raw.sector, distance: raw.distance, band: band)
    } else if let previous = shown {
      // Falling, or holding the number: same sector and distance, in the zone now shown.
      shown = ProximityWarning(sector: previous.sector, distance: previous.distance, band: band)
    } else if let raw {
      shown = ProximityWarning(sector: raw.sector, distance: raw.distance, band: band)
      numberAt = now
    }
    return shown
  }
}
