//
// ConversationAudio.swift
//
// Sends the microphone to the server as {"type":"audio"} chunks so it can tell when someone is
// speaking, which way the voice comes from, and which person in view is talking.
//
// What each microphone gives (the server takes whatever channel count arrives):
//  - iPhone built-in microphones: asked for a stereo pair (AVAudioSession polar pattern .stereo,
//    which pairs two of the phone's microphones). If the phone hands back one channel the server
//    runs without a voice direction. `spacing` is the assumed distance between that pair.
//  - Glasses: the Wearables Device Access Toolkit gives the glasses microphone only as a
//    Bluetooth hands-free (HFP) input, which is mono, and its camera-stream PCM audio is marked
//    experimental and unavailable for published apps. Mono means voice activity and lips, no
//    direction. `AudioInputHandler` (Media/) is the toolkit sample's route handling; `useGlasses`
//    feeds this sender from it instead of the phone microphone. Not testable without glasses.
//
// Nothing is recorded or kept on the phone: chunks go straight to the socket and are dropped
// when it is closed.
//

import AVFoundation
import Foundation
import os

final class ConversationAudio: @unchecked Sendable {
  static let sampleRate = 16_000.0
  /// Assumed distance in metres between the two phone microphones used for stereo.
  static let phoneSpacing = 0.13
  static let enabledKey = "metasight.talkAudio"

  private static let logger = Logger(subsystem: "com.metasight", category: "ConversationAudio")

  private let engine = AVAudioEngine()
  private let queue = DispatchQueue(label: "metasight.talk-audio")
  private var converter: AVAudioConverter?
  private var sender: (([String: Any]) -> Void)?
  private(set) var channels = 0
  private(set) var running = false
  /// Chunks sent since start, for tests and the settings screen.
  private(set) var chunksSent = 0

  var enabled: Bool {
    get { UserDefaults.standard.object(forKey: Self.enabledKey) as? Bool ?? true }
    set { UserDefaults.standard.set(newValue, forKey: Self.enabledKey) }
  }

  /// Starts the phone microphone. `send` receives the JSON object for each chunk. Returns false
  /// when the microphone is off, denied or unavailable.
  @discardableResult
  func start(send: @escaping ([String: Any]) -> Void) -> Bool {
    guard enabled, !running else { return running }
    // A unit-test host has no audio hardware: AVAudioEngine.inputNode there times out on the
    // simulator's audio RPC and aborts the whole test run.
    guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else { return false }
    sender = send
    let session = AVAudioSession.sharedInstance()
    do {
      try session.setCategory(
        .playAndRecord, mode: .measurement, options: [.mixWithOthers, .allowBluetoothHFP, .defaultToSpeaker])
      try session.setActive(true)
      Self.preferStereo(session)
    } catch {
      Self.logger.error("audio session: \(error.localizedDescription)")
      return false
    }
    let input = engine.inputNode
    let inFormat = input.outputFormat(forBus: 0)
    guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else { return false }
    let n = min(2, Int(inFormat.channelCount))
    guard
      let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: Self.sampleRate, channels: AVAudioChannelCount(n),
        interleaved: true),
      let conv = AVAudioConverter(from: inFormat, to: outFormat)
    else { return false }
    converter = conv
    channels = n
    input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
      self?.queue.async { self?.emit(buffer, to: outFormat) }
    }
    do {
      try engine.start()
    } catch {
      input.removeTap(onBus: 0)
      Self.logger.error("audio engine: \(error.localizedDescription)")
      return false
    }
    running = true
    return true
  }

  func stop() {
    guard running else { return }
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    running = false
    sender = nil
  }

  /// Mono samples from the glasses microphone (see the header). Resampled like the phone path.
  func pushGlasses(samples: [Float], sampleRate: Int) {
    guard enabled, let sender else { return }
    queue.async { [weak self] in
      guard let self else { return }
      let step = Double(sampleRate) / Self.sampleRate
      let n = Int(Double(samples.count) / step)
      var pcm = [Int16](repeating: 0, count: n)
      for i in 0..<n {
        pcm[i] = Int16(max(-1, min(1, samples[min(samples.count - 1, Int(Double(i) * step))])) * 32767)
      }
      sender(Self.message(pcm: pcm, channels: 1))
      self.chunksSent += 1
    }
  }

  static func message(pcm: [Int16], channels: Int) -> [String: Any] {
    let data = pcm.withUnsafeBufferPointer { Data(buffer: $0) }
    var m: [String: Any] = [
      "type": "audio", "sr": Int(sampleRate), "ch": channels, "pcm": data.base64EncodedString(),
    ]
    if channels >= 2 { m["spacing"] = phoneSpacing }
    return m
  }

  private func emit(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) {
    guard let converter, let sender else { return }
    let ratio = Self.sampleRate / buffer.format.sampleRate
    let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
    guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: cap) else { return }
    var fed = false
    var err: NSError?
    converter.convert(to: out, error: &err) { _, status in
      if fed {
        status.pointee = .noDataNow
        return nil
      }
      fed = true
      status.pointee = .haveData
      return buffer
    }
    guard err == nil, out.frameLength > 0, let p = out.int16ChannelData?[0] else { return }
    let count = Int(out.frameLength) * Int(format.channelCount)
    let pcm = Array(UnsafeBufferPointer(start: p, count: count))
    sender(Self.message(pcm: pcm, channels: Int(format.channelCount)))
    chunksSent += 1
  }

  private static func preferStereo(_ session: AVAudioSession) {
    guard let source = session.preferredInput?.dataSources?.first(where: { $0.supportedPolarPatterns?.contains(.stereo) == true })
      ?? session.inputDataSource
    else { return }
    if source.supportedPolarPatterns?.contains(.stereo) == true {
      try? source.setPreferredPolarPattern(.stereo)
      try? session.setInputDataSource(source)
      try? session.setPreferredInputOrientation(.portrait)
    }
  }
}
