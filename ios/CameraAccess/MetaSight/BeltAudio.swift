//
// BeltAudio.swift
//
// The belt's buzzers, heard instead of felt, for early testing. One tone per motor, placed in
// stereo by the motor's angle (front = 0, clockwise) and pitched so front and back differ, gated
// by the motor's duty so the belt's own rhythm (slow, fast, solid, guide taps) comes through as
// beeps. It plays two things the server sends: `belt` (the 8 motor duties, live) when the
// "Belt as sound" switch is on, and `beltaudio` (haptic Braille pulses, braille = "audio") always.
//
// The rules below are the same as metasight/belt_audio.py, which renders them to WAV for the
// tests; keep the two in step. Output goes to the active route (the glasses' speakers over
// Bluetooth) with mixWithOthers, like SpatialBeeper. Silent on a simulator and on a Mac.
//

import AVFoundation
import Observation
import os

/// Pure rules: pan, pitch and loudness of a motor tone.
enum BeltTones {
  static let motorCount = 8
  static let baseHz = 660.0
  static let frontBackOctaves = 0.5
  static let guideRatio = 1.5
  static let voiceGain: Float = 0.35

  static func angleDegrees(_ motor: Int) -> Double { 45.0 * Double(motor) }

  /// -1 (left) ... +1 (right). Front and back are both centred; pitch tells them apart.
  static func pan(_ motor: Int) -> Float { Float(sin(angleDegrees(motor) * .pi / 180)) }

  /// Equal-power (left, right) gains.
  static func stereoGains(_ motor: Int) -> (left: Float, right: Float) {
    let p = Double(pan(motor) + 1) * .pi / 4
    return (Float(cos(p)), Float(sin(p)))
  }

  static func frequency(_ motor: Int, guide: Bool = false) -> Double {
    let f = baseHz * pow(2, frontBackOctaves * cos(angleDegrees(motor) * .pi / 180))
    return guide ? f * guideRatio : f
  }

  static func amplitude(duty: Int) -> Float {
    voiceGain * Float(max(0, min(255, duty))) / 255
  }

  /// Target amplitude and guide flag per motor from duties and mode names ("off", "guide"...).
  static func targets(duty: [Int], modes: [String]) -> [(amp: Float, guide: Bool)] {
    (0..<motorCount).map { i in
      let d = i < duty.count ? duty[i] : 0
      let guide = i < modes.count && modes[i] == "guide"
      return (amplitude(duty: d), guide)
    }
  }
}

/// The oscillators. `setTargets` is called from the main actor, `render` from the audio thread.
final class BeltToneSynth: @unchecked Sendable {
  private struct State {
    var amp = [Float](repeating: 0, count: BeltTones.motorCount)
    var guide = [Bool](repeating: false, count: BeltTones.motorCount)
  }
  private let state = OSAllocatedUnfairLock(initialState: State())
  private var current = [Float](repeating: 0, count: BeltTones.motorCount)
  private var phase = [Double](repeating: 0, count: BeltTones.motorCount)
  private let sampleRate: Double
  private let smoothing: Float

  init(sampleRate: Double) {
    self.sampleRate = sampleRate
    smoothing = Float(1 - exp(-1 / (0.004 * sampleRate)))  // 4 ms, so a gate never clicks
  }

  func setTargets(_ targets: [(amp: Float, guide: Bool)]) {
    state.withLock { s in
      for (i, t) in targets.enumerated() where i < BeltTones.motorCount {
        s.amp[i] = t.amp
        s.guide[i] = t.guide
      }
    }
  }

  func silence() { setTargets(BeltTones.targets(duty: [], modes: [])) }

  /// Fills `left` and `right` with `count` frames.
  func render(left: UnsafeMutablePointer<Float>, right: UnsafeMutablePointer<Float>, count: Int) {
    let s = state.withLock { $0 }
    let gains = (0..<BeltTones.motorCount).map { BeltTones.stereoGains($0) }
    let steps = (0..<BeltTones.motorCount).map {
      2 * Double.pi * BeltTones.frequency($0, guide: s.guide[$0]) / sampleRate
    }
    for n in 0..<count {
      var l: Float = 0
      var r: Float = 0
      for m in 0..<BeltTones.motorCount {
        current[m] += (s.amp[m] - current[m]) * smoothing
        if current[m] < 1e-5 && s.amp[m] == 0 { continue }
        phase[m] += steps[m]
        if phase[m] > 2 * .pi { phase[m] -= 2 * .pi }
        let v = current[m] * Float(sin(phase[m]))
        l += v * gains[m].left
        r += v * gains[m].right
      }
      left[n] = tanhf(l)
      right[n] = tanhf(r)
    }
  }
}

@MainActor
@Observable
final class BeltAudio {
  static let enabledKey = "metasight.beltAudio"

  /// "Belt as sound": hear the live belt motors. Persisted; off by default, since the spatial
  /// beeps already cover obstacles.
  var enabled: Bool {
    didSet {
      defaults.set(enabled, forKey: Self.enabledKey)
      if !enabled { synth?.silence() }
    }
  }

  let canSound: Bool
  @ObservationIgnored private let defaults: UserDefaults
  @ObservationIgnored private var engine: AVAudioEngine?
  @ObservationIgnored private var synth: BeltToneSynth?
  @ObservationIgnored private var lastUpdate = Date.distantPast
  @ObservationIgnored private var brailleUntil = Date.distantPast
  @ObservationIgnored private var watchdog: Task<Void, Never>?

  /// How long a tone may hold with no new message before it is cut: a dropped link never leaves
  /// a tone running.
  static let holdSeconds = 0.5

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    enabled = defaults.bool(forKey: Self.enabledKey)
    #if targetEnvironment(simulator)
      canSound = false
    #else
      canSound = !ProcessInfo.processInfo.isiOSAppOnMac
    #endif
  }

  /// A `belt` message: live motor duties. Ignored while a Braille cue is being heard.
  func update(duty: [Int], modes: [String], now: Date = Date()) {
    guard enabled, now >= brailleUntil else { return }
    apply(BeltTones.targets(duty: duty, modes: modes), now: now)
  }

  /// A `beltaudio` message: one frame of the Braille pulses, always played.
  func updateBraille(cmd: [[Int]], now: Date = Date()) {
    brailleUntil = now.addingTimeInterval(Self.holdSeconds)
    let duty = cmd.map { $0.count > 1 && $0[0] != 0 ? $0[1] : 0 }
    let modes = cmd.map { $0.first == 5 ? "guide" : "solid" }
    apply(BeltTones.targets(duty: duty, modes: modes), now: now)
  }

  private func apply(_ targets: [(amp: Float, guide: Bool)], now: Date) {
    guard canSound, startEngine() else { return }
    synth?.setTargets(targets)
    lastUpdate = now
    if watchdog == nil {
      watchdog = Task { [weak self] in
        while !Task.isCancelled {
          try? await Task.sleep(for: .milliseconds(100))
          self?.checkHold()
        }
      }
    }
  }

  private func checkHold() {
    if Date().timeIntervalSince(lastUpdate) > Self.holdSeconds { synth?.silence() }
  }

  private func startEngine() -> Bool {
    if let engine, engine.isRunning { return true }
    let engine = self.engine ?? AVAudioEngine()
    if synth == nil {
      let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
      let sampleRate = rate > 0 ? rate : 44_100
      let synth = BeltToneSynth(sampleRate: sampleRate)
      guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)
      else { return false }
      let node = AVAudioSourceNode(format: format) { _, _, frames, list in
        let buffers = UnsafeMutableAudioBufferListPointer(list)
        guard buffers.count >= 2,
          let l = buffers[0].mData?.assumingMemoryBound(to: Float.self),
          let r = buffers[1].mData?.assumingMemoryBound(to: Float.self)
        else { return noErr }
        synth.render(left: l, right: r, count: Int(frames))
        return noErr
      }
      engine.attach(node)
      engine.connect(node, to: engine.mainMixerNode, format: format)
      self.synth = synth
    }
    self.engine = engine
    #if os(iOS)
      let session = AVAudioSession.sharedInstance()
      let mixes =
        session.categoryOptions.contains(.mixWithOthers)
        || session.categoryOptions.contains(.duckOthers)
      if !(session.category == .playback && mixes) {
        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
      }
    #endif
    return (try? engine.start()) != nil
  }
}
