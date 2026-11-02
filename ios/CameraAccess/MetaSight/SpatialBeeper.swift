//
// SpatialBeeper.swift
//
// Plays BeepCue plans as 3D sound, so the wearer hears the obstacle where it is. The engine
// layout follows Microsoft Soundscape's AudioEngine (MIT, see ios/THIRD_PARTY.md): a mono
// AVAudioPlayerNode per source rendered with HRTFHQ, through an AVAudioEnvironmentNode into
// the main mixer, the source placed from a bearing.
//
// Output goes to whatever route is active (the glasses' open-ear speakers over Bluetooth).
// The session is .playback + .mixWithOthers, so it never interrupts the glasses' camera
// stream or the answer speaker. Nothing ever plays on a simulator or a Mac: there the beeper
// is silent, and tests render offline into a buffer with `.offline`.
//
// FindHaptics, at the end, is the phone's "find my X" buzz through Core Haptics.
//

import AVFoundation
import CoreHaptics
import Observation

@MainActor
@Observable
final class SpatialBeeper {
  enum Output {
    /// The phone's active audio route.
    case device
    /// Manual rendering into memory (tests): no hardware, no sound.
    case offline(maximumFrames: AVAudioFrameCount)
  }

  static let sampleRate = 44_100.0

  /// Persisted. Silent by default while the home is being mapped; the picker is on the
  /// camera screen and in Settings.
  var volume: BeepVolume {
    didSet { UserDefaults.standard.set(volume.rawValue, forKey: BeepVolume.defaultsKey) }
  }

  /// False on the simulator and on a Mac running the iPhone app: those speakers stay silent.
  let canSound: Bool

  @ObservationIgnored private let output: Output
  @ObservationIgnored private let engine = AVAudioEngine()
  @ObservationIgnored private let environment = AVAudioEnvironmentNode()
  @ObservationIgnored private let player = AVAudioPlayerNode()
  @ObservationIgnored private var buffers: [BeepKind: AVAudioPCMBuffer] = [:]
  @ObservationIgnored private var scheduler = BeepScheduler()
  @ObservationIgnored private var latest: (reading: ProximityReading?, at: Date?) = (nil, nil)
  @ObservationIgnored private var ticker: Task<Void, Never>?
  @ObservationIgnored private var observers: [NSObjectProtocol] = []
  @ObservationIgnored private(set) var played: [BeepPlan] = []
  /// The beacon: the item being found plays from where it is, on its own HRTF source.
  @ObservationIgnored private let beaconPlayer = AVAudioPlayerNode()
  @ObservationIgnored private var beaconBuffers: [BeaconSound: AVAudioPCMBuffer] = [:]
  @ObservationIgnored private var beaconScheduler = BeaconScheduler()
  @ObservationIgnored private var latestGuide: (guide: GuideTarget?, found: Bool, at: Date?) =
    (nil, false, nil)
  @ObservationIgnored private(set) var playedBeacon: [BeaconPlan] = []
  /// The server's sound switch (`sound` in metasight.toml [safety]); off silences the beeps
  /// and the beacon, the spoken answers stay.
  var serverSound = true

  init(output: Output = .device, defaults: UserDefaults = .standard) {
    self.output = output
    volume = BeepVolume(rawValue: defaults.string(forKey: BeepVolume.defaultsKey) ?? "") ?? .silent
    switch output {
    case .device:
      canSound = !Self.isSimulatorOrMac
    case .offline:
      canSound = true
    }
    guard canSound else { return }
    for kind in BeepKind.allCases { buffers[kind] = Self.tone(kind) }
    for sound in BeaconSound.allCases { beaconBuffers[sound] = Self.beacon(sound) }
    build()
  }

  private static var isSimulatorOrMac: Bool {
    #if targetEnvironment(simulator)
      return true
    #else
      return ProcessInfo.processInfo.isiOSAppOnMac
    #endif
  }

  // MARK: - Feeding

  /// The latest proximity reading from the server. Starts the beat loop on first use.
  func update(_ reading: ProximityReading?, receivedAt: Date = Date()) {
    latest = (reading, receivedAt)
    startTicker()
  }

  /// The guide from the same safety message (nil when nothing is being looked for) and found.
  func update(guide: GuideTarget?, found: Bool, receivedAt: Date = Date()) {
    latestGuide = (guide, found, receivedAt)
    startTicker()
  }

  private func startTicker() {
    guard canSound, ticker == nil, case .device = output else { return }
    ticker = Task { [weak self] in
      while !Task.isCancelled {
        self?.tick(now: Date())
        try? await Task.sleep(for: .milliseconds(40))
      }
    }
  }

  func stop() {
    ticker?.cancel()
    ticker = nil
    player.stop()
    beaconPlayer.stop()
    if case .device = output, engine.isRunning { engine.pause() }
  }

  /// One step of the beat loop; exposed so tests can drive time.
  func tick(now: Date) {
    if let plan = beaconScheduler.next(
      guide: latestGuide.guide, found: latestGuide.found, receivedAt: latestGuide.at, now: now,
      volume: volume)
    {
      play(plan)
    }
    guard let plan = scheduler.next(
      reading: latest.reading, receivedAt: latest.at, now: now, volume: volume)
    else { return }
    play(plan)
  }

  func play(_ plan: BeaconPlan) {
    guard canSound, serverSound, volume != .silent, let buffer = beaconBuffers[plan.sound]
    else { return }
    playedBeacon.append(plan)
    if playedBeacon.count > 64 { playedBeacon.removeFirst(32) }
    guard startEngine() else { return }
    beaconPlayer.position = AVAudio3DPoint(x: plan.x, y: plan.y, z: plan.z)
    beaconPlayer.volume = plan.gain
    beaconPlayer.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    if !beaconPlayer.isPlaying { beaconPlayer.play() }
  }

  func play(_ plan: BeepPlan) {
    guard canSound, serverSound, volume != .silent, let buffer = buffers[plan.kind] else { return }
    played.append(plan)
    if played.count > 64 { played.removeFirst(32) }
    guard startEngine() else { return }
    player.position = AVAudio3DPoint(x: plan.x, y: plan.y, z: plan.z)
    player.volume = plan.gain
    player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    if !player.isPlaying { player.play() }
  }

  // MARK: - Offline rendering (tests)

  /// Renders `frames` of whatever has been scheduled into a stereo buffer. Offline output only.
  func renderOffline(frames: AVAudioFrameCount) -> AVAudioPCMBuffer? {
    guard case .offline = output, startEngine(),
      let out = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: frames)
    else { return nil }
    guard (try? engine.renderOffline(frames, to: out)) != nil else { return nil }
    return out
  }

  // MARK: - Engine

  private func build() {
    let mono = AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 1)
    let stereo = AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 2)
    if case .offline(let maximum) = output, let stereo {
      try? engine.enableManualRenderingMode(.offline, format: stereo, maximumFrameCount: maximum)
    }
    engine.attach(environment)
    for node in [player, beaconPlayer] {
      engine.attach(node)
      node.renderingAlgorithm = .HRTFHQ
      node.sourceMode = .pointSource
      engine.connect(node, to: environment, format: mono)
    }
    engine.connect(environment, to: engine.mainMixerNode, format: stereo)
    environment.listenerPosition = AVAudio3DPoint(x: 0, y: 0, z: 0)
    environment.distanceAttenuationParameters.referenceDistance = Float(BeepCue.placementMetres)

    if case .offline = output {
      engine.prepare()
      return
    }
    engine.prepare()
    let center = NotificationCenter.default
    // A route change (glasses connect or drop) stops the engine; bring it back.
    observers.append(
      center.addObserver(
        forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
      ) { [weak self] _ in
        MainActor.assumeIsolated { _ = self?.startEngine() }
      })
    #if os(iOS)
    observers.append(
      center.addObserver(
        forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
      ) { [weak self] note in
        let ended =
          (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt)
          == AVAudioSession.InterruptionType.ended.rawValue
        MainActor.assumeIsolated { if ended { _ = self?.startEngine() } }
      })
    #endif
  }

  @discardableResult
  private func startEngine() -> Bool {
    if engine.isRunning { return true }
    #if os(iOS)
      if case .device = output { configureSession() }
    #endif
    do {
      try engine.start()
      return true
    } catch {
      return false
    }
  }

  /// Playback that mixes: the glasses' stream and the answer speaker keep running. If the
  /// answer speaker already set a mixing playback category (it ducks others), leave it be.
  #if os(iOS)
  private func configureSession() {
    let session = AVAudioSession.sharedInstance()
    let mixes = session.categoryOptions.contains(.mixWithOthers)
      || session.categoryOptions.contains(.duckOthers)
    if session.category == .playback, mixes { return }
    try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
    try? session.setActive(true)
  }
  #endif

  // MARK: - Sounds

  /// Each beep carries the octave below at half strength (spatial_sound.SUB_OCTAVE): a pure
  /// 1320 or 2000 Hz tone gives the ear an interaural delay it cannot place, so the stop beep
  /// was heard far off and often on the wrong side; 1.368 keeps the pair's peak at 1.
  static let subOctave = 0.5
  static let subPeak = 1.368

  /// A short sine and its octave below, with a raised-cosine attack and release, so it never
  /// clicks.
  static func tone(_ kind: BeepKind) -> AVAudioPCMBuffer? {
    let count = AVAudioFrameCount(kind.duration * sampleRate)
    guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count),
      let samples = buffer.floatChannelData?[0]
    else { return nil }
    buffer.frameLength = count
    let ramp = max(1, Int(Double(count) * 0.25))
    for i in 0..<Int(count) {
      let edge = min(i, Int(count) - 1 - i)
      let envelope = edge >= ramp ? 1.0 : 0.5 * (1 - cos(.pi * Double(edge) / Double(ramp)))
      let phase = 2 * .pi * kind.frequency * Double(i) / sampleRate
      let tone = (sin(phase) + subOctave * sin(phase / 2)) / subPeak
      samples[i] = Float(tone * envelope)
    }
    return buffer
  }
}

extension SpatialBeeper {
  /// The beacon's mallet note (spatial_sound._pluck): the fundamental plus a quiet fourth
  /// partial, a 4 ms attack, a 90 ms decay and a short fade so the cut never clicks.
  static func pluck(_ hz: Double, duration: Double = 0.32) -> [Float] {
    let n = Int(duration * sampleRate)
    return (0..<n).map { i in
      let t = Double(i) / sampleRate
      let tone = sin(2 * .pi * hz * t) + 0.25 * sin(2 * .pi * 4 * hz * t) * exp(-t / 0.03)
      let fade = min(1, (duration - t) / (0.1 * duration))
      return Float(tone * min(1, t / 0.004) * exp(-t / 0.09) * fade)
    }
  }

  static func beacon(_ sound: BeaconSound) -> AVAudioPCMBuffer? {
    let hz = BeaconCue.hz
    var samples: [Float]
    switch sound {
    case .pluck:
      samples = pluck(hz)
    case .pair:
      let gap = Int(0.11 * sampleRate)
      let (a, b) = (pluck(hz), pluck(hz * 1.5))
      samples = [Float](repeating: 0, count: gap + b.count)
      for (i, v) in a.enumerated() { samples[i] += v }
      for (i, v) in b.enumerated() { samples[gap + i] += 0.8 * v }
    case .arrive:
      let gap = Int(0.09 * sampleRate)
      let notes = [1.0, 1.26, 1.5].map { pluck(hz * $0, duration: 0.45) }
      samples = [Float](repeating: 0, count: 2 * gap + notes[2].count)
      for (k, note) in notes.enumerated() {
        for (i, v) in note.enumerated() { samples[k * gap + i] += 0.8 * v }
      }
    }
    guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
      let out = buffer.floatChannelData?[0]
    else { return nil }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    for (i, v) in samples.enumerated() { out[i] = v }
    return buffer
  }
}

// MARK: - Find my X, on the phone's haptics

/// Plays FindHaptic on the phone's Taptic Engine: one looping continuous event whose intensity
/// and sharpness controls are set 60 times a second, so the pulses speed up smoothly as the
/// target gets closer. A short double tap when the server says it is found. The chip on the
/// camera screen reads `target` and `dist`. No Taptic Engine on a simulator or a Mac: there the
/// chip still shows and nothing buzzes.
@MainActor
@Observable
final class FindHaptics {
  private(set) var target: String?
  private(set) var dist: Double?
  /// True while the buzz is running (the target is in view with a distance).
  private(set) var buzzing = false

  let canBuzz = CHHapticEngine.capabilitiesForHardware().supportsHaptics

  @ObservationIgnored private var engine: CHHapticEngine?
  @ObservationIgnored private var player: CHHapticAdvancedPatternPlayer?
  @ObservationIgnored private var plan: FindHaptic?
  @ObservationIgnored private var started = Date()
  @ObservationIgnored private var ticker: Task<Void, Never>?
  @ObservationIgnored private var wasFound = false

  /// Every safety message: the guide (nil when nothing is being looked for) and `found`.
  func update(guide: GuideTarget?, found: Bool) {
    target = guide?.label ?? (found ? target : nil)
    dist = guide?.dist
    if found, !wasFound { foundTap() }
    wasFound = found
    plan = guide.flatMap { FindHaptic.plan(dist: $0.dist, bearing: $0.bearing, inView: $0.inView) }
    if plan != nil, !buzzing { begin() } else if plan == nil, buzzing { end() }
  }

  private func begin() {
    buzzing = true
    started = Date()
    guard canBuzz, startEngine() else { return }
    let event = CHHapticEvent(
      eventType: .hapticContinuous,
      parameters: [
        CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
        CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5),
      ],
      relativeTime: 0, duration: 30)
    guard let pattern = try? CHHapticPattern(events: [event], parameters: []),
      let player = try? engine?.makeAdvancedPlayer(with: pattern)
    else { return }
    player.loopEnabled = true
    try? player.start(atTime: CHHapticTimeImmediate)
    self.player = player
    ticker = Task { [weak self] in
      while !Task.isCancelled {
        self?.tick()
        try? await Task.sleep(for: .milliseconds(16))
      }
    }
  }

  private func tick() {
    guard let plan, let player else { return }
    let level = plan.level(at: Date().timeIntervalSince(started))
    try? player.sendParameters(
      [
        CHHapticDynamicParameter(parameterID: .hapticIntensityControl, value: level, relativeTime: 0),
        CHHapticDynamicParameter(
          parameterID: .hapticSharpnessControl, value: plan.sharpness - 0.5, relativeTime: 0),
      ],
      atTime: CHHapticTimeImmediate)
  }

  private func end() {
    buzzing = false
    ticker?.cancel()
    ticker = nil
    try? player?.stop(atTime: CHHapticTimeImmediate)
    player = nil
  }

  private func foundTap() {
    end()
    guard canBuzz, startEngine() else { return }
    let taps = [0.0, 0.14].map {
      CHHapticEvent(
        eventType: .hapticTransient,
        parameters: [
          CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
          CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.9),
        ],
        relativeTime: $0)
    }
    guard let pattern = try? CHHapticPattern(events: taps, parameters: []) else { return }
    try? engine?.makePlayer(with: pattern).start(atTime: CHHapticTimeImmediate)
  }

  /// The engine went away under us: drop it and its player, and let the next update restart.
  private func lost() {
    end()
    engine = nil
  }

  private func startEngine() -> Bool {
    if engine == nil {
      engine = try? CHHapticEngine()
      // The system stops the engine when the app goes to the background; start it again lazily.
      engine?.resetHandler = { [weak self] in Task { @MainActor in self?.lost() } }
      engine?.stoppedHandler = { [weak self] _ in Task { @MainActor in self?.lost() } }
    }
    guard let engine else { return false }
    return (try? engine.start()) != nil
  }
}
