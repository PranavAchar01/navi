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

import AVFoundation
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

  /// Persisted. Quiet by default would hide the cue entirely for most people, but it is the
  /// safe start for the most sensitive; the picker is on the camera screen.
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

  init(output: Output = .device, defaults: UserDefaults = .standard) {
    self.output = output
    volume = BeepVolume(rawValue: defaults.string(forKey: BeepVolume.defaultsKey) ?? "") ?? .normal
    switch output {
    case .device:
      canSound = !Self.isSimulatorOrMac
    case .offline:
      canSound = true
    }
    guard canSound else { return }
    for kind in BeepKind.allCases { buffers[kind] = Self.tone(kind) }
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
    if case .device = output, engine.isRunning { engine.pause() }
  }

  /// One step of the beat loop; exposed so tests can drive time.
  func tick(now: Date) {
    guard let plan = scheduler.next(
      reading: latest.reading, receivedAt: latest.at, now: now, volume: volume)
    else { return }
    play(plan)
  }

  func play(_ plan: BeepPlan) {
    guard canSound, let buffer = buffers[plan.kind] else { return }
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
    engine.attach(player)
    player.renderingAlgorithm = .HRTFHQ
    player.sourceMode = .pointSource
    engine.connect(player, to: environment, format: mono)
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

  /// A short sine with a raised-cosine attack and release, so it never clicks.
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
      samples[i] = Float(sin(2 * .pi * kind.frequency * Double(i) / sampleRate) * envelope)
    }
    return buffer
  }
}
