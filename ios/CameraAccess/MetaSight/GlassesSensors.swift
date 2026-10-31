//
// GlassesSensors.swift
//
// Attaches the glasses' Motion and Speech capabilities to a started DeviceSession
// and forwards them to the server: accelerometer samples as motion batches, spoken
// questions as "ask" messages.
//

import MWDATCore
import MWDATMotion
import MWDATSpeech
import Observation

@Observable
@MainActor
final class GlassesSensors {
  // MARK: - Observed state

  private(set) var motionStatus: String = "off"
  private(set) var voiceStatus: String = "off"
  /// Microphone permission isn't granted yet; the UI offers a button that
  /// requests it (this switches to the Meta AI app).
  private(set) var voiceNeedsPermission: Bool = false
  /// Latest recognized speech, partial or final.
  private(set) var heard: String = ""

  // MARK: - Private

  @ObservationIgnored private let wearables: WearablesInterface
  @ObservationIgnored private let link: ServerLink
  @ObservationIgnored private weak var session: DeviceSession?
  @ObservationIgnored private var motion: Motion?
  @ObservationIgnored private var motionTask: Task<Void, Never>?
  @ObservationIgnored private let motionTokens = ListenerTokenBag()
  @ObservationIgnored private var speech: Speech?
  @ObservationIgnored private let speechTokens = ListenerTokenBag()

  init(wearables: WearablesInterface, link: ServerLink) {
    self.wearables = wearables
    self.link = link
  }

  // MARK: - Lifecycle

  /// Call once the session reaches `.started`. Safe to call again after a pause.
  func attach(to session: DeviceSession) {
    guard self.session !== session else { return }
    self.session = session
    attachMotion(to: session)
    Task { await attachSpeechIfPermitted() }
  }

  /// Call when the session stops. The SDK tears the capabilities down with the
  /// session; this drops our references and listeners.
  func detach() {
    motionTask?.cancel()
    motionTask = nil
    motionTokens.clear()
    motion = nil
    speechTokens.clear()
    speech = nil
    session = nil
    motionStatus = "off"
    voiceStatus = "off"
  }

  /// User-initiated: requests microphone permission (redirects to Meta AI and
  /// back), then starts speech recognition.
  func requestVoicePermission() async {
    do {
      guard try await wearables.requestPermission(.microphone) == .granted else {
        voiceStatus = "microphone permission denied"
        return
      }
    } catch {
      voiceStatus = error.localizedDescription
      return
    }
    voiceNeedsPermission = false
    startSpeech()
  }

  // MARK: - Motion

  private func attachMotion(to session: DeviceSession) {
    guard motion == nil else { return }
    do {
      guard let newMotion = try session.addMotion(configuration: MotionConfiguration(samplingRate: .hz30)) else {
        motionStatus = "unavailable"
        return
      }
      motion = newMotion
      newMotion.statePublisher.listen { [weak self] state in
        Task { @MainActor in self?.motionStatus = state.description }
      }.store(in: motionTokens)
      newMotion.errorPublisher.listen { [weak self] error in
        Task { @MainActor in self?.motionStatus = error.localizedDescription }
      }.store(in: motionTokens)

      // Observers are registered before start() so no samples are missed.
      let samples = newMotion.samples
      let link = self.link
      motionTask = Task {
        for await sample in samples {
          guard let a = sample.accelerometer else { continue }
          let t = Double(sample.timestampNs) / Self.timestampNsPerUnit
          let acc = Self.serverAxes(a)
          var row = [t, acc.x, acc.y, acc.z]
          if let g = sample.gyroscope {
            let gyro = Self.serverAxes(g)
            row += [gyro.x, gyro.y, gyro.z]
          }
          link.addMotion(row)
        }
      }
      newMotion.start()
    } catch {
      motionStatus = error.localizedDescription
    }
  }

  /// Motion timestamp unit. The server's `motion.Sample.t` is seconds (its fall and
  /// shake windows are in seconds) and the browser client sends performance-clock
  /// seconds, so the glasses' monotonic nanoseconds are converted to seconds.
  /// Use 1_000_000 here if the server switches to milliseconds.
  nonisolated private static let timestampNsPerUnit: Double = 1_000_000_000

  /// Maps a glasses motion vector (accelerometer m/s^2 or gyroscope rad/s) into the
  /// server's device axes: x right, y up, z out of the screen (toward the wearer's
  /// face). The same mapping is used for both sensors so gyro rotation and
  /// gravity stay in one frame.
  ///
  /// Current mapping, identity: server x = DAT x, server y = DAT y, server z = DAT z.
  ///
  /// UNVERIFIED. The DAT docs (motion SKILL.md, MotionSample reference and the
  /// docs search) give units only: accelerometer m/s^2 with gravity included,
  /// "approximately 9.81 on the axis opposing gravity" (so the sign convention
  /// matches the browser's `accelerationIncludingGravity`), gyroscope rad/s. They
  /// do not state which physical direction each glasses axis points. Check on real
  /// glasses: standing still and looking straight ahead should read about
  /// (0, +9.8, 0); tilting the head down should move gravity into z; turning the
  /// head left should give a positive gy (right-handed rotation about y up). Swap
  /// or negate components here if they don't.
  nonisolated private static func serverAxes(_ v: Vector3) -> (x: Double, y: Double, z: Double) {
    (Double(v.x), Double(v.y), Double(v.z))
  }

  // MARK: - Speech

  private func attachSpeechIfPermitted() async {
    guard speech == nil else { return }
    do {
      if try await wearables.checkPermissionStatus(.microphone) == .granted {
        voiceNeedsPermission = false
        startSpeech()
      } else {
        voiceNeedsPermission = true
        voiceStatus = "needs microphone permission"
      }
    } catch {
      voiceStatus = error.localizedDescription
    }
  }

  private func startSpeech() {
    guard speech == nil, let session, session.state == .started else { return }
    do {
      guard let newSpeech = try session.addSpeech() else {
        voiceStatus = "unavailable"
        return
      }
      speech = newSpeech
      // Listeners go in before start() so initial state and results are not missed.
      newSpeech.statePublisher.listen { [weak self] state in
        Task { @MainActor in self?.voiceStatus = state.description }
      }.store(in: speechTokens)
      newSpeech.errorPublisher.listen { [weak self] error in
        Task { @MainActor in self?.voiceStatus = error.localizedDescription }
      }.store(in: speechTokens)
      newSpeech.transcriptionPublisher.listen { [weak self] result in
        Task { @MainActor in self?.handleTranscription(result) }
      }.store(in: speechTokens)
      newSpeech.start()
    } catch {
      voiceStatus = error.localizedDescription
    }
  }

  private func handleTranscription(_ result: TranscriptionResult) {
    heard = result.text
    guard result.isFinal, Self.isQuestion(result.text) else { return }
    link.ask(result.text)
  }

  /// Final transcripts that read as a question about the room go to the server.
  static func isQuestion(_ text: String) -> Bool {
    let s = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    guard !s.isEmpty else { return false }
    if s.hasSuffix("?") { return true }
    let starters = [
      "where", "what", "which", "how far", "how do i get", "is there", "are there", "is my", "find",
      "can you find", "have you seen", "do you see", "take me to", "guide me to",
    ]
    return starters.contains { s.hasPrefix($0) } || s.contains("where is") || s.contains("where's")
      || s.contains("where are") || s.contains("where did")
  }
}
