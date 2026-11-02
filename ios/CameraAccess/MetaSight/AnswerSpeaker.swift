//
// AnswerSpeaker.swift
//
// Speaks the server's answers ("Your water bottle is on the desk, two steps ahead").
// Audio goes through the current output route, which is the glasses' open-ear
// speakers when they are connected over Bluetooth.
//

import AVFoundation

@MainActor
final class AnswerSpeaker: NSObject, AVSpeechSynthesizerDelegate {
  private let synthesizer = AVSpeechSynthesizer()
  private var sessionConfigured = false
  /// Called with each range of the text as it is about to be spoken (the roll call's plucks).
  private var onRange: ((NSRange) -> Void)?

  override init() {
    super.init()
    synthesizer.delegate = self
  }

  func speak(_ text: String, onRange: ((NSRange) -> Void)? = nil) {
    self.onRange = onRange
    // Like the beeper, nothing plays on a simulator or a Mac; the answer is still on screen.
    guard !Self.isSimulatorOrMac else { return }
    configureSessionIfNeeded()
    // A new answer replaces whatever is still being read out.
    if synthesizer.isSpeaking {
      synthesizer.stopSpeaking(at: .immediate)
    }
    let utterance = AVSpeechUtterance(string: text)
    utterance.rate = AVSpeechUtteranceDefaultSpeechRate
    synthesizer.speak(utterance)
  }

  nonisolated func speechSynthesizer(
    _ synthesizer: AVSpeechSynthesizer, willSpeakRangeOfSpeechString characterRange: NSRange,
    utterance: AVSpeechUtterance
  ) {
    MainActor.assumeIsolated { onRange?(characterRange) }
  }

  private static var isSimulatorOrMac: Bool {
    #if targetEnvironment(simulator)
      return true
    #else
      return ProcessInfo.processInfo.isiOSAppOnMac
    #endif
  }

  private func configureSessionIfNeeded() {
    guard !sessionConfigured else { return }
    let session = AVAudioSession.sharedInstance()
    do {
      // The playback category routes to Bluetooth A2DP outputs (the glasses) by
      // default; `.allowBluetoothA2DP` only applies to play-and-record.
      try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
      try session.setActive(true)
      sessionConfigured = true
    } catch {
      // Leave unconfigured; the synthesizer still plays on the default route and
      // the next answer retries.
    }
  }
}
