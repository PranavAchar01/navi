//
// WatchModel.swift
//
// Receives readings from the iPhone over WatchConnectivity and plays the haptics.
//
// Patterns (see ProximityPacer for the timing):
//   ahead        directionUp, two rising taps
//   left, right  click, one crisp tap
//   behind       directionDown, two falling taps
//   stop band    failure, the strongest pattern (closer than 0.5 m, or walking into something)
//   signal lost  retry, once, so silence is never mistaken for a clear path
//   find my X    click, faster as the asked-for item gets closer (only with nothing to avoid)
//
// While on, an extended runtime session (the "mindfulness" type, which needs no paid
// entitlement) keeps the app frontmost and playing haptics with the screen dimmed, for up
// to an hour per start.
//

import Foundation
import Observation
import WatchConnectivity
import WatchKit
import os

@Observable
@MainActor
final class WatchModel: NSObject {
  private(set) var isOn = false
  private(set) var reading: ProximityReading?
  /// The reading is recent enough to trust.
  private(set) var isFresh = false
  /// Why the screen-off session ended, when it did.
  private(set) var note: String?

  @ObservationIgnored private var receivedAt: Date?
  @ObservationIgnored private var pacer = ProximityPacer()
  @ObservationIgnored private var timer: Timer?
  @ObservationIgnored private var runtime: WKExtendedRuntimeSession?
  @ObservationIgnored private let log = Logger(subsystem: "com.pranavachar.metasight.watch", category: "haptics")

  var warning: ProximityWarning? { isFresh ? reading?.warning : nil }

  override init() {
    super.init()
    if WCSession.isSupported() {
      WCSession.default.delegate = self
      WCSession.default.activate()
    }
  }

  func toggle() {
    if isOn { stop() } else { start() }
  }

  func start() {
    guard !isOn else { return }
    isOn = true
    note = nil
    pacer = ProximityPacer()
    WKInterfaceDevice.current().play(.start)
    log.notice("warnings on")
    timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
      MainActor.assumeIsolated { self?.tick() }
    }
    let session = WKExtendedRuntimeSession()
    session.delegate = self
    session.start()
    runtime = session
  }

  func stop() {
    guard isOn else { return }
    isOn = false
    timer?.invalidate()
    timer = nil
    runtime?.invalidate()
    runtime = nil
    WKInterfaceDevice.current().play(.stop)
    log.notice("warnings off")
  }

  private func tick() {
    let now = Date()
    let fresh = receivedAt.map { now.timeIntervalSince($0) < ProximityPacer.staleAfter } ?? false
    if fresh != isFresh { isFresh = fresh }
    guard let pulse = pacer.pulse(reading: reading, receivedAt: receivedAt, now: now) else { return }
    WKInterfaceDevice.current().play(Self.haptic(for: pulse))
    let w = reading?.warning
    log.notice(
      "haptic \(pulse.rawValue, privacy: .public) \(w?.directionName ?? "-", privacy: .public) \(w?.distanceText ?? "-", privacy: .public)"
    )
  }

  static func haptic(for pulse: ProximityPulse) -> WKHapticType {
    switch pulse {
    case .ahead: return .directionUp
    case .side: return .click
    case .behind: return .directionDown
    case .stop: return .failure
    case .lost: return .retry
    case .find: return .click
    }
  }

  fileprivate func receive(_ message: [String: Any]) {
    guard let r = ProximityReading(message: message) else { return }
    reading = r
    receivedAt = Date()
    if !isFresh { isFresh = true }
  }

  fileprivate func runtimeEnded(_ text: String?) {
    runtime = nil
    if isOn { note = text }
  }
}

extension WatchModel: WCSessionDelegate {
  nonisolated func session(_ session: WCSession, activationDidCompleteWith state: WCSessionActivationState, error: Error?) {
    let context = session.receivedApplicationContext
    guard !context.isEmpty else { return }
    let reading = ProximityReading(message: context)
    Task { @MainActor in if let reading { self.receive(reading.message) } }
  }

  nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
    guard let reading = ProximityReading(message: message) else { return }
    Task { @MainActor in self.receive(reading.message) }
  }

  nonisolated func session(_ session: WCSession, didReceiveApplicationContext context: [String: Any]) {
    guard let reading = ProximityReading(message: context) else { return }
    Task { @MainActor in self.receive(reading.message) }
  }
}

extension WatchModel: WKExtendedRuntimeSessionDelegate {
  nonisolated func extendedRuntimeSessionDidStart(_ session: WKExtendedRuntimeSession) {
    Task { @MainActor in self.log.notice("screen-off session started") }
  }

  nonisolated func extendedRuntimeSessionWillExpire(_ session: WKExtendedRuntimeSession) {
    Task { @MainActor in self.note = "Screen-off time almost up. Raise your wrist and press Start again." }
  }

  nonisolated func extendedRuntimeSession(
    _ session: WKExtendedRuntimeSession,
    didInvalidateWith reason: WKExtendedRuntimeSessionInvalidationReason,
    error: Error?
  ) {
    let text: String? =
      switch reason {
      case .none, .sessionInProgress: nil
      case .expired: "Screen-off time ended. Warnings only while the screen is on."
      default: "Warnings work only while the screen is on."
      }
    let detail = error?.localizedDescription ?? "none"
    Task { @MainActor in
      self.log.notice("screen-off session ended: \(reason.rawValue) \(detail, privacy: .public)")
      self.runtimeEnded(text)
    }
  }
}
