//
// WatchLink.swift
//
// Forwards the server's latest sector distances to the paired Apple Watch, which turns
// them into haptics. Live messages while the watch app is reachable (up to 5 a second,
// sooner when the band changes); otherwise the application context, so the watch has
// the latest reading as soon as it opens.
//

import Foundation
import Observation
import WatchConnectivity

@Observable
@MainActor
final class WatchLink: NSObject {
  static let shared = WatchLink()

  private static let liveInterval: TimeInterval = 0.2
  private static let contextInterval: TimeInterval = 1.0

  /// Shown on the phone so a VoiceOver user knows whether the watch is listening.
  private(set) var status: String = "no watch"

  @ObservationIgnored private var lastLive: Date = .distantPast
  @ObservationIgnored private var lastContext: Date = .distantPast
  @ObservationIgnored private var lastBand: ProximityBand = .clear

  private var session: WCSession? { WCSession.isSupported() ? WCSession.default : nil }

  func activate() {
    guard let session, session.delegate == nil else { return }
    session.delegate = self
    session.activate()
  }

  func send(_ reading: ProximityReading) {
    guard let session, session.activationState == .activated, session.isWatchAppInstalled else { return }
    let now = Date()
    let band = reading.warning?.band ?? .clear
    let bandChanged = band != lastBand
    lastBand = band
    if session.isReachable {
      guard bandChanged || now.timeIntervalSince(lastLive) >= Self.liveInterval else { return }
      lastLive = now
      session.sendMessage(reading.message, replyHandler: nil, errorHandler: nil)
    } else if bandChanged || now.timeIntervalSince(lastContext) >= Self.contextInterval {
      lastContext = now
      try? session.updateApplicationContext(reading.message)
    }
  }

  private func refreshStatus() {
    guard let session else {
      status = "no watch"
      return
    }
    if session.activationState != .activated || !session.isPaired {
      status = "no watch"
    } else if !session.isWatchAppInstalled {
      status = "watch app not installed"
    } else {
      status = session.isReachable ? "watch connected" : "watch app closed"
    }
  }
}

extension WatchLink: WCSessionDelegate {
  nonisolated func session(_ session: WCSession, activationDidCompleteWith state: WCSessionActivationState, error: Error?) {
    Task { @MainActor in self.refreshStatus() }
  }

  nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
    Task { @MainActor in self.refreshStatus() }
  }

  nonisolated func sessionWatchStateDidChange(_ session: WCSession) {
    Task { @MainActor in self.refreshStatus() }
  }

  nonisolated func sessionDidBecomeInactive(_ session: WCSession) {}

  nonisolated func sessionDidDeactivate(_ session: WCSession) {
    // A different watch was paired: start talking to it.
    session.activate()
  }
}
