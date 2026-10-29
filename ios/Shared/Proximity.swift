//
// Proximity.swift
//
// Shared by the iPhone app and the watch app. The server's "safety" message, cut down to
// what the watch needs (one distance and cue per belt sector), and the parking-sensor
// rules that turn it into haptic pulses.
//
// Bands come from the server's own cue thresholds (metasight/geometry.py): nothing at
// 2 m or more, "slow" from 2 to 1 m, "fast" from 1 to 0.5 m, "solid" (stop) below 0.5 m.
// The server can also raise the front cue a step when you walk at something quickly;
// the watch honours that by taking the stronger of its own band and the server's cue.
//

import Foundation

/// One reading of the eight sectors around the wearer, clockwise from straight ahead.
struct ProximityReading: Equatable, Sendable {
  static let sectorCount = 8

  /// Metres to the nearest obstacle per sector; nil when nothing is within range.
  var distances: [Double?]
  /// The server's cue per sector: off, slow, fast, solid, unknown or guide.
  var modes: [String]
  /// Metres to the "find my X" target while the camera sees it; nil when nothing is being
  /// looked for or it is out of view.
  var find: Double? = nil

  init(distances: [Double?], modes: [String], find: Double? = nil) {
    self.find = find
    self.distances = Array((distances + Array(repeating: nil, count: Self.sectorCount)).prefix(Self.sectorCount))
    self.modes = Array((modes + Array(repeating: "unknown", count: Self.sectorCount)).prefix(Self.sectorCount))
  }

  /// The most urgent sector: the strongest band, then straight ahead before the sides,
  /// then the nearest. Nil when every sector is clear or unknown.
  var warning: ProximityWarning? {
    var best: ProximityWarning?
    for i in 0..<Self.sectorCount where modes[i] != "unknown" {
      let d = distances[i]
      let band = max(ProximityBand(distance: d), ProximityBand(mode: modes[i]))
      guard band > .clear else { continue }
      let w = ProximityWarning(sector: i, distance: d, band: band)
      if let b = best, !w.outranks(b) { continue }
      best = w
    }
    return best
  }

  // MARK: - WatchConnectivity payload

  /// Property-list form for WCSession: "d" distances (-1 for none), "m" cue names, "f" the
  /// find target's distance when there is one.
  var message: [String: Any] {
    var m: [String: Any] = ["d": distances.map { $0 ?? -1 }, "m": modes]
    if let find { m["f"] = find }
    return m
  }

  init?(message: [String: Any]) {
    guard let d = message["d"] as? [Double], let m = message["m"] as? [String] else { return nil }
    self.init(distances: d.map { $0 < 0 ? nil : $0 }, modes: m, find: message["f"] as? Double)
  }
}

enum ProximityBand: Int, Comparable, Sendable {
  case clear, slow, fast, stop

  static let offMetres = 2.0
  static let slowMetres = 1.0
  static let fastMetres = 0.5

  init(distance: Double?) {
    guard let d = distance, d.isFinite, d < Self.offMetres else {
      self = .clear
      return
    }
    self = d >= Self.slowMetres ? .slow : (d >= Self.fastMetres ? .fast : .stop)
  }

  /// The server's cue name. "guide" points at a remembered item, not at an obstacle.
  init(mode: String) {
    switch mode {
    case "slow": self = .slow
    case "fast": self = .fast
    case "solid": self = .stop
    default: self = .clear
    }
  }

  static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
}

/// Where the nearest obstacle is, for one pattern and one spoken phrase.
enum ProximityDirection: Sendable {
  case ahead, side, behind

  init(sector: Int) {
    switch sector {
    case 0: self = .ahead
    case 3, 4, 5: self = .behind
    default: self = .side
    }
  }
}

struct ProximityWarning: Equatable, Sendable {
  let sector: Int
  let distance: Double?
  let band: ProximityBand

  var direction: ProximityDirection { ProximityDirection(sector: sector) }

  /// "Ahead", "Ahead right", "Right", "Behind right", "Behind", "Behind left", "Left", "Ahead left".
  var directionName: String {
    ["Ahead", "Ahead right", "Right", "Behind right", "Behind", "Behind left", "Left", "Ahead left"][sector]
  }

  /// "0.8 m", or "close" when the server raised the cue without a distance.
  var distanceText: String {
    guard let d = distance else { return "close" }
    return String(format: "%.1f m", d)
  }

  /// Full sentence for VoiceOver.
  var spoken: String {
    let lead = band == .stop ? "Stop. " : ""
    guard let d = distance else { return "\(lead)Obstacle \(directionName.lowercased()), close" }
    return "\(lead)Obstacle \(directionName.lowercased()), \(String(format: "%.1f", d)) meters"
  }

  func outranks(_ other: ProximityWarning) -> Bool {
    if band != other.band { return band > other.band }
    let ahead = direction == .ahead, otherAhead = other.direction == .ahead
    if ahead != otherAhead { return ahead }
    return (distance ?? 0) < (other.distance ?? 0)
  }
}

/// A haptic the watch should play now.
enum ProximityPulse: String, Sendable {
  /// Something straight ahead (two rising taps).
  case ahead
  /// Something to the left or right (one light tap).
  case side
  /// Something behind (two falling taps).
  case behind
  /// Closer than half a metre, or walking into something: the strongest pattern.
  case stop
  /// Readings stopped arriving: played once so silence is never mistaken for "clear".
  case lost
  /// The "find my X" target is in view: taps that come faster as you get closer.
  case find
}

/// Parking-sensor timing: silent when clear, pulses that come faster as the obstacle gets
/// closer, never a continuous buzz.
struct ProximityPacer: Sendable {
  /// No two pulses closer together than this.
  static let minimumGap: TimeInterval = 0.3
  /// A reading older than this counts as no signal.
  static let staleAfter: TimeInterval = 1.5
  /// An obstacle that has not moved for this long (you stopped beside a wall) pulses less often.
  static let settleAfter: TimeInterval = 4
  static let settledFactor = 2.5

  private var lastPulse: Date = .distantPast
  private var hadSignal = false
  private var settle: (sector: Int, band: ProximityBand, distance: Double, since: Date)?

  init() {}

  /// Seconds between pulses for a band at a distance: slow 1.5 s at 2 m down to 0.8 s at
  /// 1 m, fast 0.6 s at 1 m down to 0.35 s at 0.5 m, stop every 0.7 s.
  static func interval(for warning: ProximityWarning) -> TimeInterval {
    let d = warning.distance
    switch warning.band {
    case .clear: return .infinity
    case .slow: return lerp(d, from: (2.0, 1.5), to: (1.0, 0.8))
    case .fast: return lerp(d ?? 0.5, from: (1.0, 0.6), to: (0.5, 0.35))
    case .stop: return 0.7
    }
  }

  private static func lerp(_ d: Double?, from a: (Double, Double), to b: (Double, Double)) -> TimeInterval {
    guard let d else { return b.1 }
    let t = min(1, max(0, (a.0 - d) / (a.0 - b.0)))
    return a.1 + (b.1 - a.1) * t
  }

  /// What to play at `now`, given the latest reading and when it arrived.
  mutating func pulse(reading: ProximityReading?, receivedAt: Date?, now: Date) -> ProximityPulse? {
    guard let reading, let receivedAt, now.timeIntervalSince(receivedAt) < Self.staleAfter else {
      settle = nil
      guard hadSignal else { return nil }
      hadSignal = false
      lastPulse = now
      return .lost
    }
    hadSignal = true
    guard let w = reading.warning else {
      settle = nil
      // Nothing to avoid: the find taps, as fast as the watch can tap cleanly.
      guard let plan = FindHaptic.plan(dist: reading.find, bearing: 0, inView: true),
        now.timeIntervalSince(lastPulse) >= max(1 / plan.pulseHz, Self.minimumGap)
      else { return nil }
      lastPulse = now
      return .find
    }

    var interval = Self.interval(for: w)
    let d = w.distance ?? 0
    if let s = settle, s.sector == w.sector, s.band == w.band, abs(s.distance - d) < 0.15 {
      if w.band != .stop, now.timeIntervalSince(s.since) > Self.settleAfter {
        interval *= Self.settledFactor
      }
    } else {
      settle = (w.sector, w.band, d, now)
    }

    guard now.timeIntervalSince(lastPulse) >= max(interval, Self.minimumGap) else { return nil }
    lastPulse = now
    if w.band == .stop { return .stop }
    switch w.direction {
    case .ahead: return .ahead
    case .side: return .side
    case .behind: return .behind
    }
  }
}

/// "Find my X" on the phone's (and watch's) haptics: one continuous buzz that gets stronger, sharper and
/// pulses faster as the target gets closer, from the server's guide distance. Silent whenever
/// the target is not in view, so a buzz always means "it is in front of you right now".
struct FindHaptic: Equatable, Sendable {
  /// 0...1, Core Haptics' intensity and sharpness controls.
  let intensity: Float
  let sharpness: Float
  /// On-off pulses per second.
  let pulseHz: Double

  /// The server calls the item found, and stops guiding, inside `near`.
  static let near = 0.8
  /// From here out the buzz is at its faintest and slowest.
  static let far = 4.0
  /// The camera's field of view when nothing better is known (the glasses are about 70 deg).
  static let defaultHFov = 70.0

  /// `inView` is the server's word when it sends one; otherwise the target counts as in view
  /// when its bearing is inside the camera's field of view.
  static func plan(dist: Double?, bearing: Double?, inView: Bool?, hfov: Double = defaultHFov)
    -> FindHaptic?
  {
    guard let dist, let bearing, dist.isFinite else { return nil }
    guard inView ?? (abs(bearing) <= hfov / 2) else { return nil }
    let t = min(1, max(0, (far - dist) / (far - near)))
    return FindHaptic(
      intensity: Float(0.3 + 0.7 * t),
      sharpness: Float(0.15 + 0.75 * t),
      pulseHz: 1.5 + 8.5 * pow(t, 1.5))
  }

  /// The intensity at `seconds` into the buzz: full for the first half of each pulse, a faint
  /// floor for the second, so it reads as pulses without ever stopping dead.
  func level(at seconds: Double) -> Float {
    let phase = (seconds * pulseHz).truncatingRemainder(dividingBy: 1)
    return phase < 0.5 ? intensity : intensity * 0.12
  }
}
