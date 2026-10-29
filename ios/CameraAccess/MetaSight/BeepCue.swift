//
// BeepCue.swift
//
// The wearer's spatial beeps, as pure rules: which proximity band sounds like what, how often
// it repeats, and where in 3D the beep is placed. The audio engine that plays them is
// SpatialBeeper; nothing here touches AVFoundation, so every rule is unit tested. The phone's
// "find my X" buzz rule, FindHaptic, lives in Shared/Proximity.swift so the watch uses it too.
//
// Bands are the server's (metasight/geometry.py) via ProximityBand: silent at 2 m or more,
// yellow 2 to 1 m, orange 1 to 0.5 m, red below 0.5 m or when the server says "solid"
// (time to collision under 1.2 s).
//

import Foundation

/// How loud the beeps are. Some wearers have very sensitive hearing, so quiet is a real setting.
enum BeepVolume: String, CaseIterable, Identifiable, Sendable {
  case quiet, normal

  var id: String { rawValue }
  var label: String { rawValue.capitalized }
  /// Applied on top of each cue's own gain.
  var scale: Float { self == .quiet ? 0.3 : 1.0 }

  static let defaultsKey = "metasight.beepVolume"
}

/// The three sounds, faintest to clearest.
enum BeepKind: Int, CaseIterable, Sendable {
  /// Yellow: a faint, very short tick.
  case tick
  /// Orange: a soft beep.
  case soft
  /// Red: a clear beep.
  case clear

  var frequency: Double {
    switch self {
    case .tick: 2000
    case .soft: 880
    case .clear: 1320
    }
  }

  var duration: TimeInterval {
    switch self {
    case .tick: 0.012
    case .soft: 0.07
    case .clear: 0.09
    }
  }

  /// Before the volume setting. Tick is deliberately barely there.
  var gain: Float {
    switch self {
    case .tick: 0.15
    case .soft: 0.4
    case .clear: 0.8
    }
  }
}

/// One beep to play now: what, how loud, and where in space (metres around the head,
/// AVAudio3DPoint convention: x right, y up, z backwards, so ahead is negative z).
struct BeepPlan: Equatable, Sendable {
  let kind: BeepKind
  let gain: Float
  let x: Float
  let y: Float
  let z: Float
  let sector: Int
  let band: ProximityBand
}

enum BeepCue {
  /// Beeps sit this far from the head. The distance to the obstacle is carried by the
  /// repeat rate, not by loudness, so a quiet setting stays quiet.
  static let placementMetres = 1.0

  static func kind(for band: ProximityBand) -> BeepKind? {
    switch band {
    case .clear: nil
    case .slow: .tick
    case .fast: .soft
    case .stop: .clear
    }
  }

  /// Seconds between beeps: yellow is sparse (2.0 s at 2 m to 1.2 s at 1 m), orange faster
  /// (0.7 s to 0.4 s), red rapid (5 a second).
  static func interval(for warning: ProximityWarning) -> TimeInterval {
    func lerp(_ d: Double?, _ a: (Double, Double), _ b: (Double, Double)) -> TimeInterval {
      guard let d else { return b.1 }
      let t = min(1, max(0, (a.0 - d) / (a.0 - b.0)))
      return a.1 + (b.1 - a.1) * t
    }
    switch warning.band {
    case .clear: return .infinity
    case .slow: return lerp(warning.distance, (2.0, 2.0), (1.0, 1.2))
    case .fast: return lerp(warning.distance, (1.0, 0.7), (0.5, 0.4))
    case .stop: return 0.2
    }
  }

  /// Compass bearing of a sector, degrees clockwise from straight ahead (eight sectors).
  static func bearing(sector: Int) -> Double { Double(sector % ProximityReading.sectorCount) * 45 }

  static func plan(for warning: ProximityWarning, volume: BeepVolume) -> BeepPlan? {
    guard let kind = kind(for: warning.band) else { return nil }
    let radians = bearing(sector: warning.sector) * .pi / 180
    let r = placementMetres
    return BeepPlan(
      kind: kind,
      gain: kind.gain * volume.scale,
      x: Float(sin(radians) * r),
      y: 0,
      z: Float(-cos(radians) * r),
      sector: warning.sector,
      band: warning.band
    )
  }
}

/// Decides when the next beep is due. Silent when clear or when readings stop arriving
/// (a stale reading must never keep beeping at an obstacle that may be gone).
struct BeepScheduler: Sendable {
  static let staleAfter: TimeInterval = 1.5

  private var last: (at: Date, sector: Int, band: ProximityBand)?

  init() {}

  mutating func next(
    reading: ProximityReading?, receivedAt: Date?, now: Date, volume: BeepVolume
  ) -> BeepPlan? {
    guard let reading, let receivedAt, now.timeIntervalSince(receivedAt) < Self.staleAfter,
      let warning = reading.warning
    else {
      last = nil
      return nil
    }
    // A new or stronger warning sounds at once; a repeat waits out its interval.
    if let l = last, l.sector == warning.sector, l.band >= warning.band,
      now.timeIntervalSince(l.at) < BeepCue.interval(for: warning)
    {
      return nil
    }
    guard let plan = BeepCue.plan(for: warning, volume: volume) else { return nil }
    last = (now, warning.sector, warning.band)
    return plan
  }
}

