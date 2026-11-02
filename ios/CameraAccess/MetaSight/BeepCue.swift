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
/// Silent also mutes the spoken prompts, for quiet mapping walks.
enum BeepVolume: String, CaseIterable, Identifiable, Sendable {
  case silent, quiet, normal

  var id: String { rawValue }
  var label: String { rawValue.capitalized }
  /// Applied on top of each cue's own gain.
  var scale: Float {
    switch self {
    case .silent: 0
    case .quiet: 0.3
    case .normal: 1.0
    }
  }

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
  /// A warning that holds settles to one beep a second after 3 s (spatial_sound.Repeats): the
  /// belt still holds it, and five stop beeps a second at a desk is noise the wearer turns off.
  static let settleAfter: TimeInterval = 3
  static let settledInterval: TimeInterval = 1

  /// How long a band must hold before it beeps (spatial_sound.RISE_S, tuned on take1 by
  /// scripts/score_ticks.py): a one-frame phantom never sounds, a real warning beeps 0.1 to
  /// 0.3 s later. Shorter than the display's WarningFilter.rise: sound is the fast channel.
  static func rise(_ band: ProximityBand) -> TimeInterval {
    switch band {
    case .stop: 0.1
    case .fast: 0.2
    case .slow, .clear: 0.3
    }
  }
  private static let slack: TimeInterval = 0.001

  /// When the raw warning first reached each band, and has stayed at or above it since.
  private var heldSince: [ProximityBand: Date] = [:]
  /// The warning being beeped, from its first beep until all is clear: when it began and the
  /// worst band it has reached.
  private var held: (since: Date, top: ProximityBand)?
  private var lastAt: Date?

  init() {}

  mutating func next(
    reading: ProximityReading?, receivedAt: Date?, now: Date, volume: BeepVolume
  ) -> BeepPlan? {
    guard let reading, let receivedAt, now.timeIntervalSince(receivedAt) < Self.staleAfter,
      let raw = reading.warning
    else {
      held = nil
      heldSince = [:]
      return nil
    }
    for band in [ProximityBand.slow, .fast, .stop] {
      heldSince[band] = raw.band >= band ? heldSince[band] ?? now : nil
    }
    guard
      let ready = [ProximityBand.stop, .fast, .slow].first(where: {
        guard let since = heldSince[$0] else { return false }
        return now.timeIntervalSince(since) >= Self.rise($0) - Self.slack
      })
    else { return nil }
    // A band still rising plays as the band that has held, at that band's own pace.
    let warning =
      ready == raw.band ? raw : ProximityWarning(sector: raw.sector, distance: nil, band: ready)
    // The first beep, or a band worse than any yet in this warning, sounds at once. The
    // server's band flickers between fast and stop and the sector hops as the head turns, so
    // anything else, a new direction included, waits out the interval (spatial_sound.Repeats).
    if held == nil {
      held = (now, warning.band)
    } else if let h = held, warning.band > h.top {
      held = (h.since, warning.band)
    } else if let h = held, let l = lastAt {
      var interval = BeepCue.interval(for: warning)
      if now.timeIntervalSince(h.since) >= Self.settleAfter {
        interval = max(interval, Self.settledInterval)
      }
      if now.timeIntervalSince(l) < interval { return nil }
    }
    guard let plan = BeepCue.plan(for: warning, volume: volume) else { return nil }
    lastAt = now
    return plan
  }
}


// MARK: - The beacon: the item being found plays from where it is

/// What the beacon plays: one pluck off target, the pluck and the fifth above when the wearer
/// faces the item, three rising notes once on arrival.
enum BeaconSound: Int, CaseIterable, Sendable {
  case pluck, pair, arrive
}

/// One beacon sound to play now, placed like a beep (AVAudio3DPoint convention).
struct BeaconPlan: Equatable, Sendable {
  let sound: BeaconSound
  let gain: Float
  let x: Float
  let y: Float
  let z: Float
}

/// The beacon rules; metasight/spatial_sound.py is the reference and pins the same numbers.
/// It plays only while a guide has a direction and a distance, so it is never constant.
enum BeaconCue {
  static let arriveMetres = 0.8
  static let alignedDegrees = 10.0
  static let far = (metres: 4.0, seconds: 1.4)
  static let near = (metres: arriveMetres, seconds: 0.45)
  static let baseGain = 0.5
  static let hz = 523.25

  /// Seconds between plucks: 1.4 s at 4 m or more, down to 0.45 s at arrival.
  static func period(dist: Double) -> TimeInterval {
    let t = min(1, max(0, (far.metres - dist) / (far.metres - near.metres)))
    return far.seconds + (near.seconds - far.seconds) * t
  }

  /// Inverse distance from 1 m, held between a quarter and full: far stays audible.
  static func gain(dist: Double) -> Double { baseGain * min(1, max(0.25, 1 / max(dist, 1e-3))) }

  static func aligned(bearing: Double) -> Bool { abs(bearing) <= alignedDegrees }

  static func plan(dist: Double, bearing: Double, volume: BeepVolume) -> BeaconPlan {
    let radians = bearing * .pi / 180
    let r = BeepCue.placementMetres
    return BeaconPlan(
      sound: aligned(bearing: bearing) ? .pair : .pluck,
      gain: Float(gain(dist: dist)) * volume.scale,
      x: Float(sin(radians) * r), y: 0, z: Float(-cos(radians) * r))
  }
}

/// Decides when the next beacon sound is due: on the period while a guide has a fix, the
/// arrival notes once when the server says found, silence when the guide ends or goes stale.
struct BeaconScheduler: Sendable {
  private var next: Date?
  private var wasFound = false
  private var lastBearing = 0.0

  init() {}

  mutating func next(
    guide: GuideTarget?, found: Bool, receivedAt: Date?, now: Date, volume: BeepVolume
  ) -> BeaconPlan? {
    defer { wasFound = found }
    guard let receivedAt, now.timeIntervalSince(receivedAt) < BeepScheduler.staleAfter else {
      next = nil
      return nil
    }
    if found, !wasFound {
      next = nil
      let r = lastBearing * .pi / 180
      return BeaconPlan(
        sound: .arrive, gain: 0.8 * volume.scale,
        x: Float(sin(r) * BeepCue.placementMetres), y: 0, z: Float(-cos(r) * BeepCue.placementMetres))
    }
    guard let guide, let dist = guide.dist, let bearing = guide.bearing else {
      next = nil
      return nil
    }
    lastBearing = bearing
    if let due = next, now < due { return nil }
    next = now.addingTimeInterval(BeaconCue.period(dist: dist))
    return BeaconCue.plan(dist: dist, bearing: bearing, volume: volume)
  }
}

// MARK: - Room roll call: each name plucks from where the thing is

/// One named thing in the server's roll call (metasight/roll_call.py): where it is and where its
/// name sits in the spoken text (start and length in characters; the text is plain ASCII).
struct RollCallItem: Equatable, Sendable, Decodable {
  let label: String
  let bearing: Double
  let dist: Double
  let start: Int
  let length: Int
}

enum RollCall {
  /// The items whose names begin inside the range the speech synthesizer is about to say.
  static func due(_ items: [RollCallItem], spoken: NSRange) -> [RollCallItem] {
    items.filter { NSLocationInRange($0.start, spoken) }
  }

  static func plan(_ item: RollCallItem, volume: BeepVolume) -> BeaconPlan {
    BeaconCue.plan(dist: item.dist, bearing: item.bearing, volume: volume)
  }
}
