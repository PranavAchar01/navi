//
// BeepCueTests.swift
//
// The band -> beep rules and the 3D placement. The engine tests render into memory with
// manual rendering: no audio hardware is opened, so nothing is ever played.
//

import AVFoundation
@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

@MainActor
final class BeepCueTests: XCTestCase {

  private func warning(sector: Int, distance: Double?, mode: String = "off") -> ProximityWarning {
    var d = [Double?](repeating: nil, count: ProximityReading.sectorCount)
    var m = [String](repeating: "off", count: ProximityReading.sectorCount)
    d[sector] = distance
    m[sector] = mode
    return ProximityReading(distances: d, modes: m).warning!
  }

  func testBandsMapToKinds() {
    XCTAssertNil(BeepCue.kind(for: .clear))
    XCTAssertEqual(BeepCue.kind(for: .slow), .tick)
    XCTAssertEqual(BeepCue.kind(for: .fast), .soft)
    XCTAssertEqual(BeepCue.kind(for: .stop), .clear)
    // Faintest to clearest.
    XCTAssertLessThan(BeepKind.tick.gain, BeepKind.soft.gain)
    XCTAssertLessThan(BeepKind.soft.gain, BeepKind.clear.gain)
    XCTAssertLessThan(BeepKind.tick.duration, BeepKind.soft.duration)
  }

  func testDistancesPickTheServersBands() {
    XCTAssertEqual(BeepCue.plan(for: warning(sector: 0, distance: 1.5), volume: .normal)?.kind, .tick)
    XCTAssertEqual(BeepCue.plan(for: warning(sector: 0, distance: 0.8), volume: .normal)?.kind, .soft)
    XCTAssertEqual(BeepCue.plan(for: warning(sector: 0, distance: 0.3), volume: .normal)?.kind, .clear)
    // The server's "solid" (time to collision under 1.2 s) is red even at 1.4 m.
    XCTAssertEqual(
      BeepCue.plan(for: warning(sector: 0, distance: 1.4, mode: "solid"), volume: .normal)?.kind, .clear)
  }

  func testSilentWhenClear() {
    let clear = ProximityReading(
      distances: [Double?](repeating: nil, count: 8), modes: [String](repeating: "off", count: 8))
    XCTAssertNil(clear.warning)
    var scheduler = BeepScheduler()
    XCTAssertNil(scheduler.next(reading: clear, receivedAt: Date(), now: Date(), volume: .normal))
    XCTAssertNil(scheduler.next(reading: nil, receivedAt: nil, now: Date(), volume: .normal))
  }

  func testIntervalsGetShorterAsItGetsCloser() {
    let yellow = BeepCue.interval(for: warning(sector: 0, distance: 1.9))
    let yellowNear = BeepCue.interval(for: warning(sector: 0, distance: 1.1))
    let orange = BeepCue.interval(for: warning(sector: 0, distance: 0.9))
    let orangeNear = BeepCue.interval(for: warning(sector: 0, distance: 0.55))
    let red = BeepCue.interval(for: warning(sector: 0, distance: 0.3))
    XCTAssertGreaterThan(yellow, yellowNear)
    XCTAssertGreaterThan(yellowNear, orange)
    XCTAssertGreaterThan(orange, orangeNear)
    XCTAssertGreaterThan(orangeNear, red)
    XCTAssertLessThanOrEqual(red, 0.25)
  }

  func testPlacementFollowsTheSector() {
    func place(_ sector: Int) -> (Float, Float) {
      let p = BeepCue.plan(for: warning(sector: sector, distance: 0.3), volume: .normal)!
      return (p.x, p.z)
    }
    let ahead = place(0), right = place(2), behind = place(4), left = place(6)
    XCTAssertEqual(ahead.0, 0, accuracy: 1e-5)
    XCTAssertLessThan(ahead.1, -0.9)  // ahead is negative z
    XCTAssertGreaterThan(right.0, 0.9)
    XCTAssertLessThan(left.0, -0.9)
    XCTAssertGreaterThan(behind.1, 0.9)
  }

  func testQuietIsQuieterAndKeepsTheOrder() {
    let w = warning(sector: 0, distance: 0.3)
    let quiet = BeepCue.plan(for: w, volume: .quiet)!.gain
    let normal = BeepCue.plan(for: w, volume: .normal)!.gain
    XCTAssertLessThan(quiet, normal)
    XCTAssertGreaterThan(quiet, 0)
  }

  func testSchedulerRepeatsOnTheIntervalAndSoundsNewWarningsAtOnce() {
    var scheduler = BeepScheduler()
    var d = [Double?](repeating: nil, count: 8)
    d[0] = 1.5
    let yellow = ProximityReading(distances: d, modes: [String](repeating: "off", count: 8))
    let t0 = Date()
    XCTAssertNotNil(scheduler.next(reading: yellow, receivedAt: t0, now: t0, volume: .normal))
    XCTAssertNil(scheduler.next(reading: yellow, receivedAt: t0.addingTimeInterval(0.4), now: t0.addingTimeInterval(0.5), volume: .normal))
    XCTAssertNotNil(scheduler.next(reading: yellow, receivedAt: t0.addingTimeInterval(1.6), now: t0.addingTimeInterval(1.7), volume: .normal))
    // It got closer: red sounds immediately.
    d[0] = 0.3
    let red = ProximityReading(distances: d, modes: [String](repeating: "off", count: 8))
    let t1 = t0.addingTimeInterval(1.8)
    XCTAssertEqual(scheduler.next(reading: red, receivedAt: t1, now: t1, volume: .normal)?.kind, .clear)
    // Readings stopped arriving: silence, not a beep at a stale obstacle.
    XCTAssertNil(scheduler.next(reading: red, receivedAt: t1, now: t1.addingTimeInterval(3), volume: .normal))
  }

  func testAHeldWarningSettlesAndAFlickerDoesNotRestartIt() {
    var scheduler = BeepScheduler()
    var near = [Double?](repeating: nil, count: 8)
    near[0] = 0.3
    var mid = near
    mid[0] = 0.7
    let off = [String](repeating: "off", count: 8)
    let stop = ProximityReading(distances: near, modes: off)
    let fast = ProximityReading(distances: mid, modes: off)
    let t0 = Date()
    var beeps: [Double] = []
    // Ten seconds at a desk, the band flickering between stop and fast every 0.3 s.
    for step in 0..<250 {
      let t = t0.addingTimeInterval(Double(step) * 0.04)
      let reading = (step / 8) % 2 == 0 ? stop : fast
      if scheduler.next(reading: reading, receivedAt: t, now: t, volume: .normal) != nil {
        beeps.append(Double(step) * 0.04)
      }
    }
    let settled = beeps.filter { $0 >= 3.2 }
    XCTAssertGreaterThanOrEqual(beeps.filter { $0 < 3 }.count, 8)
    XCTAssertLessThanOrEqual(settled.count, 7)
    // The head turns and the warning hops right: it keeps the settled pace.
    var side = [Double?](repeating: nil, count: 8)
    side[2] = 0.3
    let right = ProximityReading(distances: side, modes: off)
    let t1 = t0.addingTimeInterval(10.04)
    XCTAssertNil(scheduler.next(reading: right, receivedAt: t1, now: t1, volume: .normal))
    // All clear, then a new warning: it sounds at once.
    let clear = ProximityReading(distances: [Double?](repeating: nil, count: 8), modes: off)
    let t2 = t1.addingTimeInterval(0.1)
    XCTAssertNil(scheduler.next(reading: clear, receivedAt: t2, now: t2, volume: .normal))
    let t3 = t2.addingTimeInterval(0.1)
    XCTAssertNotNil(scheduler.next(reading: right, receivedAt: t3, now: t3, volume: .normal))
  }

  // MARK: - Engine, rendered offline

  private func energy(_ buffer: AVAudioPCMBuffer) -> (left: Float, right: Float) {
    var l: Float = 0, r: Float = 0
    guard let ch = buffer.floatChannelData else { return (0, 0) }
    for i in 0..<Int(buffer.frameLength) {
      l += ch[0][i] * ch[0][i]
      r += ch[1][i] * ch[1][i]
    }
    return (l, r)
  }

  private func render(sector: Int) throws -> (left: Float, right: Float) {
    let beeper = SpatialBeeper(output: .offline(maximumFrames: 4096))
    let plan = try XCTUnwrap(
      BeepCue.plan(for: warning(sector: sector, distance: 0.3), volume: .normal))
    beeper.play(plan)
    let out = try XCTUnwrap(beeper.renderOffline(frames: 4096))
    return energy(out)
  }

  func testABeepOnTheRightIsLouderInTheRightEar() throws {
    let e = try render(sector: 2)
    XCTAssertGreaterThan(e.right + e.left, 0, "the beep rendered nothing")
    XCTAssertGreaterThan(e.right, e.left * 1.5)
  }

  func testABeepOnTheLeftIsLouderInTheLeftEar() throws {
    let e = try render(sector: 6)
    XCTAssertGreaterThan(e.left, e.right * 1.5)
  }

  func testQuietRendersLessEnergyThanNormal() throws {
    func total(_ v: BeepVolume) throws -> Float {
      let beeper = SpatialBeeper(output: .offline(maximumFrames: 4096))
      beeper.play(try XCTUnwrap(BeepCue.plan(for: warning(sector: 0, distance: 0.3), volume: v)))
      let e = energy(try XCTUnwrap(beeper.renderOffline(frames: 4096)))
      return e.left + e.right
    }
    XCTAssertLessThan(try total(.quiet), try total(.normal))
  }

  func testTheDeviceBeeperIsSilentOnTheSimulator() {
    #if targetEnvironment(simulator)
      XCTAssertFalse(SpatialBeeper(output: .device).canSound)
    #endif
  }

  // MARK: - The beacon (rules pinned to metasight/spatial_sound.py)

  private func guide(dist: Double?, bearing: Double?) -> GuideTarget {
    GuideTarget(label: "chair", dist: dist, bearing: bearing)
  }

  func testBeaconQuickensAsItCloses() {
    XCTAssertEqual(BeaconCue.period(dist: 6), 1.4, accuracy: 1e-9)
    XCTAssertEqual(BeaconCue.period(dist: BeaconCue.arriveMetres), 0.45, accuracy: 1e-9)
    XCTAssertLessThan(BeaconCue.period(dist: 2), BeaconCue.period(dist: 3))
    XCTAssertEqual(BeaconCue.gain(dist: 10), BeaconCue.baseGain / 4, accuracy: 1e-9)
  }

  func testBeaconPlacementLeftRightFrontBack() {
    let right = BeaconCue.plan(dist: 2, bearing: 90, volume: .normal)
    XCTAssertEqual(right.x, 1, accuracy: 1e-6)
    XCTAssertEqual(right.z, 0, accuracy: 1e-6)
    XCTAssertLessThan(BeaconCue.plan(dist: 2, bearing: -90, volume: .normal).x, -0.99)
    XCTAssertLessThan(BeaconCue.plan(dist: 2, bearing: 0, volume: .normal).z, -0.99)  // ahead is -z
    XCTAssertGreaterThan(BeaconCue.plan(dist: 2, bearing: 180, volume: .normal).z, 0.99)
    XCTAssertEqual(BeaconCue.plan(dist: 2, bearing: 5, volume: .normal).sound, .pair)
    XCTAssertEqual(BeaconCue.plan(dist: 2, bearing: -40, volume: .normal).sound, .pluck)
  }

  func testBeaconSchedulerPlaysOnThePeriodThenChimesOnce() {
    var s = BeaconScheduler()
    let t0 = Date()
    let g = guide(dist: 3, bearing: 30)
    XCTAssertNotNil(s.next(guide: g, found: false, receivedAt: t0, now: t0, volume: .normal))
    XCTAssertNil(s.next(guide: g, found: false, receivedAt: t0, now: t0 + 0.5, volume: .normal))
    let due = t0 + BeaconCue.period(dist: 3) + 0.01
    XCTAssertNotNil(s.next(guide: g, found: false, receivedAt: due, now: due, volume: .normal))
    // No direction (the map has no fresh pose): silent, not a guess.
    let blind = guide(dist: nil, bearing: nil)
    XCTAssertNil(s.next(guide: blind, found: false, receivedAt: due + 2, now: due + 2, volume: .normal))
    // Arrival: the chime once, from the last direction, then nothing.
    let a = s.next(guide: nil, found: true, receivedAt: due + 3, now: due + 3, volume: .normal)
    XCTAssertEqual(a?.sound, .arrive)
    XCTAssertGreaterThan(a?.x ?? 0, 0)
    XCTAssertNil(s.next(guide: nil, found: true, receivedAt: due + 3.1, now: due + 3.1, volume: .normal))
    // A stale guide never keeps calling.
    XCTAssertNil(s.next(guide: g, found: false, receivedAt: due + 4, now: due + 9, volume: .normal))
  }

  func testABeaconOnTheLeftIsLouderInTheLeftEarAndOffIsSilent() throws {
    let beeper = SpatialBeeper(output: .offline(maximumFrames: 16384))
    beeper.volume = .normal
    beeper.play(BeaconCue.plan(dist: 2, bearing: -90, volume: .normal))
    let e = energy(try XCTUnwrap(beeper.renderOffline(frames: 16384)))
    XCTAssertGreaterThan(e.left, e.right * 1.5)
    let off = SpatialBeeper(output: .offline(maximumFrames: 16384))
    off.volume = .normal
    off.serverSound = false
    off.play(BeaconCue.plan(dist: 2, bearing: -90, volume: .normal))
    XCTAssertTrue(off.playedBeacon.isEmpty)
  }

  func testRollCallPlucksEachNameAsItIsSpoken() {
    let items = [
      RollCallItem(label: "bed", bearing: 60, dist: 2, start: 9, length: 3),
      RollCallItem(label: "desk", bearing: -60, dist: 1, start: 40, length: 4),
    ]
    XCTAssertEqual(RollCall.due(items, spoken: NSRange(location: 9, length: 3)).map(\.label), ["bed"])
    XCTAssertEqual(RollCall.due(items, spoken: NSRange(location: 13, length: 2)), [])
    XCTAssertEqual(RollCall.due(items, spoken: NSRange(location: 40, length: 4)).map(\.label), ["desk"])
    XCTAssertGreaterThan(RollCall.plan(items[0], volume: .normal).x, 0)
    XCTAssertLessThan(RollCall.plan(items[1], volume: .normal).x, 0)
  }

  func testBeepsMatchTheRendererSampleForSample() {
    // spatial_sound.tick_sound(kind) / gain at samples 20, 100, 300 and the middle.
    let expected: [(BeepKind, Int, [Float])] = [
      (.tick, 529, [-0.049, 0.16429, -0.82709, -0.27959]),
      (.soft, 3087, [0.03334, -0.03953, -0.08884, -0.4829]),
      (.clear, 3969, [-0.03267, -0.05795, -0.10464, 0.14162]),
    ]
    for (kind, count, values) in expected {
      guard let buffer = SpatialBeeper.tone(kind), let x = buffer.floatChannelData?[0] else {
        return XCTFail("no buffer for \(kind)")
      }
      XCTAssertEqual(Int(buffer.frameLength), count)
      for (i, v) in zip([20, 100, 300, count / 2], values) {
        XCTAssertEqual(x[i], v, accuracy: 2e-4, "\(kind) sample \(i)")
      }
    }
  }

  func testPlucksMatchTheRendererSampleForSample() {
    // spatial_sound._pluck(523.25) and _pluck(523.25 * 1.26, 0.45) at samples 20, 100, 300, 7000.
    let cases: [(Double, Double, Int, [Float])] = [
      (0.32, 1.0, 14112, [0.06043, 0.33607, -0.17559, 0.05884]),
      (0.45, 1.26, 19845, [0.13858, 0.00134, 0.01981, -0.1388]),
    ]
    for (duration, ratio, count, values) in cases {
      let x = SpatialBeeper.pluck(BeaconCue.hz * ratio, duration: duration)
      XCTAssertEqual(x.count, count)
      for (i, v) in zip([20, 100, 300, 7000], values) {
        XCTAssertEqual(x[i], v, accuracy: 2e-4, "pluck x\(ratio) sample \(i)")
      }
    }
  }
}
