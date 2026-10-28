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
}
