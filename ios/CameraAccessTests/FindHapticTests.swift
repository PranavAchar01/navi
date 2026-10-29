//
// FindHapticTests.swift
//
// The "find my X" buzz rules: closer is stronger, sharper and faster, and out of view is
// silent. Pure rules only; the Core Haptics player has no hardware on the simulator.
//

@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

@MainActor
final class FindHapticTests: XCTestCase {

  func testCloserIsStrongerSharperAndFaster() throws {
    let far = try XCTUnwrap(FindHaptic.plan(dist: 3.5, bearing: 0, inView: true))
    let mid = try XCTUnwrap(FindHaptic.plan(dist: 2.0, bearing: 0, inView: true))
    let near = try XCTUnwrap(FindHaptic.plan(dist: 0.9, bearing: 0, inView: true))
    XCTAssertLessThan(far.intensity, mid.intensity)
    XCTAssertLessThan(mid.intensity, near.intensity)
    XCTAssertLessThan(far.sharpness, near.sharpness)
    XCTAssertLessThan(far.pulseHz, mid.pulseHz)
    XCTAssertLessThan(mid.pulseHz, near.pulseHz)
  }

  func testClampsAtTheEnds() throws {
    let beyond = try XCTUnwrap(FindHaptic.plan(dist: 9, bearing: 0, inView: true))
    let atFar = try XCTUnwrap(FindHaptic.plan(dist: FindHaptic.far, bearing: 0, inView: true))
    XCTAssertEqual(beyond, atFar)
    let inside = try XCTUnwrap(FindHaptic.plan(dist: 0.2, bearing: 0, inView: true))
    XCTAssertEqual(inside.intensity, 1, accuracy: 1e-6)
    XCTAssertEqual(inside.pulseHz, 10, accuracy: 1e-9)
  }

  func testSilentOutOfViewOrWithoutAFix() {
    XCTAssertNil(FindHaptic.plan(dist: 1.5, bearing: 0, inView: false))
    XCTAssertNil(FindHaptic.plan(dist: nil, bearing: 0, inView: true))
    XCTAssertNil(FindHaptic.plan(dist: 1.5, bearing: nil, inView: true))
  }

  func testWithoutTheServersWordTheBearingDecides() {
    XCTAssertNotNil(FindHaptic.plan(dist: 1.5, bearing: 30, inView: nil))
    XCTAssertNil(FindHaptic.plan(dist: 1.5, bearing: 50, inView: nil))
    XCTAssertNil(FindHaptic.plan(dist: 1.5, bearing: -120, inView: nil))
  }

  func testPulsesAlternateWithoutStopping() throws {
    let p = try XCTUnwrap(FindHaptic.plan(dist: 2, bearing: 0, inView: true))
    let period = 1 / p.pulseHz
    XCTAssertEqual(p.level(at: period * 0.25), p.intensity)
    XCTAssertGreaterThan(p.level(at: period * 0.75), 0)
    XCTAssertLessThan(p.level(at: period * 0.75), p.intensity * 0.5)
  }

  func testGuideDecodesInView() throws {
    let json = #"{"label":"watch","dist":1.2,"bearing":-4,"in_view":true}"#
    let g = try JSONDecoder().decode(GuideTarget.self, from: Data(json.utf8))
    XCTAssertEqual(g.inView, true)
    let old = try JSONDecoder().decode(
      GuideTarget.self, from: Data(#"{"label":"watch","dist":1.2,"bearing":-4}"#.utf8))
    XCTAssertNil(old.inView)
  }

  func testWatchTapsFasterWhenCloserAndOnlyWithNothingToAvoid() {
    let clear = Array(repeating: "off", count: 8)
    func taps(_ find: Double?, modes: [String] = clear, distances: [Double?] = []) -> Int {
      var pacer = ProximityPacer()
      let r = ProximityReading(distances: distances, modes: modes, find: find)
      let t0 = Date()
      return (0..<30).filter { i in
        let now = t0.addingTimeInterval(Double(i) * 0.1)
        return pacer.pulse(reading: r, receivedAt: now, now: now) == .find
      }.count
    }
    XCTAssertEqual(taps(nil), 0)
    XCTAssertLessThan(taps(3.5), taps(1.0))
    var wall = clear
    wall[0] = "fast"
    XCTAssertEqual(taps(1.0, modes: wall, distances: [0.7]), 0)
  }

  func testWatchMessageCarriesTheFindDistance() throws {
    let r = ProximityReading(distances: [], modes: [], find: 1.4)
    let back = try XCTUnwrap(ProximityReading(message: r.message))
    XCTAssertEqual(back.find, 1.4)
    XCTAssertNil(ProximityReading(message: ProximityReading(distances: [], modes: []).message)?.find)
  }
}
