//
// ProximityTests.swift
//
// The watch's parking-sensor logic: which obstacle wins, and how often the wrist taps.
//

@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class ProximityTests: XCTestCase {

  private func reading(_ pairs: [Int: Double]) -> ProximityReading {
    var d = [Double?](repeating: nil, count: ProximityReading.sectorCount)
    for (sector, metres) in pairs { d[sector] = metres }
    return ProximityReading(distances: d, modes: d.map { _ in "off" })
  }

  func testBands() {
    XCTAssertEqual(ProximityBand(distance: nil), .clear)
    XCTAssertEqual(ProximityBand(distance: 2.5), .clear)
    XCTAssertEqual(ProximityBand(distance: 1.5), .slow)
    XCTAssertEqual(ProximityBand(distance: 0.8), .fast)
    XCTAssertEqual(ProximityBand(distance: 0.3), .stop)
    XCTAssertEqual(ProximityBand(mode: "solid"), .stop)
  }

  func testClosestBandWinsThenAhead() {
    // A wall 0.4 m to the left beats a chair 1.5 m ahead.
    XCTAssertEqual(reading([0: 1.5, 6: 0.4]).warning?.sector, 6)
    // Same band: ahead wins over the side.
    XCTAssertEqual(reading([0: 1.6, 2: 1.2]).warning?.sector, 0)
    XCTAssertNil(reading([:]).warning)
  }

  func testMessageRoundTrip() {
    let r = reading([0: 0.7, 4: 1.9])
    let back = ProximityReading(message: r.message)
    XCTAssertEqual(back?.distances[0], 0.7)
    XCTAssertNil(back?.distances[1])
    XCTAssertEqual(back?.warning?.spoken, r.warning?.spoken)
  }

  func testPulsesSpeedUpAsItGetsCloser() {
    let far = ProximityPacer.interval(for: reading([0: 1.9]).warning!)
    let mid = ProximityPacer.interval(for: reading([0: 0.9]).warning!)
    let near = ProximityPacer.interval(for: reading([0: 0.6]).warning!)
    XCTAssertGreaterThan(far, mid)
    XCTAssertGreaterThan(mid, near)
  }

  func testSilentWhenClearAndPatternPerDirection() {
    var pacer = ProximityPacer()
    let t0 = Date()
    XCTAssertNil(pacer.pulse(reading: reading([:]), receivedAt: t0, now: t0))
    XCTAssertEqual(pacer.pulse(reading: reading([2: 1.2]), receivedAt: t0, now: t0.addingTimeInterval(1)), .side)
    var behind = ProximityPacer()
    XCTAssertEqual(behind.pulse(reading: reading([4: 1.2]), receivedAt: t0, now: t0), .behind)
    var stop = ProximityPacer()
    XCTAssertEqual(stop.pulse(reading: reading([0: 0.3]), receivedAt: t0, now: t0), .stop)
  }

  func testRateLimited() {
    var pacer = ProximityPacer()
    let t0 = Date()
    let r = reading([0: 0.3])
    XCTAssertEqual(pacer.pulse(reading: r, receivedAt: t0, now: t0), .stop)
    XCTAssertNil(pacer.pulse(reading: r, receivedAt: t0, now: t0.addingTimeInterval(0.2)))
  }

  func testLostSignalTapsOnce() {
    var pacer = ProximityPacer()
    let t0 = Date()
    let r = reading([0: 1.5])
    XCTAssertEqual(pacer.pulse(reading: r, receivedAt: t0, now: t0), .ahead)
    XCTAssertEqual(pacer.pulse(reading: r, receivedAt: t0, now: t0.addingTimeInterval(2)), .lost)
    XCTAssertNil(pacer.pulse(reading: r, receivedAt: t0, now: t0.addingTimeInterval(4)))
  }
}
