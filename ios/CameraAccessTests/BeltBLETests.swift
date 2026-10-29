//
// BeltBLETests.swift
//
// The packet check in front of the belt's Bluetooth write. No radio is touched.
//

@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class BeltBLETests: XCTestCase {

  func testOnlyTheBeltsCommandPacketIsAccepted() {
    var packet = Data([0x4D, 0x53, 1, 0])
    packet.append(Data(repeating: 0, count: 22))
    XCTAssertEqual(packet.count, 26)
    XCTAssertTrue(BeltBLE.accepts(packet))
    XCTAssertFalse(BeltBLE.accepts(packet.dropLast()))
    XCTAssertFalse(BeltBLE.accepts(Data(repeating: 0, count: 26)))
    XCTAssertFalse(BeltBLE.accepts(Data()))
  }

  func testStatusLabels() {
    XCTAssertEqual(BeltBLE.Status.connected.label, "Belt: connected")
    XCTAssertEqual(BeltBLE.Status.bluetoothOff.label, "Belt: Bluetooth off")
  }
}
