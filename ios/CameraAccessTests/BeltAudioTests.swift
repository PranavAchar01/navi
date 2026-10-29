//
// BeltAudioTests.swift
//
// The belt-as-sound rules (pan, pitch, loudness) and the synth's stereo placement. Rendering is
// a plain function call into memory: no audio hardware is opened, nothing is played.
//

@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class BeltAudioTests: XCTestCase {

  func testPanFollowsTheAngle() {
    XCTAssertEqual(BeltTones.pan(0), 0, accuracy: 1e-6)
    XCTAssertEqual(BeltTones.pan(2), 1, accuracy: 1e-6)
    XCTAssertEqual(BeltTones.pan(6), -1, accuracy: 1e-6)
    XCTAssertEqual(BeltTones.pan(4), 0, accuracy: 1e-6)
    let left = BeltTones.stereoGains(6)
    XCTAssertGreaterThan(left.left, 0.99)
    XCTAssertLessThan(left.right, 0.01)
  }

  func testFrontIsHigherThanBack() {
    XCTAssertEqual(BeltTones.frequency(0) / BeltTones.frequency(4), 2, accuracy: 1e-9)
    XCTAssertEqual(BeltTones.frequency(2), BeltTones.baseHz, accuracy: 1e-9)
    XCTAssertEqual(BeltTones.frequency(2, guide: true), BeltTones.baseHz * 1.5, accuracy: 1e-9)
  }

  func testTargetsFollowDutyAndGuideMode() {
    let t = BeltTones.targets(duty: [0, 0, 255], modes: ["off", "off", "guide"])
    XCTAssertEqual(t[0].amp, 0)
    XCTAssertEqual(t[2].amp, BeltTones.voiceGain, accuracy: 1e-6)
    XCTAssertTrue(t[2].guide)
    XCTAssertEqual(t.count, BeltTones.motorCount)
  }

  func testSynthPutsARightMotorOnTheRight() {
    let synth = BeltToneSynth(sampleRate: 22_050)
    var duty = [Int](repeating: 0, count: 8)
    duty[2] = 255
    synth.setTargets(BeltTones.targets(duty: duty, modes: []))
    let n = 4_000
    var l = [Float](repeating: 0, count: n)
    var r = [Float](repeating: 0, count: n)
    l.withUnsafeMutableBufferPointer { lp in
      r.withUnsafeMutableBufferPointer { rp in
        synth.render(left: lp.baseAddress!, right: rp.baseAddress!, count: n)
      }
    }
    let el = l.reduce(0) { $0 + $1 * $1 }
    let er = r.reduce(0) { $0 + $1 * $1 }
    XCTAssertGreaterThan(er, 20 * el)
  }
}
