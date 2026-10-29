//
// DisplayCardTests.swift
//
// What the Ray-Ban Display shows: which cue wins the screen, and the words on it.
//

@testable import CameraAccess
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class DisplayCardTests: XCTestCase {

  private let keys = GuideTarget(label: "keys", dist: 3.04, bearing: -88)
  private let note = AnswerNote(
    text: "Your keys are on the kitchen counter, three metres to your left.",
    label: "keys", near: "kitchen counter", at: Date())

  func testStopBeatsTheItem() {
    let stop = ProximityWarning(sector: 0, distance: 0.3, band: .stop)
    let card = DisplayCard.make(warning: stop, guide: keys, answer: note)
    XCTAssertEqual(card.title, "Stop")
    XCTAssertEqual(card.detail, "Ahead, 0.3 m")
    XCTAssertEqual(card.glyph, .obstacle(bearing: 0, band: .stop))
  }

  func testItemBeatsANearbyWall() {
    let fast = ProximityWarning(sector: 7, distance: 0.8, band: .fast)
    let card = DisplayCard.make(warning: fast, guide: keys, answer: note)
    XCTAssertEqual(card.title, "keys")
    XCTAssertEqual(card.detail, "kitchen counter · 3.0 m left")
    XCTAssertEqual(card.glyph, .target(bearing: -88))
  }

  func testItemFromAnEarlierWalkHasNoArrow() {
    let card = DisplayCard.make(
      warning: nil, guide: GuideTarget(label: "keys", dist: nil, bearing: nil), answer: note)
    XCTAssertEqual(card.detail, "near the kitchen counter")
    XCTAssertNil(card.glyph)
  }

  func testObstacleWhenNothingWasAsked() {
    let slow = ProximityWarning(sector: 2, distance: 1.4, band: .slow)
    let card = DisplayCard.make(warning: slow, guide: nil, answer: nil)
    XCTAssertEqual(card.title, "Right")
    XCTAssertEqual(card.detail, "1.4 m")
    XCTAssertEqual(DisplayCard.make(warning: nil, guide: nil, answer: nil), .clear)
  }

  func testAnswerFadesAndUnknownItemIsClipped() {
    let old = AnswerNote(text: note.text, label: "keys", near: nil, at: Date(timeIntervalSinceNow: -30))
    XCTAssertEqual(DisplayCard.make(warning: nil, guide: nil, answer: old), .clear)

    let vague = AnswerNote(
      text: "I haven't seen your passport yet. Walk past where you left them and ask again.",
      label: nil, near: nil, at: Date())
    let card = DisplayCard.make(warning: nil, guide: nil, answer: vague)
    XCTAssertLessThanOrEqual(card.title.count, 48)
    XCTAssertTrue(card.title.hasSuffix("…"))
  }

  func testDirectionWords() {
    XCTAssertEqual(DisplayCard.directionWord(0), "ahead")
    XCTAssertEqual(DisplayCard.directionWord(50), "ahead right")
    XCTAssertEqual(DisplayCard.directionWord(-95), "left")
    XCTAssertEqual(DisplayCard.directionWord(178), "behind")
    XCTAssertEqual(DisplayCard.directionWord(-150), "behind left")
  }
}
