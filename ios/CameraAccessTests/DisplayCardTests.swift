//
// DisplayCardTests.swift
//
// What the Ray-Ban Display shows: which cue wins the screen, and the words on it.
//

@testable import CameraAccess
import Foundation
import UIKit
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class DisplayCardTests: XCTestCase {

  private let keys = GuideTarget(label: "keys", dist: 3.04, bearing: -88)
  private let note = AnswerNote(
    text: "Your keys are on the kitchen counter, three metres to your left.",
    label: "keys", near: "kitchen counter", seen: Date(timeIntervalSinceNow: -540), at: Date())

  func testStopBeatsTheItem() {
    let stop = ProximityWarning(sector: 0, distance: 0.3, band: .stop)
    let card = DisplayCard.make(warning: stop, guide: keys, answer: note)
    XCTAssertEqual(card.title, "Stop")
    XCTAssertEqual(card.detail, "Ahead · 0.3 m")
    XCTAssertEqual(card.glyph, .obstacle(bearing: 0, band: .stop))
  }

  func testItemBeatsANearbyWall() {
    let fast = ProximityWarning(sector: 7, distance: 0.8, band: .fast)
    let card = DisplayCard.make(warning: fast, guide: keys, answer: note)
    XCTAssertEqual(card.title, "3.0 m")
    XCTAssertEqual(card.detail, "keys · kitchen counter")
    XCTAssertEqual(card.glyph, .target(bearing: -88))
  }

  func testItemFromAnEarlierWalkGetsAPin() {
    let card = DisplayCard.make(
      warning: nil, guide: GuideTarget(label: "keys", dist: nil, bearing: nil), answer: note)
    XCTAssertEqual(card.title, "keys")
    XCTAssertEqual(card.detail, "kitchen counter · 9 min ago")
    XCTAssertEqual(card.glyph, .pin)
  }

  func testAnswerSaysWhereAndWhen() {
    let card = DisplayCard.make(warning: nil, guide: nil, answer: note)
    XCTAssertEqual(card.title, "keys")
    XCTAssertEqual(card.detail, "kitchen counter · 9 min ago")
    XCTAssertEqual(card.glyph, .pin)
  }

  func testDistancesAreGlanceSized() {
    XCTAssertEqual(DisplayCard.distanceText(0.43), "0.4 m")
    XCTAssertEqual(DisplayCard.distanceText(0.57), "0.6 m")
    XCTAssertEqual(DisplayCard.distanceText(0.04), "0.1 m")
    XCTAssertEqual(DisplayCard.distanceText(1.3), "1.5 m")
    XCTAssertEqual(DisplayCard.distanceText(4.4), "4.0 m")
    XCTAssertEqual(DisplayCard.distanceText(2.8), "3.0 m")
    XCTAssertEqual(DisplayCard.distanceText(nil), "close")
  }

  func testObstacleWhenNothingWasAsked() {
    let slow = ProximityWarning(sector: 2, distance: 1.4, band: .slow)
    let card = DisplayCard.make(warning: slow, guide: nil, answer: nil)
    XCTAssertEqual(card.title, "1.5 m")
    XCTAssertEqual(card.detail, "Right")
    XCTAssertEqual(DisplayCard.make(warning: nil, guide: nil, answer: nil), .clear)
  }

  func testAnswerFadesAndUnknownItemIsClipped() {
    let old = AnswerNote(text: note.text, label: "keys", near: nil, seen: nil, at: Date(timeIntervalSinceNow: -30))
    XCTAssertEqual(DisplayCard.make(warning: nil, guide: nil, answer: old), .clear)

    let vague = AnswerNote(
      text: "I haven't seen your passport yet. Walk past where you left them and ask again.",
      label: nil, near: nil, seen: nil, at: Date())
    let card = DisplayCard.make(warning: nil, guide: nil, answer: vague)
    XCTAssertLessThanOrEqual(card.title.count, 48)
    XCTAssertEqual(card.title, "I haven't seen your passport yet. Walk past…")
  }

  @MainActor
  func testCardIsOneSmallSquare() {
    let card = DisplayCard.make(warning: nil, guide: keys, answer: note)
    let image = GlassesDisplay.image(for: card)
    XCTAssertEqual(image.size, CGSize(width: GlassesDisplay.cardSide, height: GlassesDisplay.cardSide))
    // Mostly black: it should cost little on the Bluetooth link.
    XCTAssertLessThan(image.pngData()?.count ?? .max, 40_000)
  }

  func testDirectionWords() {
    XCTAssertEqual(DisplayCard.directionWord(0), "ahead")
    XCTAssertEqual(DisplayCard.directionWord(50), "ahead right")
    XCTAssertEqual(DisplayCard.directionWord(-95), "left")
    XCTAssertEqual(DisplayCard.directionWord(178), "behind")
    XCTAssertEqual(DisplayCard.directionWord(-150), "behind left")
  }
}
