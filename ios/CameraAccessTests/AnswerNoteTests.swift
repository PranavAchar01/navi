import XCTest

@testable import CameraAccess

@MainActor
final class AnswerNoteTests: XCTestCase {
  private let pin = #""pin":{"label":"laptop","ts":1700000000,"near":["TV"]}"#

  func testARememberedAnswerShowsWhereAndWhen() {
    let link = ServerLink()
    link.handle(#"{"type":"answer","text":"Your laptop was near the TV.","label":"laptop","# + pin + "}")
    XCTAssertEqual(link.answerNote?.near, "TV")
    XCTAssertEqual(link.answerNote?.seen, Date(timeIntervalSince1970: 1_700_000_000))
  }

  func testALiveAnswerIsSeenNowAndDropsTheOldPlace() {
    let link = ServerLink()
    link.handle(
      #"{"type":"answer","text":"Your laptop is in view now.","label":"laptop","live":true,"#
        + pin + "}")
    XCTAssertNil(link.answerNote?.near)
    XCTAssertLessThan(abs(link.answerNote?.seen?.timeIntervalSinceNow ?? -99), 5)
  }
}
