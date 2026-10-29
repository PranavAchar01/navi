import XCTest

@testable import CameraAccess

@MainActor
final class ActivityTests: XCTestCase {
  func testActivityMessageSetsTheMode() {
    let link = ServerLink()
    XCTAssertEqual(link.activity, "guide")
    link.handle(#"{"type":"activity","mode":"talk","prev":"guide","say":"Talk."}"#)
    XCTAssertEqual(link.activity, "talk")
    link.handle(#"{"type":"activity","mode":"find","prev":"talk"}"#)
    XCTAssertEqual(link.activity, "find")
  }
}
