import XCTest

@testable import CameraAccess

@MainActor
final class OnboardingTests: XCTestCase {
  func testOnboardingEventsUpdateTheState() {
    let link = ServerLink()
    XCTAssertEqual(link.onboarding.step, "idle")
    link.handle(#"{"type":"onboarding","step":"walk","progress":0.0,"items":[],"say":"Hi."}"#)
    XCTAssertEqual(link.onboarding.step, "walk")
    link.handle(#"{"type":"onboarding","step":"walk","progress":0.5,"items":[]}"#)
    XCTAssertEqual(link.onboarding.progress, 0.5)
    link.handle(#"{"type":"onboarding","step":"name","progress":1.0,"items":["keys"]}"#)
    XCTAssertEqual(link.onboarding.items, ["keys"])
    link.handle(#"{"type":"onboarding","step":"belt","items":["keys"],"mode":"talk"}"#)
    XCTAssertEqual(link.onboarding.mode, "talk")
    link.handle(#"{"type":"onboarding","step":"done","items":["keys"]}"#)
    XCTAssertEqual(link.onboarding.step, "done")
  }
}
