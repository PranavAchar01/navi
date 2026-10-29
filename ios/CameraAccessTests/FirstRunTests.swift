import XCTest

@testable import CameraAccess

final class FirstRunTests: XCTestCase {
  func testStartsOnceThenOnlyByButton() {
    let d = UserDefaults(suiteName: "firstrun.test")!
    d.removePersistentDomain(forName: "firstrun.test")
    XCTAssertTrue(FirstRun.shouldStart(step: "idle", defaults: d))
    XCTAssertFalse(FirstRun.shouldStart(step: "walk", defaults: d))
    FirstRun.markStarted(defaults: d)
    XCTAssertFalse(FirstRun.shouldStart(step: "idle", defaults: d))
  }
}
