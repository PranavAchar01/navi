import XCTest

@testable import CameraAccess

@MainActor
final class NotesTests: XCTestCase {
  func testNotesEventsUpdateTheState() {
    let link = ServerLink()
    XCTAssertFalse(link.notes.isRecording)
    link.handle(#"{"type":"notes","state":"recording","say":"Recording this conversation."}"#)
    XCTAssertTrue(link.notes.isRecording)
    link.handle(#"{"type":"notes","state":"recording","lines":3}"#)
    XCTAssertTrue(link.notes.isRecording)
    link.handle(#"{"type":"notes","state":"summarizing","say":"Stopped."}"#)
    XCTAssertFalse(link.notes.isRecording)
    XCTAssertEqual(link.notes.state, "summarizing")
    link.handle(#"{"type":"notes","state":"ready","title":"Doctor visit","summary":"- 20 mg"}"#)
    XCTAssertEqual(link.notes.title, "Doctor visit")
    XCTAssertEqual(link.notes.summary, "- 20 mg")
  }
}
