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

@MainActor
final class CaptionTests: XCTestCase {
  func testCaptionsKeepTheLastFourLinesAndTheWordsSoFar() {
    let link = ServerLink()
    link.handle(#"{"type":"caption","text":"The dose is","final":false}"#)
    XCTAssertEqual(link.captions.partial, "The dose is")
    for i in 1...5 {
      link.handle(#"{"type":"caption","text":"Line \#(i)","who":"person 1","final":true}"#)
    }
    XCTAssertEqual(link.captions.lines.count, 4)
    XCTAssertEqual(link.captions.lines.first?.text, "Line 2")
    XCTAssertEqual(link.captions.lines.last?.who, "person 1")
    XCTAssertEqual(link.captions.partial, "")
  }

  func testANewRecordingClearsTheCaptions() {
    let link = ServerLink()
    link.handle(#"{"type":"caption","text":"old","final":true}"#)
    link.handle(#"{"type":"notes","state":"recording"}"#)
    XCTAssertTrue(link.captions.lines.isEmpty)
  }

  func testBrailleAnnouncementsAreOffByDefault() {
    let d = UserDefaults(suiteName: "captions.test")!
    d.removePersistentDomain(forName: "captions.test")
    XCTAssertFalse(CaptionState.toBraille(d))
    d.set(true, forKey: CaptionState.brailleKey)
    XCTAssertTrue(CaptionState.toBraille(d))
  }
}
