import XCTest

@testable import CameraAccess

final class WarningFilterTests: XCTestCase {
  private let t0 = Date(timeIntervalSince1970: 1_000)

  private func w(_ band: ProximityBand, _ d: Double, sector: Int = 0) -> ProximityWarning {
    ProximityWarning(sector: sector, distance: d, band: band)
  }

  func testABlipIsNotAWarning() {
    var f = WarningFilter()
    f.update(w(.fast, 0.8), now: t0)
    f.update(w(.fast, 0.8), now: t0.addingTimeInterval(0.2))
    XCTAssertNil(f.update(nil, now: t0.addingTimeInterval(0.25)))
  }

  func testAHeldCueRisesAndStopIsQuick() {
    var f = WarningFilter()
    f.update(w(.fast, 0.8), now: t0)
    XCTAssertEqual(f.update(w(.fast, 0.8), now: t0.addingTimeInterval(0.3))?.band, .fast)
    f.update(w(.stop, 0.4), now: t0.addingTimeInterval(0.35))
    XCTAssertEqual(f.update(w(.stop, 0.4), now: t0.addingTimeInterval(0.5))?.band, .stop)
  }

  func testItFallsOneZoneAtATime() {
    var f = WarningFilter()
    f.update(w(.stop, 0.3), now: t0)
    f.update(w(.stop, 0.3), now: t0.addingTimeInterval(0.2))
    XCTAssertEqual(f.update(nil, now: t0.addingTimeInterval(0.3))?.band, .stop)
    XCTAssertEqual(f.update(nil, now: t0.addingTimeInterval(0.9))?.band, .fast)
    XCTAssertEqual(f.update(nil, now: t0.addingTimeInterval(1.5))?.band, .slow)
    XCTAssertNil(f.update(nil, now: t0.addingTimeInterval(2.1)))
  }

  func testTheNumberMovesAtMostOnceASecond() {
    var f = WarningFilter()
    f.update(w(.fast, 0.8), now: t0)
    XCTAssertEqual(f.update(w(.fast, 0.8), now: t0.addingTimeInterval(0.3))?.distance, 0.8)
    XCTAssertEqual(f.update(w(.fast, 0.7), now: t0.addingTimeInterval(0.6))?.distance, 0.8)
    XCTAssertEqual(f.update(w(.fast, 0.7), now: t0.addingTimeInterval(1.3))?.distance, 0.7)
  }

  /// Replays a recording of the live safety stream (numbers only, written by a dev
  /// script to ios/build/replay, which is not committed) and prints how often the wearer
  /// is warned with and without the filter.
  @MainActor func testReplayWarningsPerMinute() throws {
    let url = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("build/replay/cues-60s.jsonl")
    guard let text = try? String(contentsOf: url, encoding: .utf8) else {
      throw XCTSkip("no recording at \(url.path)")
    }
    struct Msg: Decodable {
      struct Cue: Decodable { let mode: String }
      let type: String
      let _t: Double
      let sectors: [Double?]?
      let cues: [Cue]?
      let path: Double?
    }
    let msgs = text.split(separator: "\n").compactMap {
      try? JSONDecoder().decode(Msg.self, from: Data($0.utf8))
    }.filter { $0.type == "safety" && $0.sectors != nil && $0.cues != nil }
    var f = WarningFilter()
    var raw = Stats(), kept = Stats()
    for m in msgs {
      let now = Date(timeIntervalSince1970: m._t)
      let reading = ServerLink.proximity(sectors: m.sectors!, modes: m.cues!.map(\.mode), path: m.path)
      raw.add(reading.cueWarning, at: m._t)
      kept.add(f.update(reading.cueWarning, now: now), at: m._t)
    }
    let minutes = (msgs.last!._t - msgs.first!._t) / 60
    print("REPLAY \(msgs.count) msgs over \(String(format: "%.1f", minutes * 60)) s")
    print("REPLAY raw:      \(raw.summary(minutes))")
    print("REPLAY filtered: \(kept.summary(minutes))")
    XCTAssertLessThanOrEqual(kept.onsets, raw.onsets)
  }
}

private struct Stats {
  var onsets = 0, zoneChanges = 0, numberChanges = 0
  var seconds: [ProximityBand: Double] = [:]
  private var last: ProximityWarning?
  private var lastT: Double?

  mutating func add(_ w: ProximityWarning?, at t: Double) {
    if let lastT { seconds[last?.band ?? .clear, default: 0] += t - lastT }
    let band = w?.band ?? .clear, before = last?.band ?? .clear
    if band > before { onsets += 1 }
    if band != before { zoneChanges += 1 }
    if DisplayCard.distanceText(w?.distance) != DisplayCard.distanceText(last?.distance) { numberChanges += 1 }
    last = w
    lastT = t
  }

  func summary(_ minutes: Double) -> String {
    let total = seconds.values.reduce(0, +)
    let share = [ProximityBand.clear, .slow, .fast, .stop].map {
      "\($0): \(Int((100 * (seconds[$0] ?? 0) / total).rounded()))%"
    }.joined(separator: " ")
    return String(
      format: "%.0f warnings/min (rises), %.0f zone changes/min, %.0f number changes/min; ",
      Double(onsets) / minutes, Double(zoneChanges) / minutes, Double(numberChanges) / minutes) + share
  }
}
