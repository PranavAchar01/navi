//
// ReflexGeometryTests.swift
//
// The phone's reflex against the server's own numbers. reflex_golden.json (written by
// scripts/coreml_depth.py --golden) holds four walk frames' depth, numpy's RANSAC draws for each,
// and what metasight/safety.py `analyse` made of them, plus belt.py plan/packet cases. The
// Swift port replays the same draws, so floor, sectors and packets must match.
//

@testable import CameraAccess
import CoreML
import Foundation
// ast-grep-ignore: swift-testing/swift/no-new-xctest
import XCTest

final class ReflexGeometryTests: XCTestCase {

  private struct Golden: Decodable {
    struct Case: Decodable {
      let frame: String
      let width: Int
      let height: Int
      let depth: [Float]
      let draws: [[Int]]
      let sectors: [Double?]
      let path_m: Double?
      let drop_m: Double?
      let floor: Bool
      let plane: [Double]?
    }
    struct Plan: Decodable {
      let sectors: [Double?]
      let path_m: Double?
      let ttc_s: Double?
      let drop_m: Double?
      let cmd: [[UInt8]]
      let flags: UInt8
      let packet: String
    }
    let eye_h: Double
    let hfov: Double
    let scale: Double
    let cases: [Case]
    let plans: [Plan]
  }

  private func golden() throws -> Golden {
    let url = try XCTUnwrap(
      Bundle(for: Self.self).url(forResource: "reflex_golden", withExtension: "json"))
    return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
  }

  private func metres(_ x: Double?) -> Double { x ?? .infinity }

  private func assertClose(
    _ a: Double, _ b: Double, _ what: String, tol: Double = 0.01, line: UInt = #line
  ) {
    if a.isInfinite || b.isInfinite {
      XCTAssertEqual(a, b, what, line: line)
    } else {
      XCTAssertEqual(a, b, accuracy: tol, what, line: line)
    }
  }

  func testFloorAndSectorsMatchTheServer() throws {
    let g = try golden()
    XCTAssertEqual(g.cases.count, 4)
    for c in g.cases {
      let draws = c.draws
      let (scene, plane) = ReflexGeometry.analyse(
        depth: c.depth, w: c.width, h: c.height, scale: g.scale, prior: g.scale, hfovDeg: g.hfov,
        eyeH: g.eye_h, prev: nil, draw: { k, _ in (draws[k][0], draws[k][1], draws[k][2]) })
      XCTAssertEqual(scene.floor, c.floor, c.frame)
      for (i, s) in c.sectors.enumerated() {
        assertClose(scene.sectors[i], metres(s), "\(c.frame) sector \(i)")
      }
      assertClose(scene.pathM, metres(c.path_m), "\(c.frame) path")
      assertClose(scene.dropM, metres(c.drop_m), "\(c.frame) drop")
      if let p = c.plane {
        let found = try XCTUnwrap(plane, c.frame)
        let n = SIMD3(p[0], p[1], p[2])
        XCTAssertGreaterThan((found.n * n).sum(), 0.9999, "\(c.frame) normal")
        assertClose(found.c, p[3], "\(c.frame) offset", tol: 0.005)
      } else {
        XCTAssertNil(plane, c.frame)
      }
    }
  }

  func testBeltPacketsMatchTheServer() throws {
    for p in try golden().plans {
      let (cmd, flags) = BeltPlan.plan(
        camera: p.sectors.map(metres), pathM: metres(p.path_m), ttcS: metres(p.ttc_s),
        dropM: metres(p.drop_m))
      XCTAssertEqual(cmd.map { [$0.0, $0.1] }, p.cmd)
      XCTAssertEqual(flags, p.flags)
      let packet = BeltPlan.packet(cmd, flags: flags, seq: 513, tMs: 1_234_567)
      XCTAssertEqual(packet.map { String(format: "%02x", $0) }.joined(), p.packet)
      XCTAssertTrue(BeltBLE.accepts(packet))
    }
  }

  func testAreaResizeAveragesBlocks() {
    // 4x2 to 2x1: each output pixel is the mean of a 2x2 block.
    let src: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    XCTAssertEqual(ReflexGeometry.areaResize(src, w: 4, h: 2, toW: 2, toH: 1), [3.5, 5.5])
    // A constant image stays constant at a non-integer ratio.
    let flat = [Float](repeating: 2, count: 294 * 518)
    let out = ReflexGeometry.areaResize(flat, w: 294, h: 518, toW: 108, toH: 192)
    XCTAssertEqual(out.count, 108 * 192)
    XCTAssertEqual(out.min()!, 2, accuracy: 1e-5)
    XCTAssertEqual(out.max()!, 2, accuracy: 1e-5)
  }

  func testPersistKeepsTheFartherOfTwoFrames() {
    var p = Persist(2)
    XCTAssertEqual(p([3, 0.4, 2]), [3, 0.4, 2])
    // One near frame alone does not bring a strip closer.
    XCTAssertEqual(p([0.3, 0.4, 2]), [3, 0.4, 2])
    XCTAssertEqual(p([0.3, 0.5, 2]), [0.3, 0.5, 2])
  }

  /// Per-frame cost. The geometry runs on the golden frames; the model runs when the app was
  /// built with ReflexDepth.mlpackage. Printed as "reflex-timing" lines for docs/perf.md.
  func testPerFrameTiming() throws {
    let g = try golden()
    let c = g.cases[1]
    var tracker = ReflexTracker(eyeH: g.eye_h, prior: g.scale)
    var geo: [Double] = []
    for k in 0..<30 {
      let t0 = CFAbsoluteTimeGetCurrent()
      _ = tracker.step(depth: c.depth, w: c.width, h: c.height, hfovDeg: g.hfov, now: Double(k) / 10)
      geo.append(1000 * (CFAbsoluteTimeGetCurrent() - t0))
    }
    print("reflex-timing geometry_ms", Self.summary(geo))

    guard let model = PhoneReflex.loadModel() else { return }
    let buffer = try XCTUnwrap(Self.grey(PhoneReflex.inputSize.w, PhoneReflex.inputSize.h))
    let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
    var net: [Double] = []
    var full: [Double] = []
    for k in 0..<21 {
      let t0 = CFAbsoluteTimeGetCurrent()
      let out = try model.prediction(from: input)
      let raw = PhoneReflex.floats(try XCTUnwrap(out.featureValue(for: "depth")?.multiArrayValue))
      let t1 = CFAbsoluteTimeGetCurrent()
      let small = ReflexGeometry.areaResize(raw, w: 294, h: 518, toW: 108, toH: 192)
      _ = tracker.step(depth: small, w: 108, h: 192, hfovDeg: g.hfov, now: 10 + Double(k) / 10)
      let t2 = CFAbsoluteTimeGetCurrent()
      if k > 0 {  // the first call compiles for the device
        net.append(1000 * (t1 - t0))
        full.append(1000 * (t2 - t0))
      }
      XCTAssertEqual(raw.count, 294 * 518)
    }
    print("reflex-timing depth_ms", Self.summary(net))
    print("reflex-timing frame_ms", Self.summary(full))
  }

  private static func summary(_ ms: [Double]) -> String {
    let s = ms.sorted()
    let p95 = s[min(s.count - 1, Int((Double(s.count) * 0.95).rounded(.up)) - 1)]
    return String(format: "n=%d p50=%.1f p95=%.1f max=%.1f", s.count, s[s.count / 2], p95, s.last!)
  }

  private static func grey(_ w: Int, _ h: Int) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb)
    guard let pb else { return nil }
    CVPixelBufferLockBaseAddress(pb, [])
    memset(CVPixelBufferGetBaseAddress(pb), 128, CVPixelBufferGetDataSize(pb))
    CVPixelBufferUnlockBaseAddress(pb, [])
    return pb
  }
}
