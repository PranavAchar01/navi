//
// PhoneReflex.swift
//
// The stop buzz without the Mac. While the server link is down, camera frames that would have
// gone to the server run through the phone's own reflex instead: Depth Anything V2
// Metric-Indoor Small on Core ML (ReflexDepth.mlpackage, made by scripts/coreml_depth.py), then
// the server's floor fit and belt plan (ReflexGeometry.swift), then the belt over BLE. Only the
// three camera sectors are known (no map on the phone), and there are no labels, guide or
// speaker cues: just the obstacle and drop warnings. When the server comes back it owns the
// belt again on its next command.
//

import CoreImage
import CoreML
import Foundation
import os

/// Lens and body numbers for the reflex, as metasight.glasses.toml sets them on the server.
struct ReflexConfig: Sendable {
  var eyeHeightM = 1.22
  var depthScalePrior = 0.45
  var hfovDeg = 44.8

  static let glasses = ReflexConfig()
}

/// Per-frame cost, for the settings chip and docs/perf.md.
struct ReflexTiming: Sendable {
  var depthMs: Double
  var geometryMs: Double
  var totalMs: Double { depthMs + geometryMs }
}

final class PhoneReflex: @unchecked Sendable {
  static let inputSize = (w: 294, h: 518)
  private static let logger = Logger(subsystem: "com.metasight", category: "PhoneReflex")

  /// Where packets go: the belt's BLE link, set by the server link when it starts.
  private let belt = OSAllocatedUnfairLock<BeltBLE?>(initialState: nil)
  private let queue = DispatchQueue(label: "metasight.reflex", qos: .userInteractive)
  private let busy = OSAllocatedUnfairLock(initialState: false)
  private let context = CIContext(options: [.cacheIntermediates: false])
  // Touched only on `queue`.
  private var model: MLModel?
  private var tracker: ReflexTracker
  private var persist = Persist(2)  // two frames near before a motor fires, as on the server
  private var seq: UInt16 = 0
  private var config: ReflexConfig
  private let lastTiming = OSAllocatedUnfairLock<ReflexTiming?>(initialState: nil)

  /// The last frame's cost, nil until one has run.
  var timing: ReflexTiming? { lastTiming.withLock { $0 } }

  init(config: ReflexConfig = .glasses) {
    self.config = config
    tracker = ReflexTracker(eyeH: config.eyeHeightM, prior: config.depthScalePrior)
  }

  func attach(_ ble: BeltBLE) { belt.withLock { $0 = ble } }

  /// The bundled model, compiled into the app by Xcode. Nil if the app was built without it.
  static func loadModel() -> MLModel? {
    guard let url = Bundle.main.url(forResource: "ReflexDepth", withExtension: "mlmodelc")
    else { return nil }
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .all
    return try? MLModel(contentsOf: url, configuration: cfg)
  }

  /// Offers an upright frame. Dropped while the previous one is still running (latest wins),
  /// so the belt never acts on a queue of old frames. `focalPx` (along the image's width, in
  /// its pixels) gives the lens when the camera knows it, as the phone's own camera does.
  func offer(_ image: CIImage, focalPx: Double? = nil) {
    let claimed = busy.withLock { b in
      if b { return false }
      b = true
      return true
    }
    guard claimed else { return }
    queue.async { [self] in
      defer { busy.withLock { $0 = false } }
      let crop = Self.crop916(image.extent)
      if let focalPx { config.hfovDeg = 2 * atan(Double(crop.width) / 2 / focalPx) * 180 / .pi }
      if model == nil { model = Self.loadModel() }
      guard let model, let ble = belt.withLock({ $0 }),
        let packet = run(image.cropped(to: crop), model: model)
      else { return }
      Task { @MainActor in ble.write(packet) }
    }
  }

  /// One frame to one belt packet. Runs on `queue`.
  private func run(_ image: CIImage, model: MLModel) -> Data? {
    let t0 = CFAbsoluteTimeGetCurrent()
    guard let buffer = pixelBuffer(image),
      let out = try? model.prediction(from: ReflexInput(buffer)),
      let depth = out.featureValue(for: "depth")?.multiArrayValue
    else { return nil }
    let raw = Self.floats(depth)
    let t1 = CFAbsoluteTimeGetCurrent()
    let g = ReflexGeometry.grid
    let small = ReflexGeometry.areaResize(
      raw, w: Self.inputSize.w, h: Self.inputSize.h, toW: g.h, toH: g.w)
    let packet = plan(depth: small, w: g.h, h: g.w, now: Date().timeIntervalSince1970)
    let t2 = CFAbsoluteTimeGetCurrent()
    let timing = ReflexTiming(depthMs: 1000 * (t1 - t0), geometryMs: 1000 * (t2 - t1))
    lastTiming.withLock { $0 = timing }
    if seq % 30 == 1 {
      Self.logger.info(
        "reflex depth \(timing.depthMs, format: .fixed(precision: 1)) ms, geometry \(timing.geometryMs, format: .fixed(precision: 1)) ms"
      )
    }
    return packet
  }

  /// Grid-sized raw depth to the belt packet: floor fit, persistence, plan.
  func plan(depth: [Float], w: Int, h: Int, now: Double) -> Data {
    let (scene, ttc) = tracker.step(depth: depth, w: w, h: h, hfovDeg: config.hfovDeg, now: now)
    let live = persist(scene.sectors)
    let (cmd, flags) = BeltPlan.plan(camera: live, pathM: scene.pathM, ttcS: ttc, dropM: scene.dropM)
    seq &+= 1
    let tMs = UInt32(truncatingIfNeeded: Int64(now * 1000))
    return BeltPlan.packet(cmd, flags: flags, seq: seq, tMs: tMs)
  }

  /// The centred 9:16 crop the server is sent (PhoneCamera's FrameEncoder); a no-op for the
  /// glasses' stream, which already is 9:16.
  static func crop916(_ e: CGRect) -> CGRect {
    let w = min(e.width, e.height * 9 / 16)
    let h = min(e.height, e.width * 16 / 9)
    return CGRect(x: e.midX - w / 2, y: e.midY - h / 2, width: w, height: h)
  }

  /// The frame stretched to the model's 294 x 518 input, as the server resizes before its net.
  private func pixelBuffer(_ image: CIImage) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    CVPixelBufferCreate(
      nil, Self.inputSize.w, Self.inputSize.h, kCVPixelFormatType_32BGRA, attrs, &pb)
    guard let pb else { return nil }
    let e = image.extent
    let scaled = image.transformed(by: .init(translationX: -e.minX, y: -e.minY))
      .transformed(
        by: .init(
          scaleX: CGFloat(Self.inputSize.w) / e.width, y: CGFloat(Self.inputSize.h) / e.height))
    context.render(scaled, to: pb)
    return pb
  }

  /// A (1, H, W) depth array as row-major floats; rows may be padded (the Neural Engine
  /// aligns them), so it goes by the strides.
  static func floats(_ a: MLMultiArray) -> [Float] {
    let h = a.shape[1].intValue
    let w = a.shape[2].intValue
    let rs = a.strides[1].intValue
    let cs = a.strides[2].intValue
    var out = [Float](repeating: 0, count: h * w)
    func fill<T: MLShapedArrayScalar>(_ type: T.Type, _ f: (T) -> Float) {
      a.withUnsafeBufferPointer(ofType: type) { p in
        for r in 0..<h { for c in 0..<w { out[r * w + c] = f(p[r * rs + c * cs]) } }
      }
    }
    switch a.dataType {
    case .float16: fill(Float16.self) { Float($0) }
    case .float32: fill(Float.self) { $0 }
    default:
      for r in 0..<h { for c in 0..<w { out[r * w + c] = a[[0, r, c] as [NSNumber]].floatValue } }
    }
    return out
  }
}

/// The model's one input, a 294 x 518 BGRA buffer (normalisation is inside the model).
private final class ReflexInput: MLFeatureProvider {
  let buffer: CVPixelBuffer
  init(_ buffer: CVPixelBuffer) { self.buffer = buffer }
  var featureNames: Set<String> { ["image"] }
  func featureValue(for name: String) -> MLFeatureValue? {
    name == "image" ? MLFeatureValue(pixelBuffer: buffer) : nil
  }
}
