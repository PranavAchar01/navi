//
// PhoneCamera.swift
//
// The iPhone's own back camera as the frame source, for walking without the glasses.
// Frames go out exactly like the glasses stream (504x896 portrait, JPEG quality 0.7,
// at most 24 a second, same binary packet), so the server needs nothing new to use them.
//
// ARKit world tracking drives the camera, which adds two things the glasses lack: the
// lens's real focal length (the server switches to this field of view while the phone is
// the source) and the phone's own pose for every frame, saved with the walk on the Mac as
// a reference path for scoring the map. The server never depends on the pose.
//
// Where ARKit is unavailable (the simulator) a video file stands in for the camera:
// METASIGHT_DEMO_VIDEO if set, else the bundled plant.mp4.
//

import ARKit
import AVFoundation
import CoreImage
import CoreMotion
import Observation
import UIKit
import os

@Observable
@MainActor
final class PhoneCamera {
  private static let saveWalkKey = "metasight.saveWalk"

  private(set) var isRunning = false
  /// ARKit tracking state, or what stands in for the camera.
  private(set) var tracking = "off"
  /// The last frame sent, refreshed a few times a second.
  private(set) var preview: UIImage?
  /// Horizontal field of view of the frames sent, measured from the lens.
  private(set) var hfov: Double?
  private(set) var framesSent = 0
  /// Folder name of the walk being saved on the Mac, or nil.
  private(set) var walkID: String?
  /// Save the walk (frames and phone poses) on the Mac for scoring the map later.
  var saveWalk: Bool {
    didSet { UserDefaults.standard.set(saveWalk, forKey: Self.saveWalkKey) }
  }

  let link: ServerLink

  @ObservationIgnored private var arFeed: ARFeed?
  @ObservationIgnored private var fileFeed: FileFeed?
  @ObservationIgnored private let motion = CMMotionManager()
  @ObservationIgnored private var feedName = "arkit"

  init(link: ServerLink) {
    self.link = link
    saveWalk = UserDefaults.standard.object(forKey: Self.saveWalkKey) as? Bool ?? true
  }

  func start() {
    guard !isRunning else { return }
    framesSent = 0
    hfov = nil
    walkID = saveWalk ? Self.walkStamp() : nil
    let uplink = PhoneUplink(link: link) { [weak self] jpeg, hfov in self?.didSend(jpeg, hfov: hfov) }

    if ARWorldTrackingConfiguration.isSupported {
      feedName = "arkit"
      tracking = "starting"
      let feed = ARFeed(uplink: uplink) { [weak self] state in self?.tracking = state }
      feed.start()
      arFeed = feed
    } else if let url = Self.standInVideo {
      feedName = "video"
      tracking = "no camera tracking here, playing \(url.lastPathComponent)"
      let feed = FileFeed(url: url, uplink: uplink)
      feed.start()
      fileFeed = feed
    } else {
      tracking = "this phone cannot track its camera"
      return
    }
    startMotion()
    isRunning = true
    announce()
    UIApplication.shared.isIdleTimerDisabled = true
  }

  func stop() {
    guard isRunning else { return }
    arFeed?.stop()
    arFeed = nil
    fileFeed?.stop()
    fileFeed = nil
    motion.stopDeviceMotionUpdates()
    isRunning = false
    tracking = "off"
    // Tells the server to close the walk and go back to the profile's lens.
    link.announce(nil)
    link.send(json: ["type": "camera", "source": "iphone", "walk": NSNull()])
    UIApplication.shared.isIdleTimerDisabled = false
  }

  private func announce() {
    var message: [String: Any] = [
      "type": "camera",
      "source": "iphone",
      "feed": feedName,
      "width": Int(FrameEncoder.width),
      "height": Int(FrameEncoder.height),
      "walk": walkID ?? NSNull(),
    ]
    if let hfov { message["hfov"] = (hfov * 100).rounded() / 100 }
    link.announce(message)
  }

  private func didSend(_ jpeg: Data, hfov measured: Double?) {
    framesSent += 1
    if let measured, abs(measured - (hfov ?? 0)) > 0.2 {
      hfov = measured
      announce()
    }
    if framesSent % 4 == 1 {
      preview = UIImage(data: jpeg)
    }
  }

  /// Device motion in the server's axes (x right, y up, z out of the screen): acceleration
  /// with gravity in m/s^2, reading about +9.8 on the axis pointing up, and rotation in rad/s.
  private func startMotion() {
    guard motion.isDeviceMotionAvailable else { return }
    motion.deviceMotionUpdateInterval = 1.0 / 30
    let link = self.link
    motion.startDeviceMotionUpdates(to: .main) { data, _ in
      guard let data else { return }
      let g = 9.81
      let a = data.gravity, u = data.userAcceleration, r = data.rotationRate
      let row = [
        data.timestamp,
        -(a.x + u.x) * g, -(a.y + u.y) * g, -(a.z + u.z) * g,
        r.x, r.y, r.z,
      ]
      MainActor.assumeIsolated { link.addMotion(row) }
    }
  }

  private static func walkStamp() -> String {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyyMMdd-HHmmss"
    return f.string(from: Date()) + "-iphone"
  }

  private static var standInVideo: URL? {
    if let path = ProcessInfo.processInfo.environment["METASIGHT_DEMO_VIDEO"],
      FileManager.default.fileExists(atPath: path)
    {
      return URL(fileURLWithPath: path)
    }
    return Bundle.main.url(forResource: "plant", withExtension: "mp4")
  }
}

// MARK: - Encoding

/// Any upright camera image to the glasses' stream format: centre-cropped to 9:16, scaled
/// to 504x896, JPEG quality 0.7.
final class FrameEncoder: Sendable {
  static let width: CGFloat = 504
  static let height: CGFloat = 896
  static let quality: CGFloat = 0.7

  private let context = CIContext(options: [.cacheIntermediates: false])
  private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

  /// The JPEG, and how many source pixels wide the crop was (for the field of view).
  func encode(_ image: CIImage) -> (jpeg: Data, cropWidth: CGFloat)? {
    let e = image.extent
    let aspect = Self.width / Self.height
    var crop = e
    if e.width / e.height > aspect {
      crop.size.width = e.height * aspect
      crop.origin.x += (e.width - crop.width) / 2
    } else {
      crop.size.height = e.width / aspect
      crop.origin.y += (e.height - crop.height) / 2
    }
    let scale = Self.width / crop.width
    let out = image.cropped(to: crop)
      .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
      .applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: scale, kCIInputAspectRatioKey: 1])
      .cropped(to: CGRect(x: 0, y: 0, width: Self.width, height: Self.height))
    let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): Self.quality]
    guard let jpeg = context.jpegRepresentation(of: out, colorSpace: colorSpace, options: options) else { return nil }
    return (jpeg, crop.width)
  }
}

/// Sends camera images off the main actor: paces them to 24 a second, encodes, and hands
/// each packet (with its pose just before it) to the server link.
final class PhoneUplink: Sendable {
  static let frameInterval: TimeInterval = 1.0 / 24

  private let link: ServerLink
  private let encoder = FrameEncoder()
  private let lastSent = OSAllocatedUnfairLock(initialState: -TimeInterval.infinity)
  private let onSent: @MainActor @Sendable (Data, Double?) -> Void

  init(link: ServerLink, onSent: @escaping @MainActor @Sendable (Data, Double?) -> Void) {
    self.link = link
    self.onSent = onSent
  }

  /// `focalPx` is the focal length along the image's horizontal axis, in source pixels.
  /// `pose` is camera-to-world, 16 floats column-major.
  func send(_ upright: CIImage, at time: TimeInterval, pose: [Double]?, tracking: String?, focalPx: Double?) {
    let due = lastSent.withLock { last in
      guard time - last >= Self.frameInterval - 0.004 else { return false }
      last = time
      return true
    }
    guard due, let seq = link.frameGate.tryAcquire() else { return }
    guard let (jpeg, cropWidth) = encoder.encode(upright) else {
      link.frameGate.release()
      return
    }
    let ts = Date().timeIntervalSince1970 * 1000
    let packet = ServerLink.framePacket(jpeg: jpeg, seq: seq, timestampMs: ts)
    let hfov = focalPx.map { 2 * atan(Double(cropWidth) / 2 / $0) * 180 / .pi }
    let link = self.link
    let onSent = self.onSent
    Task { @MainActor in
      if let pose {
        link.send(json: ["type": "arpose", "seq": Int(seq), "ts": ts, "m": pose, "track": tracking ?? ""])
      }
      link.sendFrame(packet)
      onSent(jpeg, hfov)
    }
  }
}

// MARK: - Feeds

/// ARKit world tracking on the back wide camera.
final class ARFeed: NSObject, ARSessionDelegate, @unchecked Sendable {
  private let session = ARSession()
  private let queue = DispatchQueue(label: "metasight.arkit", qos: .userInitiated)
  private let uplink: PhoneUplink
  private let onTracking: @MainActor @Sendable (String) -> Void

  init(uplink: PhoneUplink, onTracking: @escaping @MainActor @Sendable (String) -> Void) {
    self.uplink = uplink
    self.onTracking = onTracking
  }

  func start() {
    session.delegate = self
    session.delegateQueue = queue
    let config = ARWorldTrackingConfiguration()
    config.worldAlignment = .gravity
    session.run(config, options: [.resetTracking, .removeExistingAnchors])
  }

  func stop() {
    session.pause()
  }

  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    // The sensor is landscape; the phone is held upright, so turn the image a quarter.
    // After the turn the image's horizontal axis is the sensor's vertical one: fy.
    let image = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
    let t = frame.camera.transform
    let pose = [t.columns.0, t.columns.1, t.columns.2, t.columns.3].flatMap {
      [Double($0.x), Double($0.y), Double($0.z), Double($0.w)]
    }
    uplink.send(
      image,
      at: frame.timestamp,
      pose: pose,
      tracking: Self.describe(frame.camera.trackingState),
      focalPx: Double(frame.camera.intrinsics.columns.1.y)
    )
  }

  func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
    let text = Self.describe(camera.trackingState)
    let onTracking = self.onTracking
    Task { @MainActor in onTracking(text) }
  }

  func session(_ session: ARSession, didFailWithError error: Error) {
    let text = "camera stopped: \(error.localizedDescription)"
    let onTracking = self.onTracking
    Task { @MainActor in onTracking(text) }
  }

  static func describe(_ state: ARCamera.TrackingState) -> String {
    switch state {
    case .normal: return "tracking"
    case .notAvailable: return "tracking not available"
    case .limited(let reason):
      switch reason {
      case .initializing: return "starting, move the phone slowly"
      case .excessiveMotion: return "moving too fast"
      case .insufficientFeatures: return "too little detail to track"
      case .relocalizing: return "finding its place again"
      @unknown default: return "tracking limited"
      }
    }
  }
}

/// A video file played in real time, looping, where there is no camera to track.
final class FileFeed: @unchecked Sendable {
  private let url: URL
  private let uplink: PhoneUplink
  private var task: Task<Void, Never>?

  init(url: URL, uplink: PhoneUplink) {
    self.url = url
    self.uplink = uplink
  }

  func start() {
    let url = self.url, uplink = self.uplink
    task = Task.detached(priority: .userInitiated) { await Self.play(url, uplink) }
  }

  func stop() {
    task?.cancel()
  }

  private static func play(_ url: URL, _ uplink: PhoneUplink) async {
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
    let fps = Double((try? await track.load(.nominalFrameRate)) ?? 24)
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let orientation = Self.orientation(of: transform)
    let clock = ContinuousClock()
    while !Task.isCancelled {
      guard let reader = try? AVAssetReader(asset: asset) else { return }
      let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
      )
      reader.add(output)
      reader.startReading()
      let start = clock.now
      var n = 0
      while !Task.isCancelled, let sample = output.copyNextSampleBuffer() {
        guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
        let image = CIImage(cvPixelBuffer: pixels).oriented(orientation)
        uplink.send(image, at: CACurrentMediaTime(), pose: nil, tracking: nil, focalPx: nil)
        n += 1
        try? await Task.sleep(until: start + .seconds(Double(n) / max(fps, 1)), clock: clock)
      }
      reader.cancelReading()
    }
  }

  private static func orientation(of t: CGAffineTransform) -> CGImagePropertyOrientation {
    switch Int((atan2(t.b, t.a) * 180 / .pi).rounded()) {
    case 90: return .right
    case -90: return .left
    case 180, -180: return .down
    default: return .up
    }
  }
}
