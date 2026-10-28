//
// DemoLaunch.swift
//
// Debug-only launch mode for the simulator: `--metasight-demo` pairs a MockDeviceKit
// pair of glasses, feeds it a video (METASIGHT_DEMO_VIDEO, a file path, or the
// bundled plant.mp4), and starts the session and preview without any taps. The mock camera
// only plays HEVC (hvc1) files; an H.264 file leaves the preview spinning with no frames.
//
//   xcrun simctl launch booted com.pranavachar.metasight --metasight-demo
//

#if DEBUG

import Foundation
import MWDATMockDevice

@MainActor
enum DemoLaunch {
  static let isEnabled = ProcessInfo.processInfo.arguments.contains("--metasight-demo")

  /// Call before the Wearables view model is created so it sees the registered state.
  static func configureMockDevice() {
    guard isEnabled else { return }
    MockDeviceKit.shared.enable()
    guard let glasses = try? MockDeviceKit.shared.pairGlasses(model: .rayBanMeta) else { return }
    glasses.powerOn()
    glasses.unfold()
    glasses.don()

    let env = ProcessInfo.processInfo.environment
    if let path = env["METASIGHT_DEMO_VIDEO"], FileManager.default.fileExists(atPath: path) {
      glasses.services.camera.setCameraFeed(fileURL: URL(fileURLWithPath: path))
    } else if let bundled = Bundle.main.url(forResource: "plant", withExtension: "mp4") {
      glasses.services.camera.setCameraFeed(fileURL: bundled)
    }
    if let server = env["METASIGHT_DEMO_SERVER"] {
      UserDefaults.standard.set(server, forKey: "metasight.serverURL")
    }
  }

  /// Starts the session, then the preview, once the mock device is active.
  static func autoStart(_ viewModel: CameraViewModel) async {
    guard isEnabled else { return }
    for _ in 0..<50 where !viewModel.hasActiveDevice {
      try? await Task.sleep(for: .milliseconds(200))
    }
    viewModel.startSession()
    for _ in 0..<50 where !viewModel.isSessionActive {
      try? await Task.sleep(for: .milliseconds(200))
    }
    await viewModel.startStreaming()
  }
}

#endif
