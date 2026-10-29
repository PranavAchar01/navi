//
// DemoLaunch.swift
//
// Debug-only launch mode for the simulator: `--metasight-demo` pairs a MockDeviceKit
// Meta Ray-Ban Display, feeds it a video (METASIGHT_DEMO_VIDEO, a file path, or the
// bundled plant.mp4), and starts the session and preview without any taps. The mock camera
// only plays HEVC (hvc1) files; an H.264 file leaves the preview spinning with no frames.
//
//   xcrun simctl launch booted com.pranavachar.metasight --metasight-demo
//
// The glasses' 600x600 display shows in a square on the camera screen, and in Chrome at
// http://127.0.0.1:9000/ with the "Meta Ray-Ban Display Simulator" extension (the mock kit's
// test server). METASIGHT_DEMO_HFOV gives the server the video's lens (a phone walk is ~40
// degrees, narrower than the glasses). scripts/sim.sh does all of this in one go.
//
// `--metasight-phone-demo` instead switches to the iPhone camera screen and presses Start;
// in the simulator the same video stands in for the camera.
//

#if DEBUG

import Foundation
import MWDATMockDevice
import SwiftUI

@MainActor
enum DemoLaunch {
  static let isEnabled = ProcessInfo.processInfo.arguments.contains("--metasight-demo")
  static let isPhoneDemo = ProcessInfo.processInfo.arguments.contains("--metasight-phone-demo")
  static let displayPreviewPort: UInt16 = 9000
  /// `--metasight-large-display` starts the in-app display preview enlarged.
  static let largeDisplayPreview = ProcessInfo.processInfo.arguments.contains("--metasight-large-display")
  /// `--metasight-settings` opens the settings sheet once the session is up (for screenshots).
  static let opensSettings = ProcessInfo.processInfo.arguments.contains("--metasight-settings")

  /// `--metasight-captions-demo` feeds the phone screen a recording with two speakers and a live
  /// partial line, for screenshots (bench/results/captions-sim.png).
  static let isCaptionsDemo = ProcessInfo.processInfo.arguments.contains("--metasight-captions-demo")

  static func feedCaptions(_ link: ServerLink) {
    guard isCaptionsDemo else { return }
    link.handle(#"{"type":"notes","state":"recording","say":""}"#)
    let lines: [(String, String)] = [
      ("you", "How has the knee been since the last visit?"),
      ("person 1", "Better, but the stairs still hurt in the morning."),
      ("you", "Is the ibuprofen helping with that?"),
      ("person 1", "Take 400 milligrams twice a day with food."),
    ]
    for (who, text) in lines {
      link.handle(#"{"type":"caption","text":"\#(text)","who":"\#(who)","final":true}"#)
    }
    link.handle(#"{"type":"caption","text":"Come back on December third and we will","final":false}"#)
  }

  /// The paired mock glasses, for the in-app display preview.
  private(set) static var glasses: (any MockGlasses)?

  /// Horizontal field of view of the demo video, when it was not shot by the glasses.
  static var videoHFOV: Double? {
    guard isEnabled, let raw = ProcessInfo.processInfo.environment["METASIGHT_DEMO_HFOV"] else { return nil }
    return Double(raw)
  }

  static func configurePhoneDemo() {
    guard isPhoneDemo else { return }
    UserDefaults.standard.set(CameraSource.iphone.rawValue, forKey: CameraSource.defaultsKey)
    if let server = ProcessInfo.processInfo.environment["METASIGHT_DEMO_SERVER"] {
      UserDefaults.standard.set(server, forKey: "metasight.serverURL")
    }
  }

  /// Call before the Wearables view model is created so it sees the registered state.
  static func configureMockDevice() {
    guard isEnabled else { return }
    UserDefaults.standard.set(CameraSource.glasses.rawValue, forKey: CameraSource.defaultsKey)
    MockDeviceKit.shared.enable()
    guard let glasses = try? MockDeviceKit.shared.pairGlasses(model: .metaRayBanDisplay) else { return }
    self.glasses = glasses
    glasses.powerOn()
    glasses.unfold()
    glasses.don()

    let env = ProcessInfo.processInfo.environment
    if let path = env["METASIGHT_DEMO_VIDEO"], FileManager.default.fileExists(atPath: path) {
      glasses.services.camera.setCameraFeed(fileURL: URL(fileURLWithPath: path))
      // A replayed walk is already on disk; don't save a second copy on the Mac.
      UserDefaults.standard.set(false, forKey: "metasight.saveWalk")
    } else if let bundled = Bundle.main.url(forResource: "plant", withExtension: "mp4") {
      glasses.services.camera.setCameraFeed(fileURL: bundled)
    }
    if let server = env["METASIGHT_DEMO_SERVER"] {
      UserDefaults.standard.set(server, forKey: "metasight.serverURL")
    }
    Task { _ = try? await MockDeviceKit.shared.startTestServer(port: displayPreviewPort) }
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

/// The mock glasses' 600x600 display, scaled into a square.
struct MockDisplayPreview: UIViewRepresentable {
  let glasses: any MockGlasses

  func makeUIView(context: Context) -> UIView {
    let container = UIView()
    container.backgroundColor = .black
    let preview = glasses.services.display.createPreviewView()
    preview.frame = container.bounds
    preview.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    container.addSubview(preview)
    return container
  }

  func updateUIView(_ uiView: UIView, context: Context) {}
}

#endif
