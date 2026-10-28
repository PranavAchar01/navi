//
// MetaSightWatchApp.swift
//
// The watch side of meta-sight: haptic obstacle warnings, like a parking sensor on the
// wrist, from the sector distances the iPhone app forwards.
//

import SwiftUI

@main
struct MetaSightWatchApp: App {
  @State private var model = WatchModel()

  var body: some Scene {
    WindowGroup {
      WatchView(model: model)
        #if DEBUG
          // Simulator demo: `simctl launch <watch> com.pranavachar.metasight.watchkitapp --metasight-watch-on`
          .task { if ProcessInfo.processInfo.arguments.contains("--metasight-watch-on") { model.start() } }
        #endif
    }
  }
}
