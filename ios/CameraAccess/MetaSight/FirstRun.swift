import Foundation

/// The first time the app reaches the server it starts the home setup once; after that the
/// "Set up my home" button is the only way back in.
enum FirstRun {
  private static let key = "metasight.homeSetupStarted"

  static func shouldStart(step: String, defaults: UserDefaults = .standard) -> Bool {
    step == "idle" && !defaults.bool(forKey: key)
  }

  static func markStarted(defaults: UserDefaults = .standard) {
    defaults.set(true, forKey: key)
  }
}
