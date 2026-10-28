//
// CameraSource.swift
//
// Which camera feeds the server: the Ray-Ban Meta glasses, or the iPhone's own back
// camera (no glasses needed). Remembered between launches.
//

import SwiftUI

enum CameraSource: String, CaseIterable, Identifiable {
  case glasses
  case iphone

  static let defaultsKey = "metasight.source"

  var id: String { rawValue }

  var title: String {
    switch self {
    case .glasses: return "Glasses"
    case .iphone: return "iPhone camera"
    }
  }
}

/// The switch shown at the top of every screen.
struct SourcePicker: View {
  @Binding var source: CameraSource

  var body: some View {
    Picker("Camera", selection: $source) {
      ForEach(CameraSource.allCases) { source in
        Text(source.title).tag(source)
      }
    }
    .pickerStyle(.segmented)
    .accessibilityLabel("Camera source")
    .accessibilityValue(source.title)
    .accessibilityHint("Choose the glasses or this iPhone's camera")
    .accessibilityIdentifier("source_picker")
  }
}
