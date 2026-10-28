//
// WatchView.swift
//
// One glance: the nearest obstacle's direction and distance in the largest type the
// screen holds, on a background that turns yellow, orange and red as it gets closer, and
// one Start / Stop button. VoiceOver reads the whole warning as a sentence.
//

import SwiftUI

struct WatchView: View {
  var model: WatchModel

  var body: some View {
    VStack(spacing: 6) {
      Text(headline)
        .font(.system(size: 34, weight: .heavy, design: .rounded))
        .multilineTextAlignment(.center)
        .lineLimit(2)
        .minimumScaleFactor(0.5)
        .foregroundStyle(foreground)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityLabel(spoken)
        .accessibilityAddTraits(.updatesFrequently)

      if let note = model.note {
        Text(note)
          .font(.system(size: 13, weight: .semibold))
          .multilineTextAlignment(.center)
          .foregroundStyle(foreground)
      }

      // A black capsule while on and a white one while off: readable on every band colour.
      Button { model.toggle() } label: {
        Text(model.isOn ? "Stop" : "Start")
          .font(.system(size: 22, weight: .heavy, design: .rounded))
          .foregroundStyle(model.isOn ? .white : .black)
          .frame(maxWidth: .infinity, minHeight: 48)
          .background(model.isOn ? Color.black : Color.white, in: Capsule())
      }
      .buttonStyle(.plain)
      .fixedSize(horizontal: false, vertical: true)
      .accessibilityLabel(model.isOn ? "Stop warnings" : "Start warnings")
      .accessibilityHint(model.isOn ? "Turns the haptic warnings off" : "Turns on haptic warnings for nearby obstacles")
    }
    .padding(.horizontal, 6)
    .containerBackground(background.gradient, for: .navigation)
    .background(background.ignoresSafeArea())
  }

  private var headline: String {
    guard model.isOn else { return "Off" }
    guard model.isFresh else { return "No signal" }
    guard let w = model.warning else { return "Clear" }
    return w.band == .stop ? "STOP\n\(w.directionName)" : "\(w.directionName)\n\(w.distanceText)"
  }

  private var spoken: String {
    guard model.isOn else { return "Warnings off" }
    guard model.isFresh else { return "No signal from the phone" }
    return model.warning?.spoken ?? "Path clear"
  }

  private var background: Color {
    guard model.isOn, let band = model.warning?.band else { return .black }
    switch band {
    case .stop: return .red
    case .fast: return .orange
    case .slow: return .yellow
    case .clear: return .black
    }
  }

  private var foreground: Color {
    switch model.warning?.band {
    case .slow, .fast: return model.isOn ? .black : .white
    default: return .white
    }
  }
}
