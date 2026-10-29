//
// GlassesDisplay.swift
//
// Puts meta-sight's cues on the Meta Ray-Ban Display's 600x600 screen: the nearest
// obstacle (direction, distance, red for fast/stop, yellow for slow), and, after a
// "where is" question, the item, where it was seen and an arrow toward it. Black
// background (clear on the lens), one coloured arrow, at most two lines of text.
//
// Attached alongside GlassesSensors once a session on display glasses has started.
// A card is only sent when what it says changes, so a steady scene costs nothing on
// the Bluetooth link.
//

import MWDATCore
import MWDATDisplay
import Observation
import UIKit

/// One screenful for the glasses.
struct DisplayCard: Equatable, Sendable {
  enum Glyph: Equatable, Sendable {
    /// Arrow toward an obstacle; degrees clockwise from straight ahead.
    case obstacle(bearing: Double, band: ProximityBand)
    /// Arrow toward the item being guided to.
    case target(bearing: Double)
  }

  var glyph: Glyph?
  var title: String
  var detail: String?
  /// Nothing to report: small, dim text.
  var quiet = false

  static let clear = DisplayCard(title: "Path clear", quiet: true)
  /// An answer stays up this long when there is no live guide to the item.
  static let answerSeconds: TimeInterval = 12

  /// Most urgent first: an obstacle within stopping distance, then the item being guided
  /// to (or the latest answer), then any nearer-than-2 m obstacle. Indoors something is
  /// nearly always within a metre, so only "stop" takes the screen from an answer; the
  /// belt and beeps still carry the slow and fast cues.
  static func make(
    warning: ProximityWarning?, guide: GuideTarget?, answer: AnswerNote?, now: Date = Date()
  ) -> DisplayCard {
    if let warning, warning.band == .stop { return obstacle(warning) }
    if let guide { return guiding(guide, answer: answer) }
    if let answer, now.timeIntervalSince(answer.at) < answerSeconds { return answered(answer) }
    if let warning { return obstacle(warning) }
    return .clear
  }

  private static func obstacle(_ w: ProximityWarning) -> DisplayCard {
    let glyph = Glyph.obstacle(bearing: Double(w.sector) * 45, band: w.band)
    if w.band == .stop {
      return DisplayCard(glyph: glyph, title: "Stop", detail: "\(w.directionName), \(w.distanceText)")
    }
    return DisplayCard(glyph: glyph, title: w.directionName, detail: w.distanceText)
  }

  /// "keys" / "kitchen counter · 3.0 m left", or "near the kitchen counter" with no fix on it.
  private static func guiding(_ g: GuideTarget, answer: AnswerNote?) -> DisplayCard {
    var parts: [String] = []
    if let answer, answer.label == g.label, let near = answer.near {
      parts.append(g.dist == nil && g.bearing == nil ? "near the \(near)" : near)
    }
    let way = g.bearing.map(directionWord)
    switch (g.dist, way) {
    case let (d?, w?): parts.append(String(format: "%.1f m %@", d, w))
    case let (d?, nil): parts.append(String(format: "%.1f m", d))
    case let (nil, w?): parts.append(w)
    case (nil, nil): break
    }
    return DisplayCard(
      glyph: g.bearing.map { .target(bearing: $0) },
      title: g.label,
      detail: parts.isEmpty ? nil : parts.joined(separator: " · "))
  }

  private static func answered(_ a: AnswerNote) -> DisplayCard {
    if let label = a.label {
      return DisplayCard(title: label, detail: a.near.map { "near the \($0)" } ?? clipped(a.text))
    }
    return DisplayCard(title: clipped(a.text))
  }

  /// Bearing in degrees (positive right) to the belt's eight directions.
  static func directionWord(_ bearing: Double) -> String {
    let sector = (Int((bearing / 45).rounded()) % 8 + 8) % 8
    return ["ahead", "ahead right", "right", "behind right", "behind", "behind left", "left", "ahead left"][sector]
  }

  /// Keeps a sentence to two display lines.
  static func clipped(_ text: String, limit: Int = 48) -> String {
    guard text.count > limit else { return text }
    return String(text.prefix(limit - 1)).trimmingCharacters(in: .whitespaces) + "…"
  }
}

@Observable
@MainActor
final class GlassesDisplay {
  /// "off", "starting", "on" or the last error.
  private(set) var status: String = "off"
  /// What the glasses are showing now.
  private(set) var shown: DisplayCard?

  @ObservationIgnored private let link: ServerLink
  @ObservationIgnored private var display: Display?
  @ObservationIgnored private let tokens = ListenerTokenBag()
  @ObservationIgnored private var refresher: Task<Void, Never>?

  init(link: ServerLink) {
    self.link = link
  }

  /// Call once a session on display glasses reaches `.started`.
  func attach(to session: DeviceSession) {
    guard display == nil else { return }
    do throws(DeviceSessionError) {
      let newDisplay = try session.addDisplay()
      display = newDisplay
      newDisplay.statePublisher.listen { [weak self] state in
        Task { @MainActor in self?.handleState(state) }
      }.store(in: tokens)
      status = "starting"
      newDisplay.start()
    } catch {
      status = error.localizedDescription
    }
  }

  /// Call when the session stops; the SDK tears the display down with it.
  func detach() {
    refresher?.cancel()
    refresher = nil
    tokens.clear()
    display = nil
    shown = nil
    status = "off"
  }

  private func handleState(_ state: DisplayState) {
    switch state {
    case .started:
      status = "on"
      startRefreshing()
    case .stopped:
      refresher?.cancel()
      refresher = nil
      shown = nil
      status = "off"
    case .starting, .stopping:
      break
    }
  }

  /// Four times a second, sends the card when it differs from what is showing.
  private func startRefreshing() {
    refresher?.cancel()
    refresher = Task { [weak self] in
      while !Task.isCancelled {
        await self?.refresh()
        try? await Task.sleep(for: .milliseconds(250))
      }
    }
  }

  private func refresh() async {
    guard let display else { return }
    let card = DisplayCard.make(
      warning: link.proximity?.warning, guide: link.guide, answer: link.answerNote)
    guard card != shown else { return }
    do {
      try await display.send(Self.view(for: card))
      shown = card
    } catch {
      status = error.localizedDescription
    }
  }

  // MARK: - Layout

  static func view(for card: DisplayCard) -> FlexBox {
    FlexBox(
      direction: .column, spacing: 20, alignment: .center, crossAlignment: .center,
      padding: EdgeInsets(all: 36)
    ) {
      if let glyph = card.glyph {
        Image(image: arrow(for: glyph), sizePreset: .fill)
      }
      Text(card.title, style: card.quiet ? .meta : .heading, color: card.quiet ? .secondary : .primary)
        .alignSelf(.center)
      if let detail = card.detail {
        Text(detail, style: .body)
          .alignSelf(.center)
      }
    }
    .flexGrow(1)
  }

  /// A thick arrow on black, turned toward the bearing: red for fast/stop, yellow for
  /// slow, white for an item. Drawn here so the colour survives the display's
  /// two-tone text styles.
  static func arrow(for glyph: DisplayCard.Glyph) -> UIImage {
    let (bearing, color): (Double, UIColor) =
      switch glyph {
      case let .obstacle(b, band): (b, band == .slow ? .systemYellow : .systemRed)
      case let .target(b): (b, .white)
      }
    // Wide and short: the display's fill preset spans the width, so this leaves room
    // for two lines of text under the arrow.
    let size = CGSize(width: 384, height: 160)
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).image { ctx in
      let cg = ctx.cgContext
      UIColor.black.setFill()
      cg.fill(CGRect(origin: .zero, size: size))
      cg.translateBy(x: size.width / 2, y: size.height / 2)
      cg.rotate(by: CGFloat(bearing) * .pi / 180)
      let path = UIBezierPath()
      path.move(to: CGPoint(x: 0, y: -66))
      path.addLine(to: CGPoint(x: 52, y: -6))
      path.addLine(to: CGPoint(x: 20, y: -6))
      path.addLine(to: CGPoint(x: 20, y: 62))
      path.addLine(to: CGPoint(x: -20, y: 62))
      path.addLine(to: CGPoint(x: -20, y: -6))
      path.addLine(to: CGPoint(x: -52, y: -6))
      path.close()
      color.setFill()
      path.fill()
    }
  }
}
