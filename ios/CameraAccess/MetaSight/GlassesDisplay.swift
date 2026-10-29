//
// GlassesDisplay.swift
//
// Puts meta-sight's cues on the Meta Ray-Ban Display's 600x600 screen: the nearest
// obstacle (direction, distance, red for fast/stop, yellow for slow), and, after a
// "where is" question, the item, where and how long ago it was seen, and an arrow
// toward it (a pin when there is no live bearing). Black
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
    /// Map pin: an item with no live fix, from an answer or an earlier walk.
    case pin
  }

  var glyph: Glyph?
  var title: String
  var detail: String?
  /// Nothing to report: small, dim text.
  var quiet = false

  static let clear = DisplayCard(title: "Path clear", quiet: true)
  /// An answer stays up this long when there is no live guide to the item.
  static let answerSeconds: TimeInterval = 12
  /// A new card waits this long after the last one unless it is a new "stop", so the
  /// nearest-obstacle cue doesn't flicker between sectors and distances on the lens.
  static let minDwell: TimeInterval = 1

  var isStop: Bool {
    if case .obstacle(_, .stop) = glyph { return true }
    return false
  }

  /// Most urgent first: an obstacle within stopping distance, then the item being guided
  /// to (or the latest answer), then any nearer-than-2 m obstacle. Indoors something is
  /// nearly always within a metre, so only "stop" takes the screen from an answer; the
  /// belt and beeps still carry the slow and fast cues.
  static func make(
    warning: ProximityWarning?, guide: GuideTarget?, answer: AnswerNote?, now: Date = Date()
  ) -> DisplayCard {
    if let warning, warning.band == .stop { return obstacle(warning) }
    if let guide { return guiding(guide, answer: answer) }
    if let answer, now.timeIntervalSince(answer.at) < answerSeconds { return answered(answer, now: now) }
    if let warning { return obstacle(warning) }
    return .clear
  }

  private static func obstacle(_ w: ProximityWarning) -> DisplayCard {
    let glyph = Glyph.obstacle(bearing: Double(w.sector) * 45, band: w.band)
    if w.band == .stop {
      return DisplayCard(glyph: glyph, title: "Stop", detail: "\(w.directionName), \(distanceText(w.distance))")
    }
    return DisplayCard(glyph: glyph, title: w.directionName, detail: distanceText(w.distance))
  }

  /// Distances in steps a glance can use: 0.1 m under 1 m, 0.5 m to 3 m, whole metres beyond.
  static func distanceText(_ d: Double?) -> String {
    guard let d else { return "close" }
    let perMetre: Double = d < 1 ? 10 : d < 3 ? 2 : 1
    return String(format: "%.1f m", max(1, (d * perMetre).rounded()) / perMetre)
  }

  /// "just now", "4 min ago", "2 h ago".
  static func ago(_ seen: Date, now: Date) -> String {
    let s = max(0, now.timeIntervalSince(seen))
    if s < 60 { return "just now" }
    if s < 3600 { return "\(Int(s / 60)) min ago" }
    return "\(Int(s / 3600)) h ago"
  }

  /// "keys" / "kitchen counter · 3.0 m left", or "near the kitchen counter" with no fix on it.
  private static func guiding(_ g: GuideTarget, answer: AnswerNote?) -> DisplayCard {
    var parts: [String] = []
    if let answer, answer.label == g.label, let near = answer.near {
      parts.append(g.dist == nil && g.bearing == nil ? "near the \(near)" : near)
    }
    let way = g.bearing.map(directionWord)
    switch (g.dist, way) {
    case let (d?, w?): parts.append("\(distanceText(d)) \(w)")
    case let (d?, nil): parts.append(distanceText(d))
    case let (nil, w?): parts.append(w)
    case (nil, nil): break
    }
    return DisplayCard(
      glyph: g.bearing.map { .target(bearing: $0) } ?? .pin,
      title: g.label,
      detail: parts.isEmpty ? nil : parts.joined(separator: " · "))
  }

  /// "keys" / "near the sink · 9 min ago" under a pin.
  private static func answered(_ a: AnswerNote, now: Date) -> DisplayCard {
    if let label = a.label {
      let place = a.near.map { "near the \($0)" }
      let when = a.seen.map { ago($0, now: now) }
      let detail = [place, when].compactMap { $0 }.joined(separator: " · ")
      return DisplayCard(glyph: .pin, title: label, detail: detail.isEmpty ? clipped(a.text) : detail)
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
  @ObservationIgnored private var shownAt = Date.distantPast

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
    let escalates = card.isStop && shown?.isStop != true
    guard escalates || Date().timeIntervalSince(shownAt) >= DisplayCard.minDwell else { return }
    do {
      try await display.send(Self.view(for: card))
      shown = card
      shownAt = Date()
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
  /// slow, white for an item; a white pin when the item has no live bearing. Drawn here so the colour survives the display's
  /// two-tone text styles.
  static func arrow(for glyph: DisplayCard.Glyph) -> UIImage {
    let (bearing, color): (Double, UIColor) =
      switch glyph {
      case let .obstacle(b, band): (b, band == .slow ? .systemYellow : .systemRed)
      case let .target(b): (b, .white)
      case .pin: (0, .white)
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
      color.setFill()
      if case .pin = glyph {
        // Teardrop map pin with a hole, the same height as the arrow.
        let pin = UIBezierPath()
        pin.addArc(withCenter: CGPoint(x: 0, y: -24), radius: 42, startAngle: .pi * 0.8, endAngle: .pi * 0.2, clockwise: true)
        pin.addLine(to: CGPoint(x: 0, y: 66))
        pin.close()
        pin.append(UIBezierPath(ovalIn: CGRect(x: -16, y: -40, width: 32, height: 32)).reversing())
        pin.fill()
        return
      }
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
      path.fill()
    }
  }
}
