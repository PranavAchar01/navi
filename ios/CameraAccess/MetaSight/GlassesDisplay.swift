//
// GlassesDisplay.swift
//
// Puts meta-sight's cues on the Meta Ray-Ban Display's 600x600 screen: the nearest
// obstacle (an arrow and its distance, red for fast/stop, yellow for slow), and, after a
// "where is" question, the item, where and how long ago it was seen, and an arrow
// toward it (a pin when there is no live bearing). Black background (clear on the
// lens), one coloured glyph, at most two lines of large text, drawn as a single image
// so the type can be bigger than the SDK's text styles.
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
    if let guide { return guiding(guide, answer: answer, now: now) }
    if let answer, now.timeIntervalSince(answer.at) < answerSeconds { return answered(answer, now: now) }
    if let warning { return obstacle(warning) }
    return .clear
  }

  private static func obstacle(_ w: ProximityWarning) -> DisplayCard {
    let glyph = Glyph.obstacle(bearing: Double(w.sector) * 45, band: w.band)
    if w.band == .stop {
      return DisplayCard(glyph: glyph, title: "Stop", detail: "\(w.directionName) · \(distanceText(w.distance))")
    }
    return DisplayCard(glyph: glyph, title: distanceText(w.distance), detail: w.directionName)
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

  /// With a live fix: an arrow, "3.0 m" big and "keys" under it. With a bearing only, the
  /// item and the way to turn. With no fix, a pin over where and when it was last seen.
  private static func guiding(_ g: GuideTarget, answer: AnswerNote?, now: Date) -> DisplayCard {
    let note = answer?.label == g.label ? answer : nil
    switch (g.dist, g.bearing) {
    case let (d?, b?):
      let near = note?.near.map { "\(g.label) · \($0)" }
      let detail = near.flatMap { $0.count <= 22 ? $0 : nil } ?? g.label
      return DisplayCard(glyph: .target(bearing: b), title: distanceText(d), detail: detail)
    case let (nil, b?):
      return DisplayCard(glyph: .target(bearing: b), title: g.label, detail: directionWord(b).capitalizedFirst)
    case let (d?, nil):
      return DisplayCard(glyph: .pin, title: g.label, detail: distanceText(d))
    case (nil, nil):
      return DisplayCard(glyph: .pin, title: g.label, detail: lastSeen(note, now: now))
    }
  }

  /// "keys" / "sink · 9 min ago" under a pin.
  private static func answered(_ a: AnswerNote, now: Date) -> DisplayCard {
    if let label = a.label {
      return DisplayCard(glyph: .pin, title: label, detail: lastSeen(a, now: now) ?? clipped(a.text))
    }
    return DisplayCard(title: clipped(a.text))
  }

  private static func lastSeen(_ a: AnswerNote?, now: Date) -> String? {
    // The pin already says "here"; the place alone keeps the line large.
    let place = a?.near
    let when = a?.seen.map { ago($0, now: now) }
    let line = [place, when].compactMap { $0 }.joined(separator: " · ")
    return line.isEmpty ? nil : line
  }

  /// Bearing in degrees (positive right) to the belt's eight directions.
  static func directionWord(_ bearing: Double) -> String {
    let sector = (Int((bearing / 45).rounded()) % 8 + 8) % 8
    return ["ahead", "ahead right", "right", "behind right", "behind", "behind left", "left", "ahead left"][sector]
  }

  /// Keeps a sentence to three short lines.
  static func clipped(_ text: String, limit: Int = 48) -> String {
    guard text.count > limit else { return text }
    let head = text.prefix(limit - 1)
    // Break at the last whole word, never mid-word.
    let cut = head.lastIndex(of: " ").map { head[..<$0] } ?? head
    return cut.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters)) + "…"
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

  /// Side of the card image: the 600 px display less the box padding.
  static let cardSide: CGFloat = 528

  static func view(for card: DisplayCard) -> FlexBox {
    FlexBox(
      direction: .column, spacing: 0, alignment: .center, crossAlignment: .center,
      padding: EdgeInsets(all: 36)
    ) {
      if card.quiet {
        Text(card.title, style: .meta, color: .secondary)
          .alignSelf(.center)
      } else {
        Image(image: image(for: card), sizePreset: .fill)
      }
    }
    .flexGrow(1)
  }

  /// The whole card drawn as one square, so the type can be as big as the lens allows:
  /// glyph, a large first line, a smaller second line, centred as a group on black.
  /// Red for fast/stop, yellow for slow, white for an item; drawn here so the colour
  /// survives the display's two-tone text styles.
  static func image(for card: DisplayCard) -> UIImage {
    let side = cardSide
    let color: UIColor =
      switch card.glyph {
      case let .obstacle(_, band): band == .slow ? .systemYellow : .systemRed
      default: .white
      }
    let maxWidth = side - 16
    let titleFont: UIFont
    let titleLines: Int
    if card.glyph == nil {
      // A sentence with no item: wrap it over up to three lines.
      titleFont = .systemFont(ofSize: 54, weight: .bold)
      titleLines = 3
    } else {
      titleFont = fitted(card.title, weight: .heavy, size: 112, width: maxWidth)
      titleLines = 1
    }
    let center = NSMutableParagraphStyle()
    center.alignment = .center
    center.lineBreakMode = titleLines == 1 ? .byClipping : .byWordWrapping
    let title = NSAttributedString(
      string: card.title,
      attributes: [.font: titleFont, .foregroundColor: color, .paragraphStyle: center])
    let titleHeight = ceil(
      title.boundingRect(
        with: CGSize(width: maxWidth, height: .greatestFiniteMagnitude),
        options: [.usesLineFragmentOrigin], context: nil
      ).height)
    let detail = card.detail.map {
      NSAttributedString(
        string: $0,
        attributes: [
          .font: fitted($0, weight: .semibold, size: 46, width: maxWidth), .foregroundColor: UIColor.white.withAlphaComponent(0.85), .paragraphStyle: center,
        ])
    }
    let detailHeight = detail.map { ceil($0.size().height) } ?? 0
    let glyphHeight: CGFloat = card.glyph == nil ? 0 : 176
    let gap: CGFloat = 14
    let total =
      glyphHeight + (glyphHeight > 0 ? gap : 0) + titleHeight + (detail == nil ? 0 : gap + detailHeight)

    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format).image { ctx in
      UIColor.black.setFill()
      ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
      var y = (side - total) / 2
      if let glyph = card.glyph {
        drawGlyph(glyph, color: color, center: CGPoint(x: side / 2, y: y + glyphHeight / 2), in: ctx.cgContext)
        y += glyphHeight + gap
      }
      title.draw(
        with: CGRect(x: (side - maxWidth) / 2, y: y, width: maxWidth, height: titleHeight),
        options: [.usesLineFragmentOrigin], context: nil)
      y += titleHeight + gap
      detail?.draw(
        with: CGRect(x: (side - maxWidth) / 2, y: y, width: maxWidth, height: detailHeight),
        options: [.usesLineFragmentOrigin], context: nil)
    }
  }

  /// The largest size up to `size` at which `text` fits on one line of `width`.
  private static func fitted(_ text: String, weight: UIFont.Weight, size: CGFloat, width: CGFloat) -> UIFont {
    let font = UIFont.systemFont(ofSize: size, weight: weight)
    let w = (text as NSString).size(withAttributes: [.font: font]).width
    guard w > width else { return font }
    return .systemFont(ofSize: floor(size * width / w), weight: weight)
  }

  /// A thick arrow turned toward the bearing, or a map pin when the item has no live
  /// bearing; both about 176 px tall.
  private static func drawGlyph(_ glyph: DisplayCard.Glyph, color: UIColor, center: CGPoint, in cg: CGContext) {
    cg.saveGState()
    defer { cg.restoreGState() }
    cg.translateBy(x: center.x, y: center.y)
    color.setFill()
    if case .pin = glyph {
      // Teardrop map pin with a hole.
      let pin = UIBezierPath()
      pin.addArc(withCenter: CGPoint(x: 0, y: -30), radius: 56, startAngle: .pi * 0.8, endAngle: .pi * 0.2, clockwise: true)
      pin.addLine(to: CGPoint(x: 0, y: 88))
      pin.close()
      pin.append(UIBezierPath(ovalIn: CGRect(x: -21, y: -51, width: 42, height: 42)).reversing())
      pin.fill()
      return
    }
    let bearing: Double =
      switch glyph {
      case let .obstacle(b, _): b
      case let .target(b): b
      case .pin: 0
      }
    cg.rotate(by: CGFloat(bearing) * .pi / 180)
    let path = UIBezierPath()
    path.move(to: CGPoint(x: 0, y: -86))
    path.addLine(to: CGPoint(x: 68, y: -8))
    path.addLine(to: CGPoint(x: 26, y: -8))
    path.addLine(to: CGPoint(x: 26, y: 82))
    path.addLine(to: CGPoint(x: -26, y: 82))
    path.addLine(to: CGPoint(x: -26, y: -8))
    path.addLine(to: CGPoint(x: -68, y: -8))
    path.close()
    path.fill()
  }
}

private extension String {
  /// "ahead right" -> "Ahead right".
  var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
