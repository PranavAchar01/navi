//
// BeltView.swift
//
// The haptic belt seen from above, front at the top: the wearer in the middle, the
// strap ring, and 8 motors numbered clockwise from the front (1 front, 3 right,
// 5 back, 7 left). Each motor lights by its duty and is coloured by its mode.
//

import SwiftUI

struct BeltView: View {
  var belt: BeltState
  /// Degrees, 0 = ahead, positive = right. Draws a dotted guide line when set.
  var guideBearing: Double?

  private static let amber = Color(red: 255 / 255, green: 190 / 255, blue: 70 / 255)
  private static let red = Color(red: 255 / 255, green: 72 / 255, blue: 72 / 255)
  private static let blue = Color(red: 90 / 255, green: 200 / 255, blue: 255 / 255)
  private static let purple = Color(red: 170 / 255, green: 120 / 255, blue: 255 / 255)

  var body: some View {
    TimelineView(.animation) { timeline in
      let t = timeline.date.timeIntervalSinceReferenceDate
      Canvas { context, size in
        draw(in: &context, size: size, time: t)
      }
    }
    .aspectRatio(1, contentMode: .fit)
    .accessibilityElement()
    .accessibilityLabel(caption)
  }

  private var caption: String {
    if belt.lost { return "BELT LINK LOST" }
    if belt.found { return "FOUND IT" }
    return "BELT"
  }

  // MARK: - Drawing

  private func draw(in context: inout GraphicsContext, size: CGSize, time t: Double) {
    let side = min(size.width, size.height)
    let center = CGPoint(x: size.width / 2, y: size.height / 2)
    let ringRadius = side * 0.36
    let motorRadius = side * 0.075

    // Strap ring.
    let ring = Path(ellipseIn: CGRect(x: center.x - ringRadius, y: center.y - ringRadius, width: ringRadius * 2, height: ringRadius * 2))
    context.stroke(ring, with: .color(.white.opacity(0.35)), lineWidth: max(2, side * 0.02))

    // Guide line toward the target.
    if let bearing = guideBearing {
      let a = bearing * .pi / 180
      var line = Path()
      line.move(to: center)
      line.addLine(to: CGPoint(x: center.x + sin(a) * ringRadius * 1.05, y: center.y - cos(a) * ringRadius * 1.05))
      context.stroke(line, with: .color(Self.blue), style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [2, 5]))
    }

    drawWearer(in: &context, center: center, ringRadius: ringRadius)

    for i in 0..<BeltState.motorCount {
      let angle = Double(i) * .pi / 4
      let anchor = CGPoint(x: center.x + sin(angle) * ringRadius, y: center.y - cos(angle) * ringRadius)
      drawMotor(index: i, at: anchor, radius: motorRadius, in: &context, time: t)
    }

    let captionColor: Color = belt.lost ? Self.purple : (belt.found ? Self.blue : .white.opacity(0.8))
    context.draw(
      Text(caption).font(.system(size: max(7, side * 0.055), weight: .bold)).foregroundStyle(captionColor),
      at: CGPoint(x: center.x, y: center.y + ringRadius * 0.52)
    )
  }

  private func drawWearer(in context: inout GraphicsContext, center: CGPoint, ringRadius: CGFloat) {
    let fill = GraphicsContext.Shading.color(.white.opacity(0.75))
    // Shoulders, then the head, then a small nose bump marking the front.
    let shoulders = CGRect(x: center.x - ringRadius * 0.48, y: center.y - ringRadius * 0.14, width: ringRadius * 0.96, height: ringRadius * 0.28)
    context.fill(Path(ellipseIn: shoulders), with: .color(.white.opacity(0.45)))
    let headRadius = ringRadius * 0.17
    context.fill(Path(ellipseIn: CGRect(x: center.x - headRadius, y: center.y - headRadius, width: headRadius * 2, height: headRadius * 2)), with: fill)
    var nose = Path()
    nose.move(to: CGPoint(x: center.x - headRadius * 0.35, y: center.y - headRadius * 0.9))
    nose.addLine(to: CGPoint(x: center.x, y: center.y - headRadius * 1.45))
    nose.addLine(to: CGPoint(x: center.x + headRadius * 0.35, y: center.y - headRadius * 0.9))
    nose.closeSubpath()
    context.fill(nose, with: fill)
  }

  private func drawMotor(index i: Int, at anchor: CGPoint, radius r: CGFloat, in context: inout GraphicsContext, time t: Double) {
    let mode = belt.modes[i]
    let duty = belt.lost ? 0.6 : Double(min(max(belt.duty[i], 0), 255)) / 255
    let color = motorColor(mode: mode)
    let lit = belt.lost || (color != nil && duty > 0)
    let level = lit ? duty : 0

    // Shake: up to ~2 px, proportional to duty.
    let shake = 2 * level
    let p = CGPoint(
      x: anchor.x + sin(t * 57 + Double(i) * 1.3) * shake,
      y: anchor.y + cos(t * 49 + Double(i) * 2.1) * shake
    )
    let disc = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))

    if mode == "unknown" && !belt.lost {
      // Unknown sector: dashed outline, unlit.
      context.fill(disc, with: .color(.black.opacity(0.5)))
      context.stroke(disc, with: .color(.white.opacity(0.6)), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
    } else if lit, let color = belt.lost ? Self.purple : color {
      // Glow.
      context.drawLayer { glow in
        glow.addFilter(.blur(radius: r * (0.4 + 0.8 * level)))
        let g = r * (1.2 + 0.5 * level)
        glow.fill(Path(ellipseIn: CGRect(x: p.x - g, y: p.y - g, width: g * 2, height: g * 2)), with: .color(color.opacity(0.75 * level)))
      }
      // Ripple ring while vibrating.
      let phase = (t * rippleRate(mode: mode)).truncatingRemainder(dividingBy: 1)
      let rr = r * (1 + 0.9 * phase)
      context.stroke(
        Path(ellipseIn: CGRect(x: p.x - rr, y: p.y - rr, width: rr * 2, height: rr * 2)),
        with: .color(color.opacity((1 - phase) * level)),
        lineWidth: 1.5
      )
      context.fill(disc, with: .color(color.opacity(0.3 + 0.7 * level)))
      context.stroke(disc, with: .color(color), lineWidth: 1)
    } else {
      context.fill(disc, with: .color(.black.opacity(0.5)))
      context.stroke(disc, with: .color(.white.opacity(0.35)), lineWidth: 1)
    }

    context.draw(
      Text("\(i + 1)").font(.system(size: r * 1.05, weight: .bold)).foregroundStyle(lit ? Color.black : Color.white.opacity(0.8)),
      at: p
    )
  }

  private func motorColor(mode: String) -> Color? {
    switch mode {
    case "slow": return Self.amber
    case "fast", "solid": return Self.red
    case "guide": return Self.blue
    default: return nil
    }
  }

  private func rippleRate(mode: String) -> Double {
    switch mode {
    case "fast": return 4
    case "slow": return 1.5
    default: return 2
    }
  }
}

#Preview {
  BeltView(
    belt: BeltState(
      duty: [255, 120, 0, 0, 0, 0, 60, 200],
      modes: ["fast", "slow", "off", "unknown", "off", "off", "slow", "guide"],
      lost: false,
      found: false,
      latencyMs: 12
    ),
    guideBearing: -40
  )
  .frame(width: 200, height: 200)
  .background(Color.black)
}
