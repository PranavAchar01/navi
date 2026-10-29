//
// ReflexGeometry.swift
//
// The server's stop layer (metasight/safety.py `analyse` and `_geometry`, metasight/pan.py
// `Persist`, metasight/belt.py `plan` and `packet`) ported line for line, so the phone can
// buzz the belt on its own when the Mac is out of reach. Depth comes in at the analysis grid
// (108 x 192 for the upright 9:16 stream), already in raw metric units; everything after that
// matches the Python. ReflexGeometryTests replays the server's own results on his walk frames.
//

import Foundation

/// A plane `height(P) = P . n + c`, in raw depth units.
struct FloorPlane: Equatable, Sendable {
  var n: SIMD3<Double>
  var c: Double
}

/// What one frame's floor-aligned geometry says, in metres (safety.py `Scene`, minus the grid
/// and trip hazards, which only the judge view draws).
struct ReflexScene: Equatable, Sendable {
  var sectors: [Double]  // front-left, front, front-right: nearest horizontal range
  var pathM: Double  // nearest obstacle in the 0.9 m walking path
  var dropM: Double  // nearest drop below the floor in the path
  var floor: Bool  // a floor plane was found this frame
}

enum ReflexGeometry {
  static let grid = (w: 192, h: 108)  // landscape; an upright frame uses it turned
  static let bodyHalfM = 0.45
  static let sectorHalfDeg = 22.5
  static let ransacTries = 48

  /// Three distinct indices below `count` for one RANSAC try. The server seeds its generator
  /// the same way on every call, so a frame always gets the same floor.
  typealias Draw = (_ attempt: Int, _ count: Int) -> (Int, Int, Int)

  static func seededDraw() -> Draw {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next(_ bound: Int) -> Int {
      state &+= 0x9E37_79B9_7F4A_7C15
      var z = state
      z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
      z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
      return Int((z ^ (z >> 31)) % UInt64(bound))
    }
    return { _, count in
      let a = next(count)
      var b = next(count - 1)
      if b >= a { b += 1 }
      var c = next(count - 2)
      for x in [a, b].sorted() where c >= x { c += 1 }
      return (a, b, c)
    }
  }

  /// numpy's default percentile (linear between the two nearest ranks).
  static func percentile(_ values: [Double], _ q: Double) -> Double {
    let s = values.sorted()
    let pos = q / 100 * Double(s.count - 1)
    let lo = Int(pos.rounded(.down))
    let hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
  }

  /// RANSAC plane through the lower part of the view, normal near vertical, oriented up; the
  /// lowest level plane with real support wins (a bed is level too, but never under the floor).
  static func fitFloor(
    _ P: [SIMD3<Double>], w: Int, h: Int, prev: FloorPlane?, tol: Double, cLo: Double,
    cHi: Double, draw: Draw
  ) -> FloorPlane? {
    var low: [SIMD3<Double>] = []
    for i in (Int(Double(h) * 0.55) * w)..<(w * h) {
      let p = P[i]
      if p.x.isFinite && p.y.isFinite && p.z.isFinite && p.z > 0.1 { low.append(p) }
    }
    guard low.count >= 50 else { return nil }
    var cands = prev.map { [$0] } ?? []
    for k in 0..<ransacTries {
      let (i, j, l) = draw(k, low.count)
      let (a, b, c) = (low[i], low[j], low[l])
      var n = simdCross(b - a, c - a)
      let nn = simdLength(n)
      if nn < 1e-6 { continue }
      n /= nn
      if abs(n.y) < 0.75 { continue }  // camera y is down: the floor normal is mostly vertical
      if n.y > 0 { n = -n }
      cands.append(FloorPlane(n: n, c: -simdDot(a, n)))
    }
    let support = 0.1 * Double(low.count)
    var best: FloorPlane?
    for p in cands {
      guard cLo < p.c, p.c < cHi else { continue }
      let inl = low.reduce(0) { $0 + (abs(simdDot($1, p.n) + p.c) < tol ? 1 : 0) }
      if Double(inl) < support { continue }
      if best == nil || p.c > best!.c + tol { best = p }
    }
    guard var plane = best else { return nil }
    // Refine on the inliers (least squares), then keep the camera above it.
    let inl = low.filter { abs(simdDot($0, plane.n) + plane.c) < tol }
    let cen = inl.reduce(SIMD3<Double>()) { $0 + $1 } / Double(inl.count)
    var n2 = smallestEigenvector(covariance(inl, cen))
    if n2.y >= 0 { n2 = -n2 }
    if abs(n2.y) >= 0.75 { plane = FloorPlane(n: n2, c: -simdDot(cen, n2)) }
    return cLo < plane.c && plane.c < cHi ? plane : nil
  }

  /// Floor-aligned obstacle geometry for one frame (safety.py `analyse`). `depth` is row-major
  /// raw metric depth at the grid size; returns the scene and the plane if one was found.
  static func analyse(
    depth: [Float], w: Int, h: Int, scale: Double, prior: Double, hfovDeg: Double,
    eyeH: Double, prev: FloorPlane?, hold: SIMD3<Double>? = nil, draw: Draw? = nil
  ) -> (ReflexScene, FloorPlane?) {
    let count = w * h
    let f = 0.5 * Double(w) / tan(hfovDeg * .pi / 180 / 2)
    var P = [SIMD3<Double>](repeating: .zero, count: count)
    var ok = [Bool](repeating: false, count: count)
    for r in 0..<h {
      let v = Double(r) - Double(h) / 2 + 0.5
      for col in 0..<w {
        let i = r * w + col
        let u = Double(col) - Double(w) / 2 + 0.5
        let d = Double(depth[i])
        P[i] = SIMD3(u * d / f, v * d / f, d)
        ok[i] = d.isFinite && d > 0.05
      }
    }
    // Plane tolerance in raw units: 4 cm at the expected scale. The gate uses the lens prior.
    var plane = fitFloor(
      P, w: w, h: h, prev: prev, tol: 0.04 / scale, cLo: 0.75 * eyeH / prior,
      cHi: 1.3 * eyeH / prior, draw: draw ?? seededDraw())
    let found = plane != nil
    if plane == nil, let hold { plane = FloorPlane(n: hold, c: eyeH / scale) }

    var rng = [Double](repeating: 0, count: count)
    var bearing = [Double](repeating: 0, count: count)
    var hgt = [Double](repeating: .nan, count: count)
    var obst = [Bool](repeating: false, count: count)
    var corridor = [Bool](repeating: false, count: count)
    if let plane {
      let n = plane.n
      var fwd = SIMD3<Double>(0, 0, 1) - n.z * n
      fwd /= simdLength(fwd)
      var right = simdCross(n, fwd)
      if right.x < 0 { right = -right }
      let level = levelSurfaces(P, w: w, h: h, up: n)
      for i in 0..<count {
        hgt[i] = (simdDot(P[i], n) + plane.c) * scale
        let x = simdDot(P[i], right) * scale
        let z = simdDot(P[i], fwd) * scale
        rng[i] = (x * x + z * z).squareRoot()
        bearing[i] = atan2(x, z) * 180 / .pi
        obst[i] =
          ok[i] && hgt[i] < eyeH + 0.3 && ((hgt[i] > 0.05 && !level[i]) || hgt[i] > 0.3)
        corridor[i] = abs(x) < bodyHalfM && z > 0
      }
    } else {
      // No floor in view (looking up, or pressed against something): plain depth, top cut.
      let top = Int(0.3 * Double(h)) * w
      for i in 0..<count {
        let u = Double(i % w) - Double(w) / 2 + 0.5
        rng[i] = Double(depth[i]) * scale
        obst[i] = ok[i] && i >= top
        bearing[i] = atan2(u, f) * 180 / .pi
        corridor[i] = obst[i] && abs(bearing[i]) < 20
      }
    }

    func nearest(_ sel: (Int) -> Bool, minPx: Int = 12) -> Double {
      var vals: [Double] = []
      for i in 0..<count where sel(i) { vals.append(rng[i]) }
      return vals.count >= minPx ? percentile(vals, 5) : .infinity
    }
    let a = sectorHalfDeg
    let sectors = [(-67.5, -a), (-a, a), (a, 67.5)].map { lo, hi in
      nearest { obst[$0] && bearing[$0] >= lo && bearing[$0] < hi }
    }
    let path = nearest { obst[$0] && corridor[$0] }
    let drop =
      plane == nil
      ? Double.infinity
      : nearest(
        { ok[$0] && corridor[$0] && hgt[$0] < -0.25 && rng[$0] < 3.0 },
        minPx: Int(0.004 * Double(w * h)))
    return (
      ReflexScene(sectors: sectors, pathM: path, dropM: drop, floor: found), found ? plane : nil
    )
  }

  /// Pixels whose surface still faces up: the depth net bends a wide floor slightly, so a
  /// floor pixel can sit a few cm off the plane. 5x5 Gaussian (OpenCV's fixed kernel, mirrored
  /// edges), numpy-style gradients, normal from their cross product.
  static func levelSurfaces(_ P: [SIMD3<Double>], w: Int, h: Int, up: SIMD3<Double>) -> [Bool] {
    let k = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    func reflect(_ i: Int, _ n: Int) -> Int {
      var i = i
      if i < 0 { i = -i }
      if i >= n { i = 2 * n - 2 - i }
      return i
    }
    var tmp = [SIMD3<Double>](repeating: .zero, count: w * h)
    var Ps = tmp
    for r in 0..<h {
      for c in 0..<w {
        var s = SIMD3<Double>()
        for t in 0..<5 { s += k[t] * P[r * w + reflect(c + t - 2, w)] }
        tmp[r * w + c] = s
      }
    }
    for r in 0..<h {
      for c in 0..<w {
        var s = SIMD3<Double>()
        for t in 0..<5 { s += k[t] * tmp[reflect(r + t - 2, h) * w + c] }
        Ps[r * w + c] = s
      }
    }
    var level = [Bool](repeating: false, count: w * h)
    for r in 0..<h {
      for c in 0..<w {
        let gx =
          c == 0
          ? Ps[r * w + 1] - Ps[r * w]
          : c == w - 1
            ? Ps[r * w + c] - Ps[r * w + c - 1] : (Ps[r * w + c + 1] - Ps[r * w + c - 1]) / 2
        let gy =
          r == 0
          ? Ps[w + c] - Ps[c]
          : r == h - 1
            ? Ps[r * w + c] - Ps[(r - 1) * w + c]
            : (Ps[(r + 1) * w + c] - Ps[(r - 1) * w + c]) / 2
        let nrm = simdCross(gx, gy)
        level[r * w + c] = abs(simdDot(nrm / (simdLength(nrm) + 1e-9), up)) > 0.9
      }
    }
    return level
  }

  /// Area-average resize (OpenCV INTER_AREA when shrinking): each output pixel is the mean of
  /// the source pixels it covers, fractional edges weighted by overlap.
  static func areaResize(_ src: [Float], w: Int, h: Int, toW: Int, toH: Int) -> [Float] {
    let sx = Double(w) / Double(toW)
    let sy = Double(h) / Double(toH)
    func spans(_ o: Int, _ s: Double, _ n: Int) -> [(Int, Double)] {
      let a = Double(o) * s
      let b = a + s
      var out: [(Int, Double)] = []
      var i = Int(a.rounded(.down))
      while Double(i) < b && i < n {
        let wgt = min(b, Double(i + 1)) - max(a, Double(i))
        if wgt > 1e-9 { out.append((i, wgt)) }
        i += 1
      }
      return out
    }
    let cols = (0..<toW).map { spans($0, sx, w) }
    var out = [Float](repeating: 0, count: toW * toH)
    for r in 0..<toH {
      let rows = spans(r, sy, h)
      for c in 0..<toW {
        var s = 0.0
        var wsum = 0.0
        for (y, wy) in rows {
          for (x, wx) in cols[c] {
            s += Double(src[y * w + x]) * wy * wx
            wsum += wy * wx
          }
        }
        out[r * toW + c] = Float(s / wsum)
      }
    }
    return out
  }

  private static func covariance(_ pts: [SIMD3<Double>], _ cen: SIMD3<Double>) -> [[Double]] {
    var m = [[Double]](repeating: [0, 0, 0], count: 3)
    for p in pts {
      let d = p - cen
      for i in 0..<3 { for j in 0..<3 { m[i][j] += d[i] * d[j] } }
    }
    return m
  }

  /// The eigenvector of the smallest eigenvalue of a symmetric 3x3 (cyclic Jacobi): the last
  /// right-singular vector numpy's SVD gives for the centred inliers.
  static func smallestEigenvector(_ a: [[Double]]) -> SIMD3<Double> {
    var a = a
    var v: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for _ in 0..<50 {
      let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2]
      if off < 1e-22 { break }
      for (p, q) in [(0, 1), (0, 2), (1, 2)] where abs(a[p][q]) > 1e-300 {
        let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
        let c = 1 / (t * t + 1).squareRoot()
        let s = t * c
        for k in 0..<3 {
          let akp = a[k][p]
          let akq = a[k][q]
          a[k][p] = c * akp - s * akq
          a[k][q] = s * akp + c * akq
        }
        for k in 0..<3 {
          let apk = a[p][k]
          let aqk = a[q][k]
          a[p][k] = c * apk - s * aqk
          a[q][k] = s * apk + c * aqk
        }
        for k in 0..<3 {
          let vkp = v[k][p]
          let vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
    }
    let i = (0..<3).min { a[$0][$0] < a[$1][$1] }!
    return SIMD3(v[0][i], v[1][i], v[2][i])
  }
}

/// The server's per-frame state around `analyse` (safety.py `_geometry`): the floor as the
/// ruler for the running scale, the floor held through brief gaps, drops confirmed over 3 of 4
/// frames, and time to contact from the path's closing speed.
struct ReflexTracker {
  var eyeH: Double
  var prior: Double
  var scale: Double
  private var plane: FloorPlane?
  private var planeT = -Double.infinity
  private var dropHist: [Double] = []
  private var pathHist: [(t: Double, d: Double)] = []

  init(eyeH: Double, prior: Double) {
    self.eyeH = eyeH
    self.prior = prior
    scale = prior
  }

  mutating func step(depth: [Float], w: Int, h: Int, hfovDeg: Double, now: Double) -> (
    ReflexScene, ttc: Double
  ) {
    let prev = now - planeT < 1.5 ? plane : nil
    let hold = now - planeT < 5.0 ? plane?.n : nil
    var (scene, found) = ReflexGeometry.analyse(
      depth: depth, w: w, h: h, scale: scale, prior: prior, hfovDeg: hfovDeg, eyeH: eyeH,
      prev: prev, hold: hold)
    if let found {
      plane = found
      planeT = now
      // The floor is the ruler: the camera sits eyeH above it; the prior bounds one frame.
      let meas = min(max(eyeH / found.c, 0.8 * prior), 1.25 * prior)
      scale = 0.85 * scale + 0.15 * meas
      scene =
        ReflexGeometry.analyse(
          depth: depth, w: w, h: h, scale: scale, prior: prior, hfovDeg: hfovDeg, eyeH: eyeH,
          prev: found
        ).0
    }
    dropHist.append(scene.dropM)
    if dropHist.count > 4 { dropHist.removeFirst() }
    if dropHist.filter(\.isFinite).count < 3 { scene.dropM = .infinity }
    pathHist.append((now, scene.pathM))
    pathHist.removeAll { now - $0.t > 0.6 }
    let pts = pathHist.filter { $0.d.isFinite }
    var ttc = Double.infinity
    if pts.count >= 5 && scene.pathM.isFinite {
      let ts = pts.map { $0.t - pts.last!.t }
      let mt = ts.reduce(0, +) / Double(ts.count)
      let md = pts.map(\.d).reduce(0, +) / Double(pts.count)
      var num = 0.0
      var den = 0.0
      for (t, p) in zip(ts, pts) {
        num += (t - mt) * (p.d - md)
        den += (t - mt) * (t - mt)
      }
      let slope = den > 0 ? num / den : 0
      if slope < -0.15 { ttc = scene.pathM / -slope }  // closing faster than a slow shuffle
    }
    return (scene, ttc)
  }
}

/// A strip must stay near for `n` frames in a row before it counts (pan.py `Persist`).
struct Persist {
  let n: Int
  private var hist: [[Double]] = []

  init(_ n: Int) { self.n = max(1, n) }

  mutating func callAsFunction(_ strips: [Double]) -> [Double] {
    hist.append(strips)
    if hist.count > n { hist.removeFirst() }
    return strips.indices.map { i in hist.map { $0[i] }.max()! }
  }
}

/// The belt's command, as metasight/belt.py `plan` and `packet` build it.
enum BeltPlan {
  enum Mode: UInt8 { case off = 0, slow, fast, solid, unknown }
  static let flagDrop: UInt8 = 1
  static let sectorCount = 8
  /// Belt order (front, front-right, ... front-left): where the camera's three strips go.
  static let cameraSlots = [7, 0, 1]

  /// Per-sector (mode, 0..255) and flags. The phone has no map, so only the three camera
  /// sectors are known; the rest stay "unknown" (no buzz), as on the server with a stale map.
  static func plan(camera: [Double], pathM: Double, ttcS: Double, dropM: Double) -> (
    [(UInt8, UInt8)], UInt8
  ) {
    var sectors = [Double](repeating: .infinity, count: sectorCount)
    for (slot, d) in zip(cameraSlots, camera) { sectors[slot] = d }
    let out: [(UInt8, UInt8)] = sectors.enumerated().map { i, d0 in
      guard cameraSlots.contains(i) else { return (Mode.unknown.rawValue, 0) }
      let d = i == 0 ? min(d0, pathM) : d0
      var (mode, inten) = cue(d)
      if i == 0 && ttcS < 1.2 && d < 2.5 {
        mode = min(Mode.solid, Mode(rawValue: max(mode.rawValue, Mode.slow.rawValue) + 1)!)
        inten = max(inten, 0.8)
      }
      return (mode.rawValue, UInt8((255 * min(1, inten)).rounded(.toNearestOrEven)))
    }
    return (out, dropM < 1.5 ? flagDrop : 0)
  }

  /// Off beyond 2 m, slow pulse 2 to 1 m, fast 1 to 0.5 m, solid below (geometry.py).
  static func cue(_ d: Double) -> (Mode, Double) {
    if !d.isFinite || d >= 2 { return (.off, 0) }
    if d >= 1 { return (.slow, 0.35 + 0.25 * (2 - d)) }
    if d >= 0.5 { return (.fast, 0.6 + 0.3 * (1 - d) / 0.5) }
    return (.solid, 1)
  }

  /// The 26-byte command packet: "MS", version 1, flags, seq u16, t_ms u32, 8 x (mode, level).
  static func packet(_ cmd: [(UInt8, UInt8)], flags: UInt8, seq: UInt16, tMs: UInt32) -> Data {
    var d = Data([0x4D, 0x53, 1, flags])
    withUnsafeBytes(of: seq.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: tMs.littleEndian) { d.append(contentsOf: $0) }
    for (m, i) in cmd { d.append(contentsOf: [m, i]) }
    return d
  }
}

extension BeltPlan.Mode: Comparable {
  static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
}

@inline(__always) private func simdDot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
  (a * b).sum()
}

@inline(__always) private func simdLength(_ a: SIMD3<Double>) -> Double { simdDot(a, a).squareRoot() }

@inline(__always) private func simdCross(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
  SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}
