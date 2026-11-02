// The haptic belt in 3D, for the HUD and the judge screen. The strap is the 38 mm nylon webbing
// of the running belt, laid out almost flat as the wearer would see it looking down: the front
// in the middle, the right on the right, the buckle at both ends. The ESP32 DevKit sits on its
// breakout shield in the middle of the strap; each vibration module is a DiGiYes-style board
// (coin motor, MOSFET, flyback diode, power LED, a 3-pin header with its Dupont plug) wired
// back to the shield. The chips, headers and USB-C port are KiCad library models (web/public/
// belt, see SOURCES.md there); boards, strap and wiring are drawn here.
//
// A running motor shakes its board, wobbles its can, and sends ripples through the webbing
// around it, coloured from yellow (a light buzz) to red (full power). 4, 8 or 16 motors: 4 is
// the belt as built (all across the front), 8 adds four across the back (the HUD default),
// 16 is the highest-fidelity layout.

import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { type Cell, LAYOUTS, type Layout, PINS4, angles, heat, spread } from "./beltlayout";

const L = 0.78; // strap length, buckle end to buckle end (m)
// The strap lies in an arc of this radius, the front farthest away: nearly flat for the four
// front motors, curved round like the worn belt when the back motors are in view too.
const ARC: Record<Layout, number> = { 4: 0.6, 8: 0.3, 16: 0.3 };
let RC = ARC[8];
const STRAP_H = 0.038;
const STRAP_T = 0.0032;
const TILE = 0.032; // one repeat of the weave along the strap
const MOD_W = 0.024; // module board, along the strap
const MOD_H = 0.017;
const PCB = 0.0016;
const MOD_Y = 0.003; // modules ride a little above the strap's middle; cables run under them
const MOD_SCALE = 1.6; // modules drawn larger than life so the parts read at HUD size
const PAD = 0.0018; // hook-and-loop under the shield
const GUIDE: [number, number, number] = [18, 167, 118];
const CAPTION: Record<Layout, string> = {
  4: "4 motors · the belt as built",
  8: "8 motors",
  16: "16 motors · highest fidelity",
};

const PARTS = [
  "esp32_wroom_32e",
  "2n7002_sot23",
  "1x03_2.54",
  "1x16_2.54",
  "usb_c_hro_31m12",
  "ams1117_3v3",
  "ch340c",
  "led_0603",
  "c_0603",
  "1n4148w_sod123",
  "reset_skhllca010",
  "1206",
] as const;
type Part = (typeof PARTS)[number];

// ------------------------------------------------------------------------------ the strap
// s runs along the strap from the left end (-L/2) through the front (0) to the right end.
const sOf = (deg: number) => (deg / 360) * L;
// Webbing never lies dead straight: a slow wave in and out and a slight sag.
const lie = (s: number) => 0.0011 * Math.sin(s * 23 + 0.7) + 0.0005 * Math.sin(s * 57 + 2.1);
const sag = (s: number) => 0.0009 * Math.sin(s * 9.5 + 0.3);
// The belt is seen from outside, as it sits round a waist: the middle bows toward the view and
// the ends wrap away behind, with the boards on the face that looks at you.
/** Out of the strap's outer face, towards the view. */
const normal = (s: number) => new THREE.Vector3(Math.sin(s / RC), 0, Math.cos(s / RC));
/** Along the strap, left to right on screen. */
const along = (s: number) => new THREE.Vector3(Math.cos(s / RC), 0, -Math.sin(s / RC));
const point = (s: number, out: number, y: number) => {
  const f = s / RC;
  const o = out + lie(s);
  return new THREE.Vector3(RC * Math.sin(f) + Math.sin(f) * o, y + sag(s), -RC * (1 - Math.cos(f)) + Math.cos(f) * o);
};

/** A length of strap from s0 to s1: a flat section with round edges, the inner face split into
 *  rows so it can ripple. Keeps each vertex's rest position, s, height and normal for that. */
interface Strip {
  geo: THREE.BufferGeometry;
  base: Float32Array;
  s: Float32Array;
  y: Float32Array;
  nx: Float32Array;
  nz: Float32Array;
  rings: number;
  per: number;
  s0: number;
  s1: number;
}
function strip(s0: number, s1: number, h: number, th: number, out: number, y0: number, segs: number, rows = 14): Strip {
  const e = th / 2;
  const prof: [number, number][] = [];
  for (let j = 0; j <= rows; j++) prof.push([e, -h / 2 + e + ((h - th) * j) / rows]);
  for (let k = 1; k < 6; k++) {
    const a = (k / 6) * Math.PI;
    prof.push([e * Math.cos(a), h / 2 - e + e * Math.sin(a)]);
  }
  prof.push([-e, h / 2 - e], [-e, -h / 2 + e]);
  for (let k = 1; k < 6; k++) {
    const a = Math.PI + (k / 6) * Math.PI;
    prof.push([e * Math.cos(a), -h / 2 + e + e * Math.sin(a)]);
  }
  const P = prof.length;
  const rings = segs + 1;
  const pos = new Float32Array(rings * P * 3);
  const uv = new Float32Array(rings * P * 2);
  const sv = new Float32Array(rings * P);
  const yv = new Float32Array(rings * P);
  const nx = new Float32Array(rings * P);
  const nz = new Float32Array(rings * P);
  const idx: number[] = [];
  for (let i = 0; i < rings; i++) {
    const s = s0 + ((s1 - s0) * i) / segs;
    const n = normal(s);
    const c = point(s, out, y0);
    prof.forEach(([pn, py], j) => {
      const v = i * P + j;
      pos.set([c.x + n.x * pn, c.y + py, c.z + n.z * pn], v * 3);
      uv.set([s / TILE, (py + h / 2) / h], v * 2);
      sv[v] = s;
      yv[v] = y0 + py;
      nx[v] = n.x;
      nz[v] = n.z;
    });
  }
  for (let i = 0; i < segs; i++) {
    for (let j = 0; j < P; j++) {
      const a = i * P + j;
      const b = a + P;
      const a2 = i * P + ((j + 1) % P);
      const b2 = a2 + P;
      idx.push(a, b, a2, a2, b, b2);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(rings * P * 3), 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return { geo, base: pos.slice(), s: sv, y: yv, nx, nz, rings, per: P, s0, s1 };
}

// ------------------------------------------------------------------------------ textures
function canvasTex(w: number, h: number, draw: (c: CanvasRenderingContext2D) => void, srgb = true) {
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  draw(cv.getContext("2d")!);
  const t = new THREE.CanvasTexture(cv);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** Nylon webbing, one tile of TILE along and the full strap height: a 2/1 twill of 0.6 mm
 *  yarns, each yarn its own shade, a tightly ribbed bound edge, and a row of lock stitching
 *  3 mm in from each edge. Gives a colour map and a normal map from the same yarn heights. */
function webbing() {
  const W = 1024;
  const H = 512;
  const mmU = (TILE * 1000) / W;
  const mmV = (STRAP_H * 1000) / H;
  let seed = 7;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  const shade = Array.from({ length: 128 }, () => 0.88 + 0.24 * rnd());
  const hgt = new Float32Array(W * H);
  const tone = new Float32Array(W * H);
  const thread = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const vy = y * mmV;
    const edge = Math.min(vy, STRAP_H * 1000 - vy);
    for (let x = 0; x < W; x++) {
      const ux = x * mmU;
      const o = y * W + x;
      let h: number;
      let t: number;
      if (edge < 2.3) {
        const p = (ux / 0.7) % 1;
        h = 0.5 + 0.5 * Math.sin(Math.PI * p) * (0.85 + 0.15 * Math.cos((vy / 0.45) * Math.PI * 2));
        t = 0.92;
      } else {
        const wi = Math.floor(vy / 0.6);
        const wf = vy / 0.6 - wi;
        const ci = Math.floor(ux / 0.6);
        const uf = ux / 0.6 - ci;
        const over = (ci + wi) % 3 === 0; // the weft crossing over this warp yarn
        h = over ? 0.3 + 0.4 * Math.sin(Math.PI * uf) : 0.4 + 0.6 * Math.sin(Math.PI * wf);
        t = shade[wi % 128] * (over ? 0.82 : 1);
      }
      if (edge > 2.75 && edge < 3.25 && ux % 3.2 < 2.3) {
        h = 1.2 + 0.2 * Math.sin((Math.PI * (ux % 3.2)) / 2.3);
        thread[o] = 1;
      }
      hgt[o] = h + (rnd() - 0.5) * 0.08;
      tone[o] = t * (0.95 + 0.1 * rnd());
    }
  }
  const at = (x: number, y: number) => hgt[Math.min(H - 1, Math.max(0, y)) * W + ((x + W) % W)];
  const nrm = canvasTex(
    W,
    H,
    (c) => {
      const img = c.createImageData(W, H);
      const k = 0.06; // yarn height per unit, mm
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const du = ((at(x + 1, y) - at(x - 1, y)) * k) / (2 * mmU);
          const dv = ((at(x, y - 1) - at(x, y + 1)) * k) / (2 * mmV); // canvas rows run down, v up
          const len = Math.hypot(du, dv, 1);
          const o = (y * W + x) * 4;
          img.data[o] = (-du / len) * 127.5 + 127.5;
          img.data[o + 1] = (-dv / len) * 127.5 + 127.5;
          img.data[o + 2] = (1 / len) * 127.5 + 127.5;
          img.data[o + 3] = 255;
        }
      }
      c.putImageData(img, 0, 0);
    },
    false,
  );
  const col = canvasTex(W, H, (c) => {
    const img = c.createImageData(W, H);
    for (let o = 0; o < W * H; o++) {
      const lit = 0.72 + 0.34 * hgt[o];
      const [r, g, b] = thread[o] ? [118, 122, 130] : [40, 42, 47];
      const t = thread[o] ? lit : tone[o] * lit;
      img.data[o * 4] = Math.min(255, r * t);
      img.data[o * 4 + 1] = Math.min(255, g * t);
      img.data[o * 4 + 2] = Math.min(255, b * t);
      img.data[o * 4 + 3] = 255;
    }
    c.putImageData(img, 0, 0);
  });
  for (const t of [nrm, col]) {
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
  }
  return { col, nrm };
}

const glowTex = () =>
  canvasTex(128, 128, (c) => {
    const g = c.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.25, "rgba(255,255,255,0.55)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    c.fillStyle = g;
    c.fillRect(0, 0, 128, 128);
  });

/** Soft shadow across a ribbon: dark in the middle, clear at both sides. */
const shadowTex = () =>
  canvasTex(8, 64, (c) => {
    const g = c.createLinearGradient(0, 0, 0, 64);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(0.5, "rgba(0,0,0,1)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    c.fillStyle = g;
    c.fillRect(0, 0, 8, 64);
  });

/** A board's top face: soldermask with its silkscreen, drawn in board millimetres (0,0 centre). */
function silk(wMm: number, hMm: number, mask: string, draw: (c: CanvasRenderingContext2D) => void) {
  const px = 24;
  return canvasTex(Math.round(wMm * px), Math.round(hMm * px), (c) => {
    c.fillStyle = mask;
    c.fillRect(0, 0, wMm * px, hMm * px);
    c.setTransform(px, 0, 0, px, (wMm / 2) * px, (hMm / 2) * px);
    c.strokeStyle = "rgba(245,245,240,0.92)";
    c.fillStyle = "rgba(245,245,240,0.92)";
    c.lineWidth = 0.18;
    c.textAlign = "center";
    c.textBaseline = "middle";
    draw(c);
  });
}
const label = (c: CanvasRenderingContext2D, s: string, x: number, y: number, mm: number, bold = false) => {
  c.font = `${bold ? 700 : 600} ${mm}px "Atkinson Hyperlegible Mono", ui-monospace, "SF Mono", Menlo, monospace`;
  c.fillText(s, x, y);
};

// ------------------------------------------------------------------------------ materials
function maskMat(map: THREE.Texture) {
  return new THREE.MeshPhysicalMaterial({
    map,
    roughness: 0.45,
    clearcoat: 0.35,
    clearcoatRoughness: 0.3,
    envMapIntensity: 0.6,
  });
}
const FR4 = new THREE.MeshStandardMaterial({ color: 0xcdbb8c, roughness: 0.8 });
const ENIG = new THREE.MeshStandardMaterial({ color: 0xdfb85e, metalness: 1, roughness: 0.24 });
const STEEL = new THREE.MeshStandardMaterial({ color: 0xc9ccd2, metalness: 1, roughness: 0.3 });
const PLASTIC = new THREE.MeshStandardMaterial({ color: 0x15161a, roughness: 0.55 });

/** KiCad's materials, Etch's rule: light unsaturated ones are metal, the rest plastic. */
function restyle(o: THREE.Object3D) {
  o.traverse((m) => {
    if (!(m instanceof THREE.Mesh)) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    for (const mat of mats) {
      if (!(mat instanceof THREE.MeshStandardMaterial)) continue;
      const hsl = { h: 0, s: 0, l: 0 };
      mat.color.getHSL(hsl);
      const metal = hsl.l > 0.45 && hsl.s < 0.25;
      mat.metalness = metal ? 0.6 : 0.05;
      mat.roughness = metal ? 0.36 : 0.48;
      mat.envMapIntensity = 0.8;
    }
  });
}

/** A board as an extruded rounded rectangle in its own XY plane, top face at z = PCB. */
function board(wM: number, hM: number, top: THREE.Material, holes: [number, number][] = []) {
  const s = new THREE.Shape();
  const r = 0.0012;
  s.moveTo(-wM / 2 + r, -hM / 2);
  s.lineTo(wM / 2 - r, -hM / 2);
  s.quadraticCurveTo(wM / 2, -hM / 2, wM / 2, -hM / 2 + r);
  s.lineTo(wM / 2, hM / 2 - r);
  s.quadraticCurveTo(wM / 2, hM / 2, wM / 2 - r, hM / 2);
  s.lineTo(-wM / 2 + r, hM / 2);
  s.quadraticCurveTo(-wM / 2, hM / 2, -wM / 2, hM / 2 - r);
  s.lineTo(-wM / 2, -hM / 2 + r);
  s.quadraticCurveTo(-wM / 2, -hM / 2, -wM / 2 + r, -hM / 2);
  for (const [x, y] of holes) {
    const h = new THREE.Path();
    h.absarc(x, y, 0.0011, 0, Math.PI * 2, true);
    s.holes.push(h);
  }
  const g = new THREE.ExtrudeGeometry(s, { depth: PCB, bevelEnabled: false, curveSegments: 10 });
  const map = (top as THREE.MeshPhysicalMaterial).map;
  if (map) {
    map.repeat.set(1 / wM, 1 / hM);
    map.offset.set(0.5, 0.5);
  }
  const m = new THREE.Mesh(g, [top, FR4]);
  for (const [x, y] of holes) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.0011, 0.0017, 20), ENIG);
    ring.position.set(x, y, PCB + 0.00002);
    m.add(ring);
  }
  return m;
}

// ------------------------------------------------------------------------------ the scene
interface Motor {
  s: number;
  root: THREE.Group;
  body: THREE.Group;
  wob: THREE.Group;
  can: THREE.MeshStandardMaterial;
  led: THREE.Sprite;
  level: number;
  born: number;
  /** Its heat colour, linear, for the ripple in the webbing. */
  lin: THREE.Color;
}

export const CAPTION_OF = CAPTION;

export class Belt3D {
  layout: Layout = 8;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(26, 1, 0.01, 10);
  private parts = new Map<Part, THREE.Object3D>();
  private ready: Promise<void>;
  private motors: Motor[] = [];
  private layoutGroup = new THREE.Group();
  private staticGroup = new THREE.Group();
  private pitch = 0.3;
  private guideMesh: THREE.Mesh | null = null;
  private guideAt: number | null = null;
  private glow = glowTex();
  private fabric = webbing();
  private strap: Strip | null = null;
  private rippled = false;
  private shield!: THREE.Object3D;
  private cells: Cell[] = Array.from({ length: 8 }, () => ({ rgb: null, level: 0 }));
  private caption: HTMLElement;
  private tourTimers: number[] = [];
  private visible = true;
  private dist = 0;
  private distTarget = 0.6;
  onLayout: ((n: Layout) => void) | null = null;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.className = "belt3d-canvas";
    host.appendChild(this.renderer.domElement);
    this.caption = document.createElement("div");
    this.caption.className = "belt3d-caption";
    host.appendChild(this.caption);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.6;
    // A soft key from the upper left, as through a window, and a white rim from behind that
    // catches the webbing's edge.
    const key = new THREE.DirectionalLight(0xfff6ec, 1.6);
    key.position.set(-0.5, 0.9, 0.8);
    const rim = new THREE.DirectionalLight(0xffffff, 0.7);
    rim.position.set(0.3, 0.6, -1);
    this.scene.add(key, rim, new THREE.HemisphereLight(0xffffff, 0xcfe2f7, 0.6), this.layoutGroup, this.staticGroup);
    this.ready = this.load();
    this.setLayout(this.layout);
    new ResizeObserver(() => this.resize()).observe(host);
    new IntersectionObserver(([e]) => (this.visible = e.isIntersecting)).observe(host);
    this.resize();
    this.renderer.setAnimationLoop((ms) => this.frame(ms));
  }

  /** The eight cells as the server drives them; the 3D belt spreads them onto its motors. */
  set(cells: Cell[], guideBearing: number | null = null) {
    this.cells = cells;
    this.setGuide(guideBearing);
  }

  setLayout(n: Layout) {
    this.layout = n;
    this.caption.textContent = CAPTION[n];
    this.host.dataset.motors = String(n);
    this.onLayout?.(n);
    RC = ARC[n];
    this.resize();
    void this.ready.then(() => {
      if (this.layout !== n) return;
      this.rebuild();
      this.buildMotors(n);
    });
  }

  /** The demo: 4, then 8, then 16 motors, then back to the HUD default of 8. */
  tour(dwellMs = 3500) {
    for (const id of this.tourTimers) clearTimeout(id);
    this.tourTimers = [];
    const steps: Layout[] = [4, 8, 16, 8];
    steps.forEach((n, i) => {
      this.tourTimers.push(window.setTimeout(() => this.setLayout(n), i * dwellMs));
    });
  }

  next() {
    this.setLayout(LAYOUTS[(LAYOUTS.indexOf(this.layout) + 1) % LAYOUTS.length]);
  }

  // ---------------------------------------------------------------------------- building
  private async load() {
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    await Promise.all(
      PARTS.map(async (name) => {
        const g = await loader.loadAsync(`/belt/${name}.glb`);
        const box = new THREE.Box3().setFromObject(g.scene);
        const c = box.getCenter(new THREE.Vector3());
        g.scene.position.set(-c.x, -box.min.y, -c.z);
        restyle(g.scene);
        const wrap = new THREE.Group();
        wrap.add(g.scene);
        this.parts.set(name, wrap);
      }),
    );
  }

  /** A KiCad part standing on a board face (board frame: x, y on the face, z out of it). */
  private put(parent: THREE.Object3D, name: Part, x: number, y: number, z: number, rotDeg = 0) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.z = (rotDeg * Math.PI) / 180;
    const p = this.parts.get(name)!.clone();
    p.rotation.x = Math.PI / 2;
    g.add(p);
    parent.add(g);
    return g;
  }

  /** A frame on the strap at s: x along the strap, y up, z out of the inner face. */
  private frameAt(s: number, out: number, y: number) {
    const g = new THREE.Group();
    const m = new THREE.Matrix4().makeBasis(along(s), new THREE.Vector3(0, 1, 0), normal(s));
    m.setPosition(point(s, out, y));
    m.decompose(g.position, g.quaternion, g.scale);
    return g;
  }

  /** Strap, buckle, shadow and controller for the current arc. */
  private rebuild() {
    for (const o of [...this.staticGroup.children]) {
      this.staticGroup.remove(o);
      o.traverse((m) => {
        if (m instanceof THREE.Mesh) m.geometry.dispose();
      });
    }
    this.rippled = false;
    this.guideAt = null;
    this.buildStatic();
    this.buildController();
  }

  private buildStatic() {
    const { col, nrm } = this.fabric;
    this.strap = strip(-L / 2, L / 2, STRAP_H, STRAP_T, 0, 0, 640);
    const strap = new THREE.Mesh(
      this.strap.geo,
      new THREE.MeshPhysicalMaterial({
        map: col,
        normalMap: nrm,
        normalScale: new THREE.Vector2(1, 1),
        roughness: 0.74,
        sheen: 1,
        sheenColor: new THREE.Color(0x6a7184),
        sheenRoughness: 0.42,
        side: THREE.DoubleSide,
      }),
    );
    // The ripple's colour, added over the weave so the yarn still shows through it.
    const warm = new THREE.Mesh(
      this.strap.geo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      }),
    );
    this.staticGroup.add(strap, warm);

    // Side-release buckle: the socket on the left end, the tongue on the right.
    const buckle = new THREE.MeshPhysicalMaterial({ color: 0x1a1b1f, roughness: 0.5, clearcoat: 0.2 });
    for (const [side, w, h] of [
      [-1, 0.034, 0.048],
      [1, 0.03, 0.042],
    ] as const) {
      const f = this.frameAt(side * (L / 2 - w / 2 + 0.004), 0, 0);
      const box = new THREE.Mesh(new RoundedBoxGeometry(w, h, 0.0105, 4, 0.0035), buckle);
      f.add(box);
      this.staticGroup.add(f);
    }

    // A soft contact shadow under the strap and a darker one under the board.
    const floorY = -0.034;
    const pos: number[] = [];
    const uv: number[] = [];
    const idx: number[] = [];
    const N = 120;
    for (let i = 0; i <= N; i++) {
      const s = -L / 2 + (L * i) / N;
      const n = normal(s);
      const c = point(s, 0, floorY);
      for (const k of [-1, 1]) {
        pos.push(c.x + n.x * k * 0.028, floorY, c.z + n.z * k * 0.028);
        uv.push(i / N, (k + 1) / 2);
      }
      if (i < N) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    const shadow = new THREE.Mesh(
      g,
      new THREE.MeshBasicMaterial({ map: shadowTex(), transparent: true, opacity: 0.22, depthWrite: false }),
    );
    const under = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: this.glow, color: 0x002864, transparent: true, opacity: 0.2, depthWrite: false }),
    );
    under.rotation.x = -Math.PI / 2;
    under.scale.set(0.12, 0.05, 1);
    under.position.set(0, floorY + 0.0005, 0.012);
    this.staticGroup.add(shadow, under);
  }

  /** The ESP32 DevKit seated in its breakout shield, on hook-and-loop in the middle of the strap. */
  private buildController() {
    const root = this.frameAt(0, STRAP_T / 2, 0);
    this.staticGroup.add(root);
    const pad = new THREE.Mesh(
      new RoundedBoxGeometry(0.05, 0.03, PAD, 2, 0.0006),
      new THREE.MeshStandardMaterial({ color: 0x2a2c31, roughness: 0.95 }),
    );
    pad.position.z = PAD / 2;
    root.add(pad);

    // The shield: three rows of S/V/G ports along each long edge.
    const SW = 66;
    const SH = 54;
    const shieldTex = silk(SW, SH, "#16306a", (c) => {
      label(c, "ESP32 SHIELD", 0, -3, 2.4, true);
      label(c, "5V  ●  3V3", 0, 1.4, 1.6);
      for (const [y, rows] of [
        [-24.5, ["G", "V", "S"]],
        [24.5, ["S", "V", "G"]],
      ] as const) {
        rows.forEach((r, k) => label(c, r, -31.2, y + (y < 0 ? 1 : -1) * k * 2.54 - (y < 0 ? 2.54 : -2.54), 1.5));
      }
      for (const [x, pin] of [
        [-20, "D4"],
        [-6, "D13"],
        [8, "D14"],
        [22, "D26"],
      ] as const) {
        label(c, pin, x, 12.5, 1.7, true);
      }
      c.strokeRect(-30, -10, 60, 20);
    });
    const shield = board(SW / 1000, SH / 1000, maskMat(shieldTex), [
      [-0.03, -0.024],
      [0.03, -0.024],
      [-0.03, 0.024],
      [0.03, 0.024],
    ]);
    shield.position.z = PAD;
    root.add(shield);
    this.shield = shield;
    for (const side of [-1, 1]) {
      for (let k = 0; k < 3; k++) {
        const y = side * (0.0245 - 0.00254 * (k + 1) + 0.00127);
        this.put(shield, "1x16_2.54", 0.004, y, PCB - 0.003);
      }
    }

    // The DevKit on two rows of female headers, 8.5 mm up.
    const DW = 54.5;
    const DH = 28;
    const devTex = silk(DW, DH, "#121419", (c) => {
      label(c, "ESP32-DevKitC", -6, -8.6, 2.0, true);
      label(c, "EN", -21, -9.6, 1.4);
      label(c, "BOOT", -21, 9.8, 1.4);
      const left = ["3V3", "EN", "VP", "VN", "34", "35", "32", "33", "25", "26", "27", "14", "12", "13"];
      const right = ["GND", "23", "22", "TX", "RX", "21", "19", "18", "5", "17", "16", "4", "0", "2"];
      left.forEach((s, i) => label(c, s, 24 - i * 2.54 * 1.36, -11.6, 0.95));
      right.forEach((s, i) => label(c, s, 24 - i * 2.54 * 1.36, 11.6, 0.95));
    });
    const dev = board(DW / 1000, DH / 1000, maskMat(devTex));
    dev.position.set(0.004, 0, PCB + 0.0085);
    shield.add(dev);
    for (const side of [-1, 1]) {
      const sock = new THREE.Mesh(new THREE.BoxGeometry(0.048, 0.0025, 0.0085), PLASTIC);
      sock.position.set(0.004, side * 0.0127, PCB + 0.00425);
      shield.add(sock);
    }
    this.put(dev, "esp32_wroom_32e", DW / 2000 - 0.0128 + 0.003, 0, PCB, 90);
    this.put(dev, "usb_c_hro_31m12", -DW / 2000 + 0.0035, 0, PCB, -90);
    this.put(dev, "ch340c", -0.008, 0.004, PCB, 90);
    this.put(dev, "ams1117_3v3", -0.0125, -0.0055, PCB, 0);
    this.put(dev, "reset_skhllca010", -0.021, -0.0085, PCB, 0);
    this.put(dev, "reset_skhllca010", -0.021, 0.0085, PCB, 0);
    for (const [x, y] of [
      [-0.0035, -0.006],
      [-0.0035, -0.0035],
      [-0.016, 0.003],
    ]) {
      this.put(dev, "c_0603", x, y, PCB, 90);
    }
    this.put(dev, "led_0603", -0.017, -0.0015, PCB, 0);
    const power = this.sprite(0xff3b30, 0.006, 0.9);
    power.position.set(-0.017, -0.0015, PCB + 0.001);
    dev.add(power);
  }

  /** A soft glow. `through`: drawn over the strap, so a motor facing away still shows. */
  private sprite(color: number, size: number, opacity: number, through = false) {
    const s = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this.glow,
        color,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: !through,
      }),
    );
    s.scale.setScalar(size);
    return s;
  }

  private tag(text: string) {
    const tex = canvasTex(128, 48, (c) => {
      c.font = '700 30px "Atkinson Hyperlegible Mono", ui-monospace, "SF Mono", Menlo, monospace';
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillStyle = "rgba(255,255,255,0.92)";
      c.fillText(text, 64, 26);
    });
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    s.scale.set(0.024, 0.009, 1);
    return s;
  }

  /** One vibration module: board, coin motor and its leads, driver parts, header and plug. */
  private module(name: string) {
    const g = new THREE.Group();
    const tex = silk(MOD_W * 1000, MOD_H * 1000, "#1d4fb0", (c) => {
      c.beginPath();
      c.arc(3, 0, 5.6, 0, Math.PI * 2);
      c.stroke();
      label(c, "IN", -6.6, -2.54, 1.3, true);
      label(c, "VCC", -6.9, 0, 1.15, true);
      label(c, "GND", -6.9, 2.54, 1.15, true);
      label(c, name, 8.3, -7.1, 1.25, true);
      label(c, "VIB", 9.6, 6.9, 1.1);
    });
    g.add(board(MOD_W, MOD_H, maskMat(tex), [[0.0095, 0.0055]]));
    // The 10 mm coin motor: a pressed steel can on its foam pad.
    const prof = [
      new THREE.Vector2(0, 0),
      new THREE.Vector2(0.0049, 0),
      new THREE.Vector2(0.005, 0.0003),
      new THREE.Vector2(0.005, 0.0024),
      new THREE.Vector2(0.0047, 0.0027),
      new THREE.Vector2(0, 0.0027),
    ];
    const can = STEEL.clone();
    const motor = new THREE.Mesh(new THREE.LatheGeometry(prof, 48), can);
    motor.rotation.x = Math.PI / 2;
    motor.position.z = 0.0004;
    const wob = new THREE.Group(); // tilts round the pad as the eccentric weight spins
    wob.position.set(0.003, 0, PCB);
    wob.add(motor);
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.0048, 0.0048, 0.0004, 40), PLASTIC);
    pad.rotation.x = Math.PI / 2;
    pad.position.set(0.003, 0, PCB + 0.0002);
    g.add(pad, wob);
    for (const [y, color] of [
      [0.0011, 0xd22a2a],
      [-0.0011, 0x2a6ad2],
    ] as const) {
      const lead = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-0.0019, y, PCB + 0.0012),
        new THREE.Vector3(-0.0035, y * 1.6, PCB + 0.001),
        new THREE.Vector3(-0.0045, y * 2.6, PCB + 0.0002),
      ]);
      g.add(
        new THREE.Mesh(
          new THREE.TubeGeometry(lead, 12, 0.00028, 6),
          new THREE.MeshStandardMaterial({ color, roughness: 0.5 }),
        ),
      );
    }
    this.put(g, "2n7002_sot23", -0.0045, -0.0062, PCB, 0);
    this.put(g, "1n4148w_sod123", -0.0045, 0.0062, PCB, 0);
    this.put(g, "1206", 0.0045, -0.0068, PCB, 0);
    this.put(g, "c_0603", 0.009, 0.0002, PCB, 90);
    this.put(g, "led_0603", 0.0095, -0.0035, PCB, 0);
    this.put(g, "1x03_2.54", -0.0095, 0, PCB - 0.003, 0);
    // The female Dupont plug on the header, its three wires leaving the far end.
    const plug = new THREE.Mesh(new THREE.BoxGeometry(0.0026, 0.0078, 0.012), PLASTIC);
    plug.position.set(-0.0095, 0, PCB + 0.0027 + 0.006);
    g.add(plug);
    return { g, can, wob };
  }

  private buildMotors(n: Layout) {
    for (const o of [...this.layoutGroup.children]) {
      this.layoutGroup.remove(o);
      o.traverse((m) => {
        if (m instanceof THREE.Mesh) m.geometry.dispose();
      });
    }
    this.motors = [];
    this.scene.updateMatrixWorld(true);
    const ang = angles(n);
    const now = performance.now();
    ang.forEach((deg, i) => {
      const s = sOf(deg);
      const root = this.frameAt(s, STRAP_T / 2, MOD_Y);
      const name = n === 4 ? PINS4[i] : `M${i + 1}`;
      const { g, can, wob } = this.module(name);
      root.add(g);
      const led = this.sprite(0xff3b30, 0.005, 0);
      led.position.set(0.0095, -0.0035, PCB + 0.001);
      g.add(led);
      if (n !== 16) {
        const tag = this.tag(n === 4 ? PINS4[i] : String(i + 1));
        tag.position.set(0, (STRAP_H / 2 - MOD_Y) / MOD_SCALE + 0.0075, 0.004);
        root.add(tag);
      }
      this.layoutGroup.add(root);
      this.motors.push({ s, root, body: g, wob, can, led, level: 0, born: now + i * 45, lin: new THREE.Color() });
      const rank = ang.filter((a) => Math.sign(a) === Math.sign(deg) && Math.abs(a) < Math.abs(deg)).length;
      this.cable(s, rank, Math.sign(deg));
    });
  }

  /** Three Dupont wires from a module's plug, down to the strap's lower edge, along it to the
   *  middle and up into the shield's pin header: one pin per module, nearest modules innermost. */
  private cable(s: number, rank: number, side: number) {
    const f = this.frameAt(s, STRAP_T / 2, MOD_Y);
    f.scale.setScalar(MOD_SCALE);
    f.updateMatrixWorld();
    const plugTop = f.localToWorld(new THREE.Vector3(-0.0095, 0, PCB + 0.0145));
    const out0 = STRAP_T / 2 + 0.0011 + 0.0012 * rank;
    const end = side * 0.042;
    const pin = 8 + side * (rank + 0.5) - 0.5; // 0..15 along the 1x16 header
    const colors = [0x6b3a1e, 0xc0262d, 0xe8761b];
    colors.forEach((color, k) => {
      const dy = (k - 1) * 0.00115;
      const pts = [plugTop.clone().add(new THREE.Vector3(0, dy, 0))];
      pts.push(point(s - 0.012, out0 + 0.006, MOD_Y - 0.011 + dy));
      const y = -STRAP_H / 2 + 0.0045 + dy;
      const steps = Math.max(2, Math.ceil(Math.abs(end - s) / 0.03));
      for (let i = 1; i <= steps; i++) pts.push(point(s + ((end - s) * i) / steps, out0, y));
      const rowY = -(0.0245 - 0.00254 * (k + 1) + 0.00127);
      const x = 0.004 + 0.00254 * (pin - 7.5);
      pts.push(this.shield.localToWorld(new THREE.Vector3(x, rowY - 0.006, PCB + 0.006)));
      pts.push(this.shield.localToWorld(new THREE.Vector3(x, rowY, PCB + 0.0088)));
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      const mesh = new THREE.Mesh(
        new THREE.TubeGeometry(curve, Math.max(32, steps * 8), 0.00055, 6),
        new THREE.MeshStandardMaterial({ color, roughness: 0.45 }),
      );
      this.layoutGroup.add(mesh);
    });
  }

  private setGuide(bearing: number | null) {
    if (bearing === null) {
      if (this.guideMesh) this.guideMesh.visible = false;
      this.guideAt = null;
      return;
    }
    if (this.guideAt !== null && Math.abs(bearing - this.guideAt) < 1 && this.guideMesh) {
      this.guideMesh.visible = true;
      return;
    }
    this.guideAt = bearing;
    const b = ((bearing + 540) % 360) - 180;
    const half = 10 + 40 * Math.min(1, Math.abs(b) / 150);
    const s0 = Math.max(-L / 2 + 0.03, sOf(b - half));
    const s1 = Math.min(L / 2 - 0.03, sOf(b + half));
    const g = strip(s0, s1, 0.004, STRAP_T + 0.0016, 0, STRAP_H / 2 - 0.0025, 48, 1).geo;
    if (!this.guideMesh) {
      this.guideMesh = new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({
          color: 0x0b3d2c,
          emissive: new THREE.Color(`rgb(${GUIDE.join(",")})`),
          emissiveIntensity: 1.6,
        }),
      );
      this.scene.add(this.guideMesh);
    } else {
      this.guideMesh.geometry.dispose();
      this.guideMesh.geometry = g;
    }
    this.guideMesh.visible = true;
  }

  // ---------------------------------------------------------------------------- frames
  /** Frame the motors in use: the front for 4, the whole strap, buckle to buckle, for 8 and 16. */
  private resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const reach = this.layout === 4 ? sOf(75) + 0.034 : L / 2 + 0.006;
    const f = reach / RC;
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const tanH = tanV * this.camera.aspect;
    // The ends wrap away from the view; frame the full width with a little air either side.
    this.distTarget = Math.max((1.08 * RC * Math.sin(f)) / tanH, 0.05 / tanV);
    if (!this.dist) this.dist = this.distTarget;
  }

  /** Ripples in the webbing round each running motor: rings moving out from it, faster and
   *  deeper the harder it runs, tinted by its heat colour. */
  private ripple(sec: number) {
    const st = this.strap;
    if (!st) return;
    const live = this.motors.filter((m) => m.level > 0.01);
    if (!live.length && !this.rippled) return;
    this.rippled = live.length > 0;
    const pos = st.geo.attributes.position.array as Float32Array;
    const col = st.geo.attributes.color.array as Float32Array;
    pos.set(st.base);
    col.fill(0);
    const R = 0.075;
    const ds = (st.s1 - st.s0) / (st.rings - 1);
    for (const m of live) {
      const k = m.level;
      const amp = 0.0011 * k;
      const w = Math.PI * 2 * (1.4 + 2.2 * k);
      const kappa = (Math.PI * 2) / 0.018;
      const i0 = Math.max(0, Math.floor((m.s - R - st.s0) / ds));
      const i1 = Math.min(st.rings - 1, Math.ceil((m.s - st.s0 + R) / ds));
      for (let v = i0 * st.per; v < (i1 + 1) * st.per; v++) {
        const dx = st.s[v] - m.s;
        const dy = (st.y[v] - MOD_Y) * 1.3;
        const d2 = dx * dx + dy * dy;
        if (d2 > R * R) continue;
        const d = Math.sqrt(d2);
        const wave = Math.sin(sec * w - d * kappa);
        const env = Math.exp(-d2 / (0.034 * 0.034)) * (1 - d / R);
        const disp = amp * env * wave;
        pos[v * 3] += st.nx[v] * disp;
        pos[v * 3 + 2] += st.nz[v] * disp;
        const c = k * env * (0.18 + 0.82 * Math.max(0, wave)) * 0.5;
        col[v * 3] += m.lin.r * c;
        col[v * 3 + 1] += m.lin.g * c;
        col[v * 3 + 2] += m.lin.b * c;
      }
    }
    st.geo.attributes.position.needsUpdate = true;
    st.geo.attributes.color.needsUpdate = true;
    st.geo.computeVertexNormals();
  }

  private frame(ms: number) {
    if (!this.visible || document.hidden) return;
    this.dist += (this.distTarget - this.dist) * 0.08;
    const sway = Math.sin(ms / 5200) * 0.035;
    this.pitch += ((this.layout === 4 ? 0.3 : 0.5) - this.pitch) * 0.08;
    const pitch = this.pitch;
    const look = new THREE.Vector3(0, -0.004, 0.01);
    this.camera.position.set(
      look.x + Math.sin(sway) * Math.cos(pitch) * this.dist,
      look.y + Math.sin(pitch) * this.dist,
      look.z + Math.cos(sway) * Math.cos(pitch) * this.dist,
    );
    this.camera.lookAt(look);

    const drive = spread(this.cells, this.layout);
    const sec = ms / 1000;
    const col = new THREE.Color();
    this.motors.forEach((m, i) => {
      const target = drive[i]?.level ?? 0;
      m.level += (target - m.level) * (target > m.level ? 0.6 : 0.25);
      const k = m.level;
      const [r, g, b] = heat(k);
      col.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      m.lin.copy(col);
      // Pop in after a layout change, one after another along the belt.
      const age = Math.min(1, Math.max(0, (performance.now() - m.born) / 320));
      const ease = 1 - (1 - age) ** 3;
      m.root.scale.setScalar(MOD_SCALE * (0.001 + 0.999 * ease));
      // A running motor shakes its board and wobbles its can round the eccentric weight.
      const amp = 0.00035 * k;
      m.body.position.set(amp * Math.sin(sec * 97 + i), amp * Math.cos(sec * 83 + i * 2), 0);
      const tilt = 0.11 * k;
      m.wob.rotation.set(tilt * Math.sin(sec * 38 + i), tilt * Math.cos(sec * 38 + i), 0);
      m.can.emissive.copy(col).multiplyScalar(0.22 * k);
      m.led.material.color.copy(col);
      m.led.material.opacity = k > 0.02 ? 1 : 0;
    });
    this.ripple(sec);
    if (this.guideMesh?.visible) {
      (this.guideMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 1.2 + 0.6 * Math.sin(sec * 5);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
