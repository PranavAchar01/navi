// The haptic belt in 3D, for the HUD and the judge screen. The strap is the 38 mm webbing of the
// fanny pack, each vibration module is a DiGiYes-style board (coin motor, MOSFET, flyback diode,
// power LED, a 3-pin header with its Dupont plug), and the controller is the ESP32 DevKit on its
// breakout shield in the pouch at the back. The chips, headers and USB-C port are KiCad library
// models (web/public/belt, see SOURCES.md there); boards, strap and wiring are drawn here.
//
// The view sits behind and above the wearer, so right on screen is right on the body. Motors
// light and shake as hard as the server says they run. 4, 8 or 16 motors: 4 is the belt as
// built, 8 the protocol (the HUD default), 16 the highest-fidelity layout.

import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { type Cell, LAYOUTS, type Layout, PINS4, angles, spread } from "./beltlayout";

const A = 0.165; // waist, half width (m)
const B = 0.125; // waist, half depth
const STRAP_H = 0.038;
const STRAP_T = 0.003;
const MOD_W = 0.024; // module board, along the strap
const MOD_H = 0.017;
const PCB = 0.0016;
const MOD_Y = 0.004;
const MOD_SCALE = 1.6; // modules drawn larger than life so the parts read at HUD size // modules ride a little above the strap's middle; cables run under them
const POUCH = { w: 0.112, h: 0.074, d: 0.034 };
const GUIDE: [number, number, number] = [90, 176, 255];
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

// ------------------------------------------------------------------------------ the waist
const point = (t: number, out: number, y: number) => {
  const n = normal(t);
  return new THREE.Vector3(A * Math.sin(t) + n.x * out, y, -B * Math.cos(t) + n.z * out);
};
const normal = (t: number) => new THREE.Vector3(Math.sin(t) / A, 0, -Math.cos(t) / B).normalize();
/** Along the strap, pointing so the module's +x is the viewer's right when they face it. */
const along = (t: number) => new THREE.Vector3(-A * Math.cos(t), 0, -B * Math.sin(t)).normalize();

/** A band around the waist from t0 to t1 with a rounded-rectangle section. */
function band(t0: number, t1: number, h: number, th: number, out: number, y: number, segs: number) {
  const r = Math.min(th / 2, 0.001);
  const prof: [number, number][] = [];
  const corners: [number, number, number][] = [
    [th / 2 - r, h / 2 - r, 0],
    [-th / 2 + r, h / 2 - r, 90],
    [-th / 2 + r, -h / 2 + r, 180],
    [th / 2 - r, -h / 2 + r, 270],
  ];
  for (const [cx, cy, a0] of corners) {
    for (let k = 0; k <= 3; k++) {
      const a = ((a0 + k * 30) * Math.PI) / 180;
      prof.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
  }
  const rings = segs + 1; // a closed band repeats its first ring, so the weave does not jump
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const P = prof.length;
  for (let i = 0; i < rings; i++) {
    const t = t0 + ((t1 - t0) * i) / segs;
    const n = normal(t);
    const c = point(t, out, y);
    prof.forEach(([pn, py], j) => {
      pos.push(c.x + n.x * pn, c.y + py, c.z + n.z * pn);
      uv.push((i / segs) * 48, j / P);
    });
  }
  for (let i = 0; i < segs; i++) {
    const a = i * P;
    const b = (i + 1) * P;
    for (let j = 0; j < P; j++) {
      const j2 = (j + 1) % P;
      idx.push(a + j, b + j, a + j2, a + j2, b + j, b + j2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// ------------------------------------------------------------------------------ textures
function canvasTex(w: number, h: number, draw: (c: CanvasRenderingContext2D) => void, srgb = true) {
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  draw(cv.getContext("2d")!);
  const t = new THREE.CanvasTexture(cv);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/** Twill webbing: a colour map with the edge stitching and a normal map from the weave. */
function webbing() {
  const W = 256;
  const H = 64;
  const height = (x: number, y: number) => {
    const d = ((x + y * 2) % 8) / 8;
    return 0.5 + 0.5 * Math.sin(d * Math.PI * 2) * (0.75 + 0.25 * Math.sin((y / H) * Math.PI * 16));
  };
  const nrm = canvasTex(
    W,
    H,
    (c) => {
      const img = c.createImageData(W, H);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const dx = height(x + 1, y) - height(x - 1, y);
          const dy = height(x, y + 1) - height(x, y - 1);
          const v = new THREE.Vector3(-dx * 1.6, -dy * 1.6, 1).normalize();
          const o = (y * W + x) * 4;
          img.data[o] = (v.x * 0.5 + 0.5) * 255;
          img.data[o + 1] = (v.y * 0.5 + 0.5) * 255;
          img.data[o + 2] = (v.z * 0.5 + 0.5) * 255;
          img.data[o + 3] = 255;
        }
      }
      c.putImageData(img, 0, 0);
    },
    false,
  );
  const col = canvasTex(W, H, (c) => {
    c.fillStyle = "#16181d";
    c.fillRect(0, 0, W, H);
    for (let x = 0; x < W; x += 2) {
      c.fillStyle = `rgba(255,255,255,${0.02 + 0.02 * ((x / 2) % 3)})`;
      c.fillRect(x, 0, 1, H);
    }
    // Lock stitching along both edges of the section (v = 0..1 runs round the profile).
    c.fillStyle = "rgba(190,196,210,0.55)";
    for (let x = 0; x < W; x += 16) {
      c.fillRect(x, H * 0.06, 10, 2);
      c.fillRect(x, H * 0.44, 10, 2);
    }
  });
  for (const t of [nrm, col]) {
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.RepeatWrapping;
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
  t: number;
  root: THREE.Group;
  body: THREE.Group;
  can: THREE.MeshStandardMaterial;
  halo: THREE.Sprite;
  led: THREE.Sprite;
  rings: THREE.Mesh[];
  level: number;
  born: number;
}

export const CAPTION_OF = CAPTION;

export class Belt3D {
  layout: Layout = 8;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.01, 10);
  private parts = new Map<Part, THREE.Object3D>();
  private ready: Promise<void>;
  private motors: Motor[] = [];
  private layoutGroup = new THREE.Group();
  private guideMesh: THREE.Mesh | null = null;
  private guideAt: number | null = null;
  private glow = glowTex();
  private fabric = webbing();
  private cells: Cell[] = Array.from({ length: 8 }, () => ({ rgb: null, level: 0 }));
  private caption: HTMLElement;
  private tourTimers: number[] = [];
  private visible = true;
  private dist = 0.9;
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
    this.scene.environmentIntensity = 0.55;
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(0.4, 1, 0.6);
    this.scene.add(sun, new THREE.HemisphereLight(0xffffff, 0x404858, 0.7), this.layoutGroup);
    this.buildStatic();
    this.ready = this.load().then(() => {
      this.buildController();
      this.setLayout(this.layout);
    });
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
    void this.ready.then(() => this.buildMotors(n));
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

  /** A frame on the strap at t: x along the strap, y up, z out of the body. */
  private frameAt(t: number, out: number, y: number) {
    const g = new THREE.Group();
    const n = normal(t);
    g.matrix.makeBasis(along(t), new THREE.Vector3(0, 1, 0), n);
    g.matrix.setPosition(point(t, out, y));
    g.matrix.decompose(g.position, g.quaternion, g.scale);
    return g;
  }

  private buildStatic() {
    const { col, nrm } = this.fabric;
    const strap = new THREE.Mesh(
      band(0, Math.PI * 2, STRAP_H, STRAP_T, 0, 0, 360),
      new THREE.MeshPhysicalMaterial({
        map: col,
        normalMap: nrm,
        normalScale: new THREE.Vector2(0.7, 0.7),
        roughness: 0.86,
        sheen: 0.6,
        sheenColor: new THREE.Color(0x4a5060),
        sheenRoughness: 0.6,
        side: THREE.DoubleSide,
      }),
    );
    this.scene.add(strap);

    // A soft contact shadow and an arrow on the floor pointing the way the wearer faces.
    const floorY = -STRAP_H / 2 - POUCH.h - 0.012;
    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: this.glow, color: 0x000000, transparent: true, opacity: 0.45, depthWrite: false }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.scale.set(A * 3.0, B * 3.2, 1);
    shadow.position.y = floorY;
    const arrow = new THREE.Shape();
    arrow.moveTo(0, 0.026);
    arrow.lineTo(0.022, -0.004);
    arrow.lineTo(0.009, -0.004);
    arrow.lineTo(0.009, -0.02);
    arrow.lineTo(-0.009, -0.02);
    arrow.lineTo(-0.009, -0.004);
    arrow.lineTo(-0.022, -0.004);
    arrow.closePath();
    const front = new THREE.Mesh(
      new THREE.ShapeGeometry(arrow),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4, depthWrite: false }),
    );
    front.rotation.x = -Math.PI / 2;
    front.position.set(0, floorY + 0.001, -B - 0.075);
    this.scene.add(shadow, front);
  }

  /** The pouch at the back and, on it, the ESP32 DevKit seated in its breakout shield. */
  private buildController() {
    const back = this.frameAt(Math.PI, STRAP_T / 2, -STRAP_H / 2 - POUCH.h / 2 + 0.006);
    this.scene.add(back);
    const pouch = new THREE.Mesh(
      new RoundedBoxGeometry(POUCH.w, POUCH.h, POUCH.d, 5, 0.009),
      new THREE.MeshPhysicalMaterial({
        color: 0x1d2027,
        roughness: 0.92,
        sheen: 0.5,
        sheenColor: new THREE.Color(0x4a5060),
        normalMap: this.fabric.nrm,
        normalScale: new THREE.Vector2(0.5, 0.5),
      }),
    );
    pouch.position.z = POUCH.d / 2;
    back.add(pouch);

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
    shield.position.z = POUCH.d + 0.0005;
    back.add(shield);
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
    motor.position.set(0.003, 0, PCB + 0.0004);
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.0048, 0.0048, 0.0004, 40), PLASTIC);
    pad.rotation.x = Math.PI / 2;
    pad.position.set(0.003, 0, PCB + 0.0002);
    g.add(pad, motor);
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
    return { g, can, motor };
  }

  private buildMotors(n: Layout) {
    for (const o of [...this.layoutGroup.children]) {
      this.layoutGroup.remove(o);
      o.traverse((m) => {
        if (m instanceof THREE.Mesh) m.geometry.dispose();
      });
    }
    this.motors = [];
    const ang = angles(n);
    const now = performance.now();
    ang.forEach((deg, i) => {
      const t = (deg * Math.PI) / 180;
      const root = this.frameAt(t, STRAP_T / 2, MOD_Y);
      const name = n === 4 ? PINS4[i] : `M${i + 1}`;
      const { g, can } = this.module(name);
      root.add(g);
      const halo = this.sprite(0xffffff, 0.05, 0, true);
      halo.position.set(0.003, 0, 0.008);
      const led = this.sprite(0xff3b30, 0.005, 0);
      led.position.set(0.0095, -0.0035, PCB + 0.001);
      const rings = [0, 1].map(() => {
        const r = new THREE.Mesh(
          new THREE.RingGeometry(0.0062, 0.0072, 48),
          new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
        );
        r.position.set(0.003, 0, PCB + 0.0032);
        g.add(r);
        return r;
      });
      g.add(halo, led);
      if (n !== 16) {
        const tag = this.tag(n === 4 ? PINS4[i] : String(i + 1));
        tag.position.set(0, STRAP_H / 2 - MOD_Y + 0.009, 0.004);
        root.add(tag);
      }
      this.layoutGroup.add(root);
      this.motors.push({ t, root, body: g, can, halo, led, rings, level: 0, born: now + i * 45 });
      this.cable(t, i, n);
    });
  }

  /** Three Dupont wires from a module's plug along the bottom of the strap to the pouch. */
  private cable(t: number, i: number, n: Layout) {
    const dir = t <= Math.PI ? 1 : -1;
    const end = Math.PI - dir * (0.05 + 0.012 * (i % (n / 2)));
    const lane = Math.abs(end - t) / Math.PI; // farther modules ride further out, on top
    const out0 = STRAP_T / 2 + 0.0012 + 0.0016 * Math.round(lane * (n / 2));
    const f = this.frameAt(t, STRAP_T / 2, MOD_Y);
    f.scale.setScalar(MOD_SCALE);
    f.updateMatrixWorld();
    const plugTop = f.localToWorld(new THREE.Vector3(-0.0095, 0, PCB + 0.0145));
    const colors = [0x6b3a1e, 0xc0262d, 0xe8761b];
    colors.forEach((color, k) => {
      const dy = (k - 1) * 0.00115;
      const pts = [plugTop.clone().add(new THREE.Vector3(0, dy, 0))];
      pts.push(point(t - dir * 0.012, out0 + 0.006, MOD_Y - 0.01 + dy));
      const y = -STRAP_H / 2 + 0.0055 + dy;
      const steps = Math.max(2, Math.ceil(Math.abs(end - t) / 0.06));
      for (let s = 1; s <= steps; s++) pts.push(point(t + ((end - t) * s) / steps, out0, y));
      pts.push(point(end, out0 + 0.002, -STRAP_H / 2 - 0.006 + dy));
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      const mesh = new THREE.Mesh(
        new THREE.TubeGeometry(curve, Math.max(24, steps * 6), 0.00055, 6),
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
    const t = (bearing * Math.PI) / 180;
    const half = ((10 + 40 * Math.min(1, Math.abs(bearing) / 150)) * Math.PI) / 180;
    const g = band(t - half, t + half, 0.004, STRAP_T + 0.0016, 0, STRAP_H / 2 - 0.0025, 48);
    if (!this.guideMesh) {
      this.guideMesh = new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({
          color: 0x10223a,
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
  private resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Fit a sphere round the belt and pouch, whatever the panel's shape.
    const r = 0.205;
    const half = (this.camera.fov * Math.PI) / 360;
    const fit = Math.min(half, Math.atan(Math.tan(half) * this.camera.aspect));
    this.dist = r / Math.sin(fit);
    this.camera.updateProjectionMatrix();
  }

  private frame(ms: number) {
    if (!this.visible || document.hidden) return;
    const sway = Math.sin(ms / 4200) * 0.07;
    const pitch = 0.62;
    this.camera.position.set(
      Math.sin(sway) * Math.cos(pitch) * this.dist,
      Math.sin(pitch) * this.dist - 0.042,
      Math.cos(sway) * Math.cos(pitch) * this.dist,
    );
    this.camera.lookAt(0, -0.042, 0);

    const drive = spread(this.cells, this.layout);
    const sec = ms / 1000;
    this.motors.forEach((m, i) => {
      const target = drive[i]?.level ?? 0;
      m.level += (target - m.level) * (target > m.level ? 0.6 : 0.25);
      const k = m.level;
      const rgb = drive[i]?.rgb ?? [255, 190, 70];
      const col = new THREE.Color(`rgb(${rgb.join(",")})`);
      // Pop in after a layout change, one after another round the belt.
      const age = Math.min(1, Math.max(0, (performance.now() - m.born) / 320));
      const ease = 1 - (1 - age) ** 3;
      m.root.scale.setScalar(MOD_SCALE * (0.001 + 0.999 * ease));
      // A running motor shakes its board, enough to see from across a room.
      const amp = 0.0007 * k;
      m.body.position.set(amp * Math.sin(sec * 97 + i), amp * Math.cos(sec * 83 + i * 2), amp * 0.6 * Math.sin(sec * 71));
      m.can.emissive.copy(col).multiplyScalar(0.55 * k);
      m.halo.material.color.copy(col);
      m.halo.material.opacity = 0.85 * k;
      m.halo.scale.setScalar(0.026 + 0.03 * k);
      m.led.material.opacity = k > 0.02 ? 1 : 0;
      m.rings.forEach((r, j) => {
        const p = (sec * (0.8 + 1.6 * k) + j * 0.5) % 1;
        r.scale.setScalar(1 + p * 1.8);
        (r.material as THREE.MeshBasicMaterial).color.copy(col);
        (r.material as THREE.MeshBasicMaterial).opacity = k > 0.02 ? (1 - p) * 0.8 * k : 0;
      });
    });
    if (this.guideMesh?.visible) {
      (this.guideMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 1.2 + 0.6 * Math.sin(sec * 5);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
