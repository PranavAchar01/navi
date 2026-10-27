import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PinJson, Pose } from "./types";

const CAP = 2_500_000;
const ACCENT = 0x0a6cff;

function label(text: string, strong: boolean): THREE.Sprite {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d")!;
  const font = `${strong ? 600 : 500} ${strong ? 30 : 24}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 28;
  c.width = w;
  c.height = 48;
  ctx.font = font;
  ctx.fillStyle = strong ? "#0a6cff" : "rgba(255,255,255,0.92)";
  ctx.beginPath();
  ctx.roundRect(0, 4, w, 40, 10);
  ctx.fill();
  if (!strong) {
    ctx.strokeStyle = "#dddddd";
    ctx.stroke();
  }
  ctx.fillStyle = strong ? "#fff" : "#5e5e5e";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 14, 25);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  const h = strong ? 0.13 : 0.09;
  sp.scale.set((w / 48) * h, h, 1);
  sp.renderOrder = 10;
  return sp;
}

export class MapView {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private pos = new Float32Array(CAP * 3);
  private col = new Uint8Array(CAP * 3);
  private geo = new THREE.BufferGeometry();
  private count = 0;
  private trail = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: ACCENT }),
  );
  private rig = new THREE.Group();
  private pinGroup = new THREE.Group();
  private pinMeshes = new Map<number, THREE.Object3D>();
  private pinData = new Map<number, PinJson>();
  private hot: number | null = null;
  private tween: { dur: number; t0: number; from: THREE.Vector3; to: THREE.Vector3; fromT: THREE.Vector3; toT: THREE.Vector3 } | null = null;
  private pulse = new THREE.Mesh(
    new THREE.RingGeometry(0.1, 0.13, 40),
    new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, side: THREE.DoubleSide }),
  );
  private fitted = 0;
  private aim: THREE.Line;
  onCount: (n: number) => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene.background = new THREE.Color(0xffffff);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    this.camera.position.set(0, 7, 7);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(this.col, 3, true));
    this.geo.setDrawRange(0, 0);
    const pts = new THREE.Points(
      this.geo,
      new THREE.PointsMaterial({ size: 0.045, vertexColors: true, sizeAttenuation: true }),
    );
    pts.frustumCulled = false;
    this.scene.add(pts);

    const grid = new THREE.GridHelper(40, 40, 0xdcdcdc, 0xececea);
    (grid.material as THREE.Material).depthWrite = false;
    this.scene.add(grid);

    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.35, 4),
      new THREE.MeshBasicMaterial({ color: ACCENT }),
    );
    cone.rotation.x = -Math.PI / 2; // tip along -z (forward)
    this.rig.add(cone);
    this.rig.visible = false;
    this.scene.add(this.rig);

    this.pulse.rotation.x = -Math.PI / 2;
    this.pulse.visible = false;
    this.scene.add(this.pulse);
    this.aim = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineDashedMaterial({ color: ACCENT, dashSize: 0.12, gapSize: 0.08 }),
    );
    this.aim.visible = false;
    this.scene.add(this.aim);
    this.scene.add(this.pinGroup);

    const ro = new ResizeObserver(() => this.resize());
    ro.observe(canvas);
    this.resize();
    this.animate();
  }

  private resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  addChunk(n: number, pts: Float32Array, cols: Uint8Array) {
    const take = Math.min(n, CAP - this.count);
    if (take <= 0) return;
    this.pos.set(pts.subarray(0, take * 3), this.count * 3);
    this.col.set(cols.subarray(0, take * 3), this.count * 3);
    this.count += take;
    this.geo.setDrawRange(0, this.count);
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
    this.onCount(this.count);
    if (this.fitted < 3) {
      this.fitted++;
      this.fit();
    }
  }

  fit() {
    if (!this.count) return;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    const step = Math.max(1, Math.floor(this.count / 6000));
    for (let i = 0; i < this.count; i += step) {
      v.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      box.expandByPoint(v);
    }
    const c = box.getCenter(new THREE.Vector3());
    const r = Math.max(2.5, box.getSize(new THREE.Vector3()).length() * 0.5);
    this.goTo(new THREE.Vector3(c.x + r * 0.35, c.y + r * 0.95, c.z + r * 0.95), c, 0.6);
  }

  setTrail(points: number[][]) {
    const arr = new Float32Array(points.length * 3);
    points.forEach((p, i) => arr.set(p.slice(0, 3), i * 3));
    this.trail.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    this.trail.geometry.computeBoundingSphere();
  }

  setPose(p: Pose) {
    this.rig.visible = true;
    this.rig.position.set(...p.pos);
    const f = new THREE.Vector3(...p.fwd);
    f.y = 0;
    this.rig.rotation.set(0, Math.atan2(-f.x, -f.z), 0);
  }

  setPins(pins: PinJson[]) {
    const seen = new Set<number>();
    for (const p of pins) {
      seen.add(p.id);
      this.pinData.set(p.id, p);
      let obj = this.pinMeshes.get(p.id);
      if (!obj) {
        const g = new THREE.Group();
        const strong = p.kind === "object";
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(strong ? 0.07 : 0.05, 16, 12),
          new THREE.MeshBasicMaterial({ color: strong ? ACCENT : 0x9a948a }),
        );
        const sp = label(p.label, strong);
        sp.position.set(0, strong ? 0.28 : 0.2, 0);
        g.add(dot, sp);
        this.pinGroup.add(g);
        this.pinMeshes.set(p.id, g);
        obj = g;
      }
      obj.position.set(...p.pos);
    }
    for (const [id, o] of this.pinMeshes) {
      if (!seen.has(id)) {
        this.pinGroup.remove(o);
        this.pinMeshes.delete(id);
        this.pinData.delete(id);
      }
    }
    if (this.hot !== null) this.showPulse(this.hot);
  }

  private showPulse(id: number) {
    const p = this.pinData.get(id);
    if (!p) return;
    this.hot = id;
    this.pulse.position.set(p.pos[0], p.pos[1] + 0.02, p.pos[2]);
    this.pulse.visible = true;
    const from = this.rig.visible ? this.rig.position : null;
    if (from) {
      this.aim.geometry.setFromPoints([from.clone(), new THREE.Vector3(...p.pos)]);
      this.aim.computeLineDistances();
      this.aim.visible = true;
    }
  }

  clearHot() {
    this.hot = null;
    this.pulse.visible = false;
    this.aim.visible = false;
  }

  /** Fly the camera to look at a pin, from the side the user is on. */
  flyTo(pin: PinJson) {
    this.showPulse(pin.id);
    const target = new THREE.Vector3(...pin.pos);
    const dir = this.rig.visible
      ? this.rig.position.clone().sub(target).setY(0)
      : new THREE.Vector3(1, 0, 1);
    if (dir.lengthSq() < 0.01) dir.set(1, 0, 1);
    dir.normalize().multiplyScalar(3.0);
    this.goTo(target.clone().add(dir).add(new THREE.Vector3(0, 1.8, 0)), target, 1.3);
  }

  private goTo(pos: THREE.Vector3, target: THREE.Vector3, seconds: number) {
    this.tween = {
      dur: seconds * 1000,
      t0: performance.now(),
      from: this.camera.position.clone(),
      to: pos,
      fromT: this.controls.target.clone(),
      toT: target,
    };
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    if (this.tween) {
      const k = Math.min(1, (performance.now() - this.tween.t0) / this.tween.dur);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      this.camera.position.lerpVectors(this.tween.from, this.tween.to, e);
      this.controls.target.lerpVectors(this.tween.fromT, this.tween.toT, e);
      if (k >= 1) this.tween = null;
    }
    if (this.pulse.visible) {
      const t = (performance.now() % 1400) / 1400;
      this.pulse.scale.setScalar(1 + t * 3);
      (this.pulse.material as THREE.MeshBasicMaterial).opacity = 1 - t;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
