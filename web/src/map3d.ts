import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PinJson, Pose } from "./types";

const CAP = 2_500_000;
const ACCENT = 0x5ab0ff;

function label(text: string, strong: boolean): THREE.Sprite {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d")!;
  const font = `${strong ? 600 : 500} ${strong ? 30 : 24}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 28;
  c.width = w;
  c.height = 48;
  ctx.font = font;
  ctx.fillStyle = strong ? "#5ab0ff" : "rgba(14,20,44,0.72)";
  ctx.beginPath();
  ctx.roundRect(0, 4, w, 40, 10);
  ctx.fill();
  if (!strong) {
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.stroke();
  }
  ctx.fillStyle = strong ? "#06122e" : "rgba(255,255,255,0.88)";
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
  private splat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexColors: true,
    uniforms: {
      uScale: { value: 600 },
      uSize: { value: 0.075 },
      uCeil: { value: 1e9 },
      uTint: { value: 0 },
      uMin: { value: 1.5 },
      uMax: { value: 26 },
    },
    vertexShader: `
      uniform float uScale; uniform float uSize; uniform float uCeil; uniform float uTint;
      uniform float uMin; uniform float uMax;
      varying vec3 vC;
      void main() {
        vC = color;
        if (position.y > uCeil) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(uSize * uScale / -mv.z, uMin, uMax);
        // First person: far points sink toward the sky colour, 20% at 8 m and beyond.
        vC = mix(vC, vec3(0.16, 0.2, 0.32), uTint * clamp(-mv.z / 8.0, 0.0, 1.0));
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vC;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        float luma = dot(vC, vec3(0.299, 0.587, 0.114));
        vec3 c = mix(vec3(luma), vC, 1.3) * 1.06;
        gl_FragColor = vec4(c, exp(-r2 * 2.4) * 0.92);
      }`,
  });
  private fitted = 0;
  private follow = false;
  // Minimap: straight down, roof cut off, one big label per kind of object.
  private top = false;
  private labelScale = 1;
  private view = new THREE.Mesh(
    new THREE.CircleGeometry(1.1, 24, Math.PI / 2 - 0.62, 1.24),
    new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.28, depthTest: false }),
  );
  private aim: THREE.Line;
  private grid: THREE.GridHelper;
  // First person: the camera sits where the phone is, with its lens, and keeps turning with the
  // live heading between map poses.
  private fp = false;
  private lens: { hfov: number; aspect: number } | null = null;
  private fpPose: { pos: THREE.Vector3; fwd: THREE.Vector3; up: THREE.Vector3; yaw: number } | null = null;
  private liveYaw: number | null = null;
  private turn = 0;
  onCount: (n: number) => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    this.camera.position.set(0, 7, 7);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(this.col, 3, true));
    this.geo.setDrawRange(0, 0);
    // Each map point is drawn as a soft Gaussian disc sized in metres, a cheap splat look that
    // keeps the map streamable and the metric scale intact.
    const pts = new THREE.Points(this.geo, this.splat);
    pts.frustumCulled = false;
    this.scene.add(pts);

    this.grid = new THREE.GridHelper(40, 40, 0xffffff, 0xffffff);
    const gm = this.grid.material as THREE.LineBasicMaterial;
    gm.depthWrite = false;
    gm.transparent = true;
    gm.opacity = 0.07;
    this.scene.add(this.grid);

    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.35, 4),
      new THREE.MeshBasicMaterial({ color: ACCENT }),
    );
    cone.rotation.x = -Math.PI / 2; // tip along -z (forward)
    this.rig.add(cone);
    this.view.rotation.x = -Math.PI / 2;
    this.view.renderOrder = 9;
    this.view.visible = false;
    this.rig.add(this.view);
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
    this.applyLens();
  }

  private applyLens() {
    if (this.fp && this.lens) {
      // The lens is horizontal, measured on the frame the map was solved from.
      this.camera.fov =
        (2 * Math.atan(Math.tan((this.lens.hfov * Math.PI) / 360) / this.lens.aspect) * 180) / Math.PI;
    }
    this.camera.updateProjectionMatrix();
    const px = this.renderer.getPixelRatio();
    this.splat.uniforms.uScale.value =
      (this.canvas.clientHeight * px) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
  }

  /** First-person mode: the map seen through the phone's own camera, tinted with distance. */
  setFirstPerson(on: boolean) {
    this.fp = on;
    this.controls.enabled = !on;
    this.grid.visible = !on;
    this.trail.visible = !on;
    this.rig.visible = false;
    this.splat.uniforms.uTint.value = on ? 0.2 : 0;
    this.splat.uniforms.uMin.value = on ? 2.5 : 1.5;
    this.splat.uniforms.uMax.value = on ? 9 : 26;
    this.splat.uniforms.uSize.value = on ? 0.03 : 0.075;
    this.camera.up.set(0, 1, 0);
    this.applyLens();
  }

  /** The live map-frame heading (radians, atan2(fwd_x, fwd_z)), for turning between poses. */
  setLiveYaw(yaw: number | null) {
    this.liveYaw = yaw;
  }

  get hasPose(): boolean {
    return this.fpPose !== null && this.count > 0;
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
    if (this.fitted < 3 || this.top) {
      this.fitted++;
      this.fit();
    }
  }

  /** Replace the whole map (outside mode trims what is behind you; switching modes clears it). */
  replaceMap(n: number, pts: Float32Array, cols: Uint8Array) {
    this.count = 0;
    this.fitted = 3;
    this.clearHot();
    this.addChunk(n, pts, cols);
    if (n === 0) {
      this.geo.setDrawRange(0, 0);
      this.onCount(0);
    }
  }

  /** Outside mode: the camera trails the wearer instead of framing the whole map. */
  setFollow(on: boolean) {
    this.follow = on;
    if (on && !this.top) this.goTo(this.rig.position.clone().add(new THREE.Vector3(2.5, 4.5, 5.5)), this.rig.position.clone(), 0.8);
  }

  /** Minimap mode: a floor plan seen from straight above, the walk's first heading pointing up. */
  setTopDown(on: boolean) {
    this.top = on;
    this.controls.enabled = !on;
    this.camera.up.set(0, on ? 0 : 1, on ? -1 : 0);
    this.view.visible = on;
    this.rig.scale.setScalar(on ? 2.2 : 1);
    if (!on) this.splat.uniforms.uCeil.value = 1e9;
    this.styleLabels();
    this.fit();
  }

  private styleLabels() {
    const shown = new Set<string>();
    for (const [id, g] of this.pinMeshes) {
      const p = this.pinData.get(id)!;
      const [dot, sp] = g.children as [THREE.Mesh, THREE.Sprite];
      const base = sp.userData.base as [number, number];
      const first = p.kind === "object" && !shown.has(p.label);
      if (first) shown.add(p.label);
      g.visible = !this.top || first;
      const k = this.top ? this.labelScale : 1;
      sp.scale.set(base[0] * k, base[1] * k, 1);
      sp.position.y = (p.kind === "object" ? 0.28 : 0.2) * k;
      dot.scale.setScalar(this.top ? Math.min(k, 3) : 1);
    }
  }

  fit() {
    if (!this.count) return;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    const step = Math.max(1, Math.floor(this.count / 6000));
    const ys: number[] = [];
    for (let i = 0; i < this.count; i += step) {
      v.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      box.expandByPoint(v);
      ys.push(v.y);
    }
    const c = box.getCenter(new THREE.Vector3());
    if (this.top) {
      // Floor = low percentile of height; cut 1.5 m above it so walls read as lines, not a lid.
      ys.sort((a, b) => a - b);
      const floor = ys[Math.floor(ys.length * 0.03)];
      this.splat.uniforms.uCeil.value = floor + 1.5;
      const size = box.getSize(new THREE.Vector3());
      const half = Math.max(size.z / 2, size.x / 2 / this.camera.aspect, 1.5) * 1.08;
      const hgt = half / Math.tan((this.camera.fov * Math.PI) / 360);
      this.labelScale = Math.max(1, (2 * half) / 4.5);
      this.styleLabels();
      c.y = floor;
      this.goTo(new THREE.Vector3(c.x, floor + hgt, c.z), c, 0.6);
      return;
    }
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
    if (this.fp) {
      const fwd = new THREE.Vector3(...p.fwd);
      this.fpPose = {
        pos: new THREE.Vector3(...p.pos),
        fwd,
        up: new THREE.Vector3(...p.up),
        yaw: Math.atan2(fwd.x, fwd.z),
      };
      this.turn = 0;
      if (p.hfov && p.aspect) {
        this.lens = { hfov: p.hfov, aspect: p.aspect };
        this.applyLens();
      }
      return;
    }
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
        sp.userData.base = [sp.scale.x, sp.scale.y];
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
    this.styleLabels();
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
    if (this.top) return; // the whole floor plan is already in view
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
    if (this.follow && !this.tween && this.rig.visible) {
      const d = this.rig.position.clone().sub(this.controls.target).multiplyScalar(0.08);
      this.controls.target.add(d);
      this.camera.position.add(d);
    }
    if (this.fp) {
      const q = this.fpPose;
      if (q) {
        if (this.liveYaw !== null) {
          const d = Math.atan2(Math.sin(this.liveYaw - q.yaw), Math.cos(this.liveYaw - q.yaw));
          this.turn += (d - this.turn) * 0.5;
        }
        const fwd = q.fwd.clone().applyAxisAngle(q.up, this.turn);
        this.camera.up.copy(q.up);
        this.camera.position.copy(q.pos);
        this.camera.lookAt(q.pos.clone().add(fwd));
      }
    } else if (this.top) this.camera.lookAt(this.controls.target);
    else this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
