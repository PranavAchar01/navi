import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PinJson, Pose } from "./types";
import { GUIDE, MEMORY, YOU, answerGlow, bearingDeg, ease, hex, isPerson } from "./visual";

const CAP = 2_500_000;
const QUIET_M = 3;

// One soft round glow, white, tinted per use: pins, the wearer's puck.
let haloTex: THREE.Texture | null = null;
function halo(): THREE.Texture {
  if (haloTex) return haloTex;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.12, "rgba(255,255,255,0.95)");
  g.addColorStop(0.3, "rgba(255,255,255,0.4)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  haloTex = new THREE.CanvasTexture(c);
  haloTex.colorSpace = THREE.SRGBColorSpace;
  return haloTex;
}
function glowSprite(color: number, size: number, opacity: number): THREE.Sprite {
  const sp = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: halo(), color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  sp.scale.setScalar(size);
  return sp;
}

/** The room mesh over the camera is drawn as outlines: the mesh goes into an offscreen buffer
 *  (view normal, depth), and lines are drawn where depth jumps (an edge in front of what is
 *  behind it) or the surface turns (a wall meeting the floor). Its own colours, laid over the
 *  same picture, added nothing a viewer could see; the fused mesh is too bumpy for crease edges
 *  (about nine in ten of its edges bend over 30 degrees). Lines nearer than 0.6 m are dropped and
 *  the far ones fade, amber near to blue far. */
const ROOM_VIEW_STALE_MS = 1000; // first person draws the room only from a room pose this fresh
const OUTLINE_STEP_PX = 3; // neighbour distance of the edge test
const OUTLINE_DEPTH = 0.02; // relative depth jump that starts a line
const OUTLINE_TURN = 0.9; // normal agreement below which a line starts
function roomGbuffer(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    vertexShader: `
      varying vec3 vN; varying float vD;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vD = -mv.z;
        vN = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vN; varying float vD;
      void main() { gl_FragColor = vec4(gl_FrontFacing ? normalize(vN) : -normalize(vN), vD); }`,
  });
}
function outlineMaterial(tex: THREE.Texture): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthTest: false,
    uniforms: { tG: { value: tex }, px: { value: new THREE.Vector2(1, 1) } },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: `
      uniform sampler2D tG; uniform vec2 px; varying vec2 vUv;
      vec4 g(vec2 o) {
        vec4 s = texture2D(tG, vUv + o * px * ${OUTLINE_STEP_PX.toFixed(1)});
        return s.a > 0.0 ? s : vec4(0.0, 0.0, 0.0, 100.0); // nothing drawn: far away
      }
      void main() {
        vec4 c = g(vec2(0.0));
        float d = c.a, dd = 0.0, nd = 1.0;
        vec2 offs[8] = vec2[8](vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1),
                               vec2(1, 1), vec2(-1, -1), vec2(1, -1), vec2(-1, 1));
        for (int i = 0; i < 8; i++) {
          vec4 s = g(offs[i]);
          d = min(d, s.a);
          dd = max(dd, abs(s.a - c.a));
          if (s.a < 99.0 && c.a < 99.0) nd = min(nd, dot(s.xyz, c.xyz));
        }
        if (d > 99.0) discard;
        float jump = smoothstep(${OUTLINE_DEPTH.toFixed(3)}, ${(2 * OUTLINE_DEPTH).toFixed(3)}, dd / d);
        float turn = smoothstep(${OUTLINE_TURN.toFixed(2)}, ${(OUTLINE_TURN - 0.25).toFixed(2)}, nd);
        float a = max(jump, turn) * smoothstep(0.6, 1.2, d) * (1.0 - smoothstep(5.0, 7.0, d));
        if (a < 0.02) discard;
        gl_FragColor = vec4(mix(vec3(1.0, 0.78, 0.35), vec3(0.45, 0.85, 1.0), smoothstep(1.0, 4.0, d)), a);
      }`,
  });
}

function label(text: string, strong: boolean): THREE.Sprite {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d")!;
  const font = `${strong ? 600 : 500} ${strong ? 30 : 24}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 28;
  c.width = w;
  c.height = 48;
  ctx.font = font;
  ctx.fillStyle = strong ? `rgb(${MEMORY.join(",")})` : "rgba(14,20,44,0.72)";
  ctx.beginPath();
  ctx.roundRect(0, 4, w, 40, 10);
  ctx.fill();
  if (!strong) {
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.stroke();
  }
  ctx.fillStyle = strong ? "#04262a" : "rgba(255,255,255,0.88)";
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
  private size = new Uint8Array(CAP); // disc size per point, 0.5 cm steps; 0 = the default
  private geo = new THREE.BufferGeometry();
  private count = 0;
  // The walked trail: a thin white line that fades with age.
  private trail = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }),
  );
  private rig = new THREE.Group();
  private pinGroup = new THREE.Group();
  private pinMeshes = new Map<number, THREE.Object3D>();
  private pinData = new Map<number, PinJson>();
  private hot: number | null = null;
  // When the selected pin answered a question (performance.now), for its three pulses and hold.
  private hotAt: number | null = null;
  private hover: number | null = null;
  private lastPose: Pose | null = null;
  private floor: number | null = null;
  private target: PinJson | null = null;
  // The way to go: a ribbon on the floor from the wearer to the target, fading toward the wearer.
  private ribbon = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      uniforms: { uColor: { value: new THREE.Color(hex(GUIDE)) } },
      vertexShader: `
        attribute float aT; attribute float aS;
        varying float vT; varying float vS;
        void main() { vT = aT; vS = aS; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vT; varying float vS;
        void main() {
          float a = mix(0.06, 0.8, pow(vT, 1.3)) * smoothstep(1.0, 0.15, abs(vS));
          gl_FragColor = vec4(uColor, a);
        }`,
    }),
  );
  private lastFrame = performance.now();
  /** A pin was clicked in the map. */
  onPick: (pin: PinJson) => void = () => {};
  /** Pins this close to the wearer show their label, in map units (metres when the scale holds). */
  labelRadius = 2;
  private tween: { dur: number; t0: number; from: THREE.Vector3; to: THREE.Vector3; fromT: THREE.Vector3; toT: THREE.Vector3 } | null = null;
  private pulse = new THREE.Mesh(
    new THREE.RingGeometry(0.1, 0.13, 40),
    new THREE.MeshBasicMaterial({ color: hex(MEMORY), transparent: true, side: THREE.DoubleSide, depthWrite: false }),
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
      uSolid: { value: 0 },
      uGrow: { value: 0 },
      uDensity: { value: 0 },
    },
    vertexShader: `
      uniform float uScale; uniform float uSize; uniform float uCeil; uniform float uTint;
      uniform float uMin; uniform float uMax; uniform float uGrow; uniform float uDensity;
      attribute float aSize;
      varying vec3 vC;
      void main() {
        vC = color;
        if (position.y > uCeil) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float w = (uDensity > 0.5 && aSize > 0.5) ? aSize * 0.005 : uSize;
        gl_PointSize = clamp(max(w, uGrow * -mv.z) * uScale / -mv.z, uMin, uMax);
        // First person: far points sink toward the sky colour, 20% at 8 m and beyond.
        vC = mix(vC, vec3(0.16, 0.2, 0.32), uTint * clamp(-mv.z / 8.0, 0.0, 1.0));
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uSolid;
      varying vec3 vC;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        // The real colours, about 30 percent desaturated and slightly darker, so the accent
        // colours (memory, guide, you) always read on top.
        float luma = dot(vC, vec3(0.299, 0.587, 0.114));
        vec3 c = mix(vec3(luma), vC, 0.7) * 0.9;
        // First person: opaque discs on the depth buffer, so the nearest surface wins and
        // overlapping discs close the gaps between points.
        gl_FragColor = vec4(c, mix(exp(-r2 * 2.4) * 0.92, 1.0, uSolid));
      }`,
  });
  private fitted = 0;
  private follow = false;
  private points!: THREE.Points;
  // The dense room mesh (room_watch's), placed in the walk's frame by the home fit.
  private room: THREE.Mesh | null = null;
  private roomBuf = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
  private outline = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), outlineMaterial(this.roomBuf.texture));
  private outlineScene = new THREE.Scene().add(this.outline);
  private flatCam = new THREE.Camera();
  // The room seen from the latest frame placed in it by PnP (room_pose), in the room's own frame.
  private roomCam = Object.assign(new THREE.PerspectiveCamera(), { matrixAutoUpdate: false, matrixWorldAutoUpdate: false });
  private roomViewAt = -Infinity;
  // Minimap: straight down, roof cut off, one big label per kind of object.
  private top = false;
  private labelScale = 1;
  private grid: THREE.GridHelper;
  // First person: the camera sits where the phone is, with its lens, and keeps turning with the
  // live heading between map poses.
  private fp = false;
  private lens: { hfov: number; aspect: number } | null = null;
  private lensFixed = false; // set by setLens: the poses' own lens no longer replaces it
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
    this.geo.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1));
    this.geo.setDrawRange(0, 0);
    // Each map point is drawn as a soft Gaussian disc sized in metres, a cheap splat look that
    // keeps the map streamable and the metric scale intact.
    const pts = new THREE.Points(this.geo, this.splat);
    pts.frustumCulled = false;
    this.scene.add(pts);
    this.points = pts;

    this.grid = new THREE.GridHelper(40, 40, 0xffffff, 0xffffff);
    const gm = this.grid.material as THREE.LineBasicMaterial;
    gm.depthWrite = false;
    gm.transparent = true;
    gm.opacity = 0.07;
    this.scene.add(this.grid);

    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    // You: a soft white puck with a short heading cone, brightest at the puck.
    const puck = glowSprite(hex(YOU), 0.36, 0.95);
    puck.renderOrder = 9;
    this.rig.add(puck);
    const fan = [0, 0, 0];
    const fanA = [0.4];
    for (let i = 0; i <= 12; i++) {
      const a = -0.45 + (0.9 * i) / 12;
      fan.push(0.7 * Math.sin(a), 0, -0.7 * Math.cos(a));
      fanA.push(0);
    }
    const coneGeo = new THREE.BufferGeometry();
    coneGeo.setAttribute("position", new THREE.Float32BufferAttribute(fan, 3));
    coneGeo.setAttribute("color", new THREE.Float32BufferAttribute(fanA.flatMap((a) => [1, 1, 1, a]), 4));
    coneGeo.setIndex(Array.from({ length: 12 }, (_, i) => [0, i + 2, i + 1]).flat());
    const cone = new THREE.Mesh(
      coneGeo,
      new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide }),
    );
    cone.renderOrder = 8;
    this.rig.add(cone);
    this.rig.visible = false;
    this.scene.add(this.rig);

    this.pulse.rotation.x = -Math.PI / 2;
    this.pulse.visible = false;
    this.scene.add(this.pulse);
    this.ribbon.visible = false;
    this.ribbon.frustumCulled = false;
    this.ribbon.renderOrder = 7;
    this.scene.add(this.ribbon);
    this.scene.add(this.pinGroup);
    this.pickEvents();

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
    const r = this.renderer.getPixelRatio();
    this.roomBuf.setSize(w * r, h * r);
    (this.outline.material as THREE.ShaderMaterial).uniforms.px.value.set(1 / (w * r), 1 / (h * r));
    this.camera.aspect = w / h;
    this.applyLens();
  }

  private applyLens() {
    if (this.fp && this.lens) {
      // The lens is horizontal, measured on the frame the map was solved from. The picture under
      // the canvas is cover-fitted, so a canvas wider than the frame shows its full width.
      const fit = Math.max(this.lens.aspect, this.camera.aspect);
      this.camera.fov = (2 * Math.atan(Math.tan((this.lens.hfov * Math.PI) / 360) / fit) * 180) / Math.PI;
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
    this.pinGroup.visible = !on;
    this.rig.visible = false;
    this.splat.uniforms.uTint.value = on ? 0.2 : 0;
    this.splat.uniforms.uMin.value = on ? 3 : 1.5;
    this.splat.uniforms.uMax.value = on ? 20 : 26;
    this.splat.uniforms.uSolid.value = on ? 1 : 0;
    this.splat.uniforms.uDensity.value = on ? 1 : 0;
    this.splat.uniforms.uGrow.value = on ? 0.012 : 0; // a disc at least as wide as a source pixel
    this.splat.transparent = !on;
    this.splat.depthWrite = on;
    this.splat.depthTest = true;
    this.splat.needsUpdate = true;
    this.splat.uniforms.uSize.value = on ? 0.03 : 0.075;
    this.camera.up.set(0, 1, 0);
    this.applyLens();
  }

  /** The room mesh, its vertex colours unlit (they are the camera's own), placed by `m`
   *  (4x4 rows, mesh frame to this map's frame). Replaces any mesh set before. */
  setRoomMesh(geo: THREE.BufferGeometry, m: number[][]) {
    if (this.room?.geometry === geo) {
      this.room.matrix.set(...(m.flat() as Parameters<THREE.Matrix4["set"]>));
      return;
    }
    if (this.room) {
      this.scene.remove(this.room);
      this.room.geometry.dispose();
    }
    if (!geo.hasAttribute("normal")) geo.computeVertexNormals();
    this.room = new THREE.Mesh(geo, roomGbuffer());
    this.room.matrixAutoUpdate = false;
    this.room.matrix.set(...(m.flat() as Parameters<THREE.Matrix4["set"]>));
    this.room.frustumCulled = false;
    this.scene.add(this.room);
  }

  /** A lens measured elsewhere (the room's), kept over the one each pose carries. */
  setLens(hfov: number, aspect: number) {
    this.lens = { hfov, aspect };
    this.lensFixed = true;
    this.applyLens();
  }

  /** Show or hide the live map points (the judge overlay shows the room mesh alone). */
  setPointsVisible(on: boolean) {
    this.points.visible = on;
  }

  /** The live map-frame heading (radians, atan2(fwd_x, fwd_z)), for turning between poses. */
  setLiveYaw(yaw: number | null) {
    this.liveYaw = yaw;
  }

  get hasPose(): boolean {
    return this.fpPose !== null && (this.count > 0 || this.room !== null);
  }

  get hasRoom(): boolean {
    return this.room !== null;
  }

  addChunk(n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) {
    const take = Math.min(n, CAP - this.count);
    if (take <= 0) return;
    this.pos.set(pts.subarray(0, take * 3), this.count * 3);
    this.col.set(cols.subarray(0, take * 3), this.count * 3);
    this.size.set(sizes ? sizes.subarray(0, take) : new Uint8Array(take), this.count);
    this.geo.attributes.aSize.needsUpdate = true;
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
  replaceMap(n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) {
    this.count = 0;
    this.fitted = 3;
    this.clearHot();
    this.addChunk(n, pts, cols, sizes);
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
    this.rig.scale.setScalar(on ? 2.2 : 1);
    if (!on) this.splat.uniforms.uCeil.value = 1e9;
    this.styleLabels();
    this.fit();
  }

  private styleLabels() {
    const shown = new Set<string>();
    for (const [id, g] of this.pinMeshes) {
      const p = this.pinData.get(id)!;
      const [dot, sp] = g.children as [THREE.Sprite, THREE.Sprite];
      const base = sp.userData.base as [number, number];
      const first = p.kind === "object" && !shown.has(p.label);
      if (first) shown.add(p.label);
      g.visible = !this.top || first || id === this.hot;
      const k = this.top ? this.labelScale : 1;
      sp.scale.set(base[0] * k, base[1] * k, 1);
      sp.position.y = (p.kind === "object" ? 0.28 : 0.2) * k;
      dot.userData.k = this.top ? Math.min(k, 3) : 1;
    }
    this.updateRibbon();
  }

  /** Labels show on hover, on selection, or when the pin is within 2 m of the wearer; the answer
   *  pin pulses three times, then holds bright. Eased, so nothing pops. */
  private updatePins(now: number, k: number) {
    const you = this.rig.visible ? this.rig.position : null;
    const glow = this.hotAt === null ? null : answerGlow(now - this.hotAt);
    const shown = this.declutter(you);
    for (const [id, g] of this.pinMeshes) {
      const p = this.pinData.get(id)!;
      const [dot, sp] = g.children as [THREE.Sprite, THREE.Sprite];
      const show = shown.has(id);
      const lm = sp.material as THREE.SpriteMaterial;
      lm.opacity += ((show ? 1 : 0) - lm.opacity) * k;
      sp.visible = lm.opacity > 0.01;
      const bright = id === this.hot ? Math.max(glow?.bright ?? 0, 0.4) : id === this.hover ? 0.4 : 0;
      const dm = dot.material as THREE.SpriteMaterial;
      // Unlabelled pins farther than QUIET_M from the wearer go quiet: dimmer, smaller halo.
      const quiet = !show && bright === 0 && (you === null || you.distanceTo(g.position) > QUIET_M);
      const base = Math.min(p.kind === "object" ? 0.8 : 0.4, quiet ? 0.35 : 1);
      dm.opacity += (base + (1 - base) * bright - dm.opacity) * k;
      const size = (p.kind === "object" ? 0.26 : 0.16) * (quiet ? 0.65 : 1) * (1 + 0.4 * bright) * ((dot.userData.k as number) ?? 1);
      dot.scale.setScalar(dot.scale.x + (size - dot.scale.x) * k);
    }
    if (glow && glow.ring !== null) {
      this.pulse.visible = true;
      this.pulse.scale.setScalar((1 + glow.ring * 3) * (this.top ? Math.min(this.labelScale, 3) : 1));
      (this.pulse.material as THREE.MeshBasicMaterial).opacity = 1 - glow.ring;
    } else this.pulse.visible = false;
  }

  /** Which labels show: the selected and hovered pins always, then pins within `labelRadius`,
   *  nearest first, one per name, and none that would cover a label already placed on screen. */
  private declutter(you: THREE.Vector3 | null): Set<number> {
    const want: { id: number; rank: number; sp: THREE.Sprite; name: string }[] = [];
    for (const [id, g] of this.pinMeshes) {
      if (!g.visible) continue;
      const d = you === null ? Infinity : you.distanceTo(g.position);
      const rank = id === this.hot ? -2 : id === this.hover ? -1 : d < this.labelRadius ? d : null;
      if (rank !== null) want.push({ id, rank, sp: g.children[1] as THREE.Sprite, name: this.pinData.get(id)!.label });
    }
    want.sort((a, b) => a.rank - b.rank);
    const out = new Set<number>();
    const names = new Set<string>();
    const boxes: [number, number, number, number][] = [];
    const { width: W, height: H } = this.renderer.domElement;
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    const c = new THREE.Vector3();
    const e = new THREE.Vector3();
    const s = new THREE.Vector3();
    for (const w of want) {
      if (w.rank >= 0 && names.has(w.name)) continue;
      w.sp.getWorldPosition(c);
      w.sp.getWorldScale(s);
      e.copy(c).addScaledVector(right, s.x / 2).project(this.camera);
      c.project(this.camera);
      if (c.z > 1) continue;
      const hw = Math.abs(e.x - c.x) * (W / 2);
      const hh = hw * (s.y / s.x);
      const x = (c.x + 1) * (W / 2);
      const y = (1 - c.y) * (H / 2);
      const box: [number, number, number, number] = [x - hw, y - hh, x + hw, y + hh];
      if (w.rank >= 0 && boxes.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
      boxes.push(box);
      names.add(w.name);
      out.add(w.id);
    }
    return out;
  }

  /** The pin the wearer is being guided to, or null: draws the ribbon on the floor. */
  setGuideTarget(pin: PinJson | null) {
    this.target = pin;
    this.updateRibbon();
  }

  private updateRibbon() {
    const t = this.target;
    const you = this.lastPose;
    this.ribbon.visible = !this.fp && t !== null && you !== null;
    if (!this.ribbon.visible) return;
    const y = (this.floor ?? you!.pos[1] - 1.5) + 0.03;
    const a = new THREE.Vector3(you!.pos[0], y, you!.pos[2]);
    const b = new THREE.Vector3(t!.pos[0], y, t!.pos[2]);
    const dir = b.clone().sub(a);
    if (dir.lengthSq() < 1e-4) {
      this.ribbon.visible = false;
      return;
    }
    const side = new THREE.Vector3(-dir.z, 0, dir.x).normalize().multiplyScalar(0.11 * (this.top ? Math.min(this.labelScale, 3) : 1));
    const N = 24;
    const pos: number[] = [];
    const at: number[] = [];
    const as: number[] = [];
    for (let i = 0; i <= N; i++) {
      const c = a.clone().lerp(b, i / N);
      pos.push(c.x - side.x, c.y, c.z - side.z, c.x + side.x, c.y, c.z + side.z);
      at.push(i / N, i / N);
      as.push(-1, 1);
    }
    const geo = this.ribbon.geometry;
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("aT", new THREE.Float32BufferAttribute(at, 1));
    geo.setAttribute("aS", new THREE.Float32BufferAttribute(as, 1));
    geo.setIndex(Array.from({ length: N }, (_, i) => [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2]).flat());
  }

  /** Signed bearing from the wearer's heading to a point, degrees clockwise (right positive);
   *  null before the first pose. */
  bearingTo(p: [number, number, number]): number | null {
    const f = this.view3();
    if (!f) return null;
    return bearingDeg(f.pos.toArray(), f.fwd.toArray(), f.up.toArray(), p);
  }

  /** Where pins sit in the camera frame (normalised 0..1, first-person views only), with their
   *  distance from the wearer. Pins behind or outside the frame are left out. */
  project(pins: PinJson[]): { pin: PinJson; x: number; y: number; dist: number }[] {
    const f = this.view3();
    if (!f || !this.lens) return [];
    const right = f.fwd.clone().cross(f.up).normalize();
    const tx = Math.tan((this.lens.hfov * Math.PI) / 360);
    const ty = tx / this.lens.aspect;
    const out: { pin: PinJson; x: number; y: number; dist: number }[] = [];
    for (const pin of pins) {
      if (isPerson(pin.label)) continue;
      const v = new THREE.Vector3(...pin.pos).sub(f.pos);
      const z = v.dot(f.fwd);
      if (z < 0.3) continue;
      const x = 0.5 + (0.5 * v.dot(right)) / z / tx;
      const y = 0.5 - (0.5 * v.dot(f.up)) / z / ty;
      if (x < -0.02 || x > 1.02 || y < -0.02 || y > 1.02) continue;
      out.push({ pin, x, y, dist: v.length() });
    }
    return out;
  }

  /** The wearer's eye: position, forward and up, with the live turn in first person. */
  private view3(): { pos: THREE.Vector3; fwd: THREE.Vector3; up: THREE.Vector3 } | null {
    if (this.fp && this.fpPose) {
      const q = this.fpPose;
      return { pos: q.pos, fwd: q.fwd.clone().applyAxisAngle(q.up, this.turn).normalize(), up: q.up.clone().normalize() };
    }
    const p = this.lastPose;
    if (!p) return null;
    return { pos: new THREE.Vector3(...p.pos), fwd: new THREE.Vector3(...p.fwd).normalize(), up: new THREE.Vector3(...p.up).normalize() };
  }

  // Hover shows a pin's label; a click (not a drag) picks it.
  private pickEvents() {
    const nearest = (ev: PointerEvent): number | null => {
      if (this.fp) return null;
      const r = this.canvas.getBoundingClientRect();
      let best: number | null = null;
      let bd = 22;
      const v = new THREE.Vector3();
      for (const [id, g] of this.pinMeshes) {
        if (!g.visible) continue;
        v.copy(g.position).project(this.camera);
        if (v.z > 1) continue;
        const d = Math.hypot(((v.x + 1) / 2) * r.width - (ev.clientX - r.left), ((1 - v.y) / 2) * r.height - (ev.clientY - r.top));
        if (d < bd) {
          bd = d;
          best = id;
        }
      }
      return best;
    };
    let down: [number, number] | null = null;
    this.canvas.addEventListener("pointermove", (ev) => {
      this.hover = nearest(ev);
      this.canvas.style.cursor = this.hover === null ? "" : "pointer";
    });
    this.canvas.addEventListener("pointerleave", () => (this.hover = null));
    this.canvas.addEventListener("pointerdown", (ev) => (down = [ev.clientX, ev.clientY]));
    this.canvas.addEventListener("pointerup", (ev) => {
      if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 5) return;
      const id = nearest(ev);
      const pin = id === null ? undefined : this.pinData.get(id);
      if (pin) this.onPick(pin);
    });
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
    // Floor = low percentile of height.
    ys.sort((a, b) => a - b);
    const floor = ys[Math.floor(ys.length * 0.03)];
    this.floor = floor;
    this.updateRibbon();
    if (this.top) {
      // Cut 1.5 m above the floor so walls read as lines, not a lid.
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
    const col = new Float32Array(points.length * 4);
    points.forEach((p, i) => {
      arr.set(p.slice(0, 3), i * 3);
      // Oldest first: the newest stretch is brightest.
      col.set([1, 1, 1, 0.7 * Math.pow((i + 1) / points.length, 1.6)], i * 4);
    });
    this.trail.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    this.trail.geometry.setAttribute("color", new THREE.BufferAttribute(col, 4));
    this.trail.geometry.computeBoundingSphere();
  }

  /** A frame's camera in the room (4x4, camera to room, OpenCV axes), from room_pose. */
  setRoomView(pose: number[][]) {
    const m = new THREE.Matrix4().set(...(pose.flat() as Parameters<THREE.Matrix4["set"]>));
    this.roomCam.matrixWorld.copy(m.multiply(new THREE.Matrix4().makeScale(1, -1, -1)));
    this.roomCam.matrixWorldInverse.copy(this.roomCam.matrixWorld).invert();
    this.roomViewAt = performance.now();
  }

  setPose(p: Pose) {
    this.lastPose = p;
    if (this.fp) {
      const fwd = new THREE.Vector3(...p.fwd);
      this.fpPose = {
        pos: new THREE.Vector3(...p.pos),
        fwd,
        up: new THREE.Vector3(...p.up),
        yaw: Math.atan2(fwd.x, fwd.z),
      };
      this.turn = 0;
      if (p.hfov && p.aspect && !this.lensFixed) {
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
    this.updateRibbon();
  }

  setPins(all: PinJson[]) {
    // Decision 8: the map never marks people. They stay in memory, just not drawn.
    const pins = all.filter((p) => !isPerson(p.label));
    const seen = new Set<number>();
    for (const p of pins) {
      seen.add(p.id);
      this.pinData.set(p.id, p);
      let obj = this.pinMeshes.get(p.id);
      if (!obj) {
        const g = new THREE.Group();
        const strong = p.kind === "object";
        // A calm memory-coloured halo; landmarks smaller and dimmer.
        const dot = glowSprite(hex(MEMORY), strong ? 0.26 : 0.16, strong ? 0.8 : 0.4);
        dot.renderOrder = 9; // after the splats, which do not write depth
        const sp = label(p.label, strong);
        (sp.material as THREE.SpriteMaterial).opacity = 0;
        sp.visible = false;
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
    if (this.hot !== null) {
      const p = this.pinData.get(this.hot);
      if (p) this.pulse.position.set(p.pos[0], p.pos[1] + 0.02, p.pos[2]);
    }
  }

  private showPulse(id: number, answer: boolean) {
    const p = this.pinData.get(id) ?? (this.target?.id === id ? this.target : undefined);
    if (!p) return;
    this.hot = id;
    this.hotAt = answer ? performance.now() : null;
    this.pulse.position.set(p.pos[0], p.pos[1] + 0.02, p.pos[2]);
    this.styleLabels();
  }

  clearHot() {
    this.hot = null;
    this.hotAt = null;
    this.pulse.visible = false;
  }

  /** Fly the camera to look at a pin, from the side the user is on. `answer`: the pin answers a
   *  question, so it pulses three times and holds bright; otherwise it is just selected. */
  flyTo(pin: PinJson, answer = true) {
    this.showPulse(pin.id, answer);
    if (this.top) return; // the whole floor plan is already in view
    const target = new THREE.Vector3(...pin.pos);
    const dir = this.rig.visible
      ? this.rig.position.clone().sub(target).setY(0)
      : new THREE.Vector3(1, 0, 1);
    if (dir.lengthSq() < 0.01) dir.set(1, 0, 1);
    dir.normalize().multiplyScalar(3.0);
    this.goTo(target.clone().add(dir).add(new THREE.Vector3(0, 1.8, 0)), target, 1.3);
  }

  /** Put the camera at `from`, looking at `at`, for a scripted path (the capture scenes). */
  lookFrom(from: [number, number, number], at: [number, number, number]) {
    this.tween = null;
    this.camera.position.set(...from);
    this.controls.target.set(...at);
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
    const now = performance.now();
    this.updatePins(now, ease(now - this.lastFrame));
    this.lastFrame = now;
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
    if (this.room) {
      // The room as outlines: only the room into the buffer, the rest of the map as usual,
      // then the edge pass on top. First person draws it from the frame's own PnP pose in the
      // room (the map pose lags a window), and not at all once that pose goes stale.
      const fresh = performance.now() - this.roomViewAt < ROOM_VIEW_STALE_MS;
      this.renderer.setRenderTarget(this.roomBuf);
      this.renderer.clear();
      if (this.fp && fresh) {
        const placed = this.room.matrix.clone();
        this.room.matrix.identity();
        this.roomCam.projectionMatrix.copy(this.camera.projectionMatrix);
        this.roomCam.projectionMatrixInverse.copy(this.camera.projectionMatrixInverse);
        this.renderer.render(this.room, this.roomCam);
        this.room.matrix.copy(placed);
        this.room.updateMatrixWorld();
      } else if (!this.fp) this.renderer.render(this.room, this.camera);
      this.renderer.setRenderTarget(null);
      this.room.visible = false;
      this.renderer.render(this.scene, this.camera);
      this.room.visible = true;
      this.renderer.autoClear = false;
      this.renderer.render(this.outlineScene, this.flatCam);
      this.renderer.autoClear = true;
      return;
    }
    this.renderer.render(this.scene, this.camera);
  };
}
