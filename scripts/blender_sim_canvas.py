"""The six training worlds side by side on one ground, as one Blender scene and one camera move:
a wide look at every room walking at once (I0, then I6), then a flight down into the bedroom
where the chair-finding walk replays, first try then final policy.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Rooms, furniture and walks come straight from the
recorded scene files (sim_canvas_plan.SCENES); nothing is moved or retimed beyond playback speed.

    blender -b --factory-startup --python scripts/blender_sim_canvas.py -- \
        --out data/demo-takes/sim/canvas [--res 1280x720 --samples 8 --blend PATH]
    blender -b PATH.blend --python scripts/blender_sim_canvas.py -- --render 1-160 --out DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_sim_room as R  # noqa: E402
import sim_canvas_plan as P  # noqa: E402
import sim_motion  # noqa: E402

CYAN = (0.25, 0.8, 1.0)
RED = (1.0, 0.16, 0.1)
AMBER = (1.0, 0.5, 0.12)
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return {a[k].lstrip("-"): a[k + 1] for k in range(0, len(a), 2)}


def emit_mat(name, col, strength, base=(0.02, 0.02, 0.025)):
    return R.mat(name, base, 0.6, emit=(col, strength))


def keep_hidden(objs, show, hide):
    """Render-visible only on frames show..hide-1."""
    for o in objs:
        o.hide_render = show > 1
        o.keyframe_insert("hide_render", frame=1)
        if show > 1:
            o.hide_render = False
            o.keyframe_insert("hide_render", frame=show)
        if hide <= P.FRAMES:
            o.hide_render = True
            o.keyframe_insert("hide_render", frame=hide)


def fcurves(action):
    """Every F-curve of an action, layered (Blender 4.4+) or not."""
    if not getattr(action, "layers", None):
        return list(action.fcurves)
    out = []
    for slot in action.slots:
        for layer in action.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot)
                if cb is not None:
                    out += list(cb.fcurves)
    return out


def trail(name, pts, mat, start, n_out, show, hide, z=0.04, r=0.045):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, (x, y) in zip(sp.points, pts, strict=True):
        p.co = (x, y, z, 1.0)
    cu.bevel_depth = r
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    cu.bevel_factor_mapping_end = "SEGMENTS"
    cu.bevel_factor_end = 0.0
    cu.keyframe_insert("bevel_factor_end", frame=start)
    cu.bevel_factor_end = 1.0
    cu.keyframe_insert("bevel_factor_end", frame=start + n_out - 1)
    for fc in fcurves(cu.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    cu.materials.append(mat)
    o = R._link(bpy.data.objects.new(name, cu))
    keep_hidden([o], show, hide)
    return o


def disc(name, x, y, r, mat, z=0.035):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=r, depth=0.012, location=(x, y, z))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


def ring(name, x, y, r, mat, z=0.04, thick=0.018):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=r, minor_radius=thick, major_segments=48, minor_segments=6, location=(x, y, z)
    )
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


def text(name, body, x, y, size, mat, z=0.005):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    try:
        cu.font = bpy.data.fonts.load(MONO, check_existing=True)
    except RuntimeError:
        pass
    cu.align_x = "LEFT"
    cu.align_y = "TOP"
    cu.materials.append(mat)
    o = R._link(bpy.data.objects.new(name, cu))
    o.location = (x, y, z)
    return o


def corner_brackets(name, bounds, mat, pad=0.32, arm=0.7, z=-0.15):
    x0, x1, y0, y1 = bounds
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    out = []
    for cx, sx in ((x0, 1), (x1, -1)):
        for cy, sy in ((y0, 1), (y1, -1)):
            cu = bpy.data.curves.new(name, "CURVE")
            cu.dimensions = "3D"
            sp = cu.splines.new("POLY")
            sp.points.add(2)
            for p, (x, y) in zip(
                sp.points, ((cx + sx * arm, cy), (cx, cy), (cx, cy + sy * arm)), strict=True
            ):
                p.co = (x, y, z, 1.0)
            cu.bevel_depth = 0.025
            cu.materials.append(mat)
            out.append(R._link(bpy.data.objects.new(name, cu)))
    return out


def build_world(world: str, wins: list, M, mats) -> dict:
    """Build one room at the origin with every walk shown in it; returns its info."""
    before = set(bpy.data.objects)
    sc = wins[0].scene()
    bounds = R.room_bounds(sc["walls"])
    R.build_room(sc, M, ("S", "E"))
    for o in [o for o in bpy.data.objects if o not in before and o.name.startswith("backdrop")]:
        bpy.data.objects.remove(o)
    chairs = R.build_furniture(sc, M, bounds)
    seat = R.seat_of(sc, chairs)
    rects = [
        (o["obb"][0], o["obb"][1], o["obb"][3], o["obb"][4], o["obb"][6]) for o in sc["objects"]
    ]
    turns = {}
    info = {"bounds": bounds, "walks": {}}
    for w in wins:
        s = w.scene()
        frames = s["frames"]
        n = len(frames)
        plan = sim_motion.plan(frames, s["motor_deg"], seat=seat and seat[:3], rects=rects)
        turns[w.key] = [m.chair_turn for m in plan]
        walker = R.Walker(s["motor_deg"], M)
        last = min(w.n_out(n), w.hide - w.start) + w.start - 1
        for f in range(w.start, last + 1):
            k = w.sim_at(f, n)
            walker.pose(plan[k], frames[k], f, seat and seat[3])
        R.finish_walker(walker)
        keep_hidden([walker.root, *walker.root.children_recursive], w.show, w.hide)
        # the trail it leaves, and where it touched something
        wide_i0 = w.arm == "I0" and w.key.startswith("wide")
        trail_hide = P.CLOSE_I0 if (world == P.HERO and w.key.startswith("wide")) else P.FRAMES + 1
        n_out = w.n_out(n)
        pts = [(fr["x"], fr["y"]) for fr in frames]
        tm = mats["trail_i0"] if w.arm == "I0" else mats["trail_i6"]
        trail(f"trail-{w.key}", pts, tm, w.start, n_out, w.show, trail_hide)
        hits = []
        prev = 0
        for k, fr in enumerate(frames):
            if fr["collisions"] > prev:
                f = int(round(w.out_of(k)))
                hits.append((k, f))
                d = disc(f"hit-{w.key}-{k}", fr["x"], fr["y"], 0.15, mats["hit"])
                keep_hidden([d], f, trail_hide)
                pr = ring(f"pulse-{w.key}-{k}", fr["x"], fr["y"], 0.18, mats["hit"], z=0.06)
                pr.scale = (1, 1, 1)
                pr.keyframe_insert("scale", frame=f)
                pr.scale = (4.5, 4.5, 1)
                pr.keyframe_insert("scale", frame=f + 12)
                keep_hidden([pr], f, min(f + 13, trail_hide))
            prev = fr["collisions"]
        info["walks"][w.key] = {
            "arm": w.arm,
            "n": n,
            "contacts": frames[-1]["collisions"],
            "clean": bool(s["metrics"]["success"]),
            "hits": hits,
            "xy": [[round(fr["x"], 3), round(fr["y"], 3)] for fr in frames],
            "coll": [fr["collisions"] for fr in frames],
            "wide_i0": wide_i0,
        }
    # chair turns follow whichever walk is on screen in this room
    if seat is not None and any(abs(t) > 1e-4 for ts in turns.values() for t in ts):
        root = chairs[sc["target"]][0]
        m0 = root.matrix_world.copy()
        from mathutils import Matrix

        c = Matrix.Translation((seat[0], seat[1], 0.0))
        for f in range(1, P.FRAMES + 1):
            w = next(w for w in wins if w.show <= f < w.hide)
            ts = turns[w.key]
            a = ts[w.sim_at(f, len(ts))]
            root.matrix_world = c @ Matrix.Rotation(a, 4, "Z") @ c.inverted() @ m0
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_euler", frame=f)
    # goal, ceiling light, footprint brackets, name tag
    gx, gy = sc["goal"]
    ring(f"goal-{world}", gx, gy, 0.32, mats["goal"], z=0.02, thick=0.022)
    x0, x1, y0, y1 = bounds
    ld = bpy.data.lights.new(f"ceiling-{world}", "AREA")
    ld.energy, ld.color, ld.shape, ld.size = 260, (1.0, 0.88, 0.76), "DISK", 1.8
    lo = R._link(bpy.data.objects.new(f"ceiling-{world}", ld))
    lo.location = ((x0 + x1) / 2, (y0 + y1) / 2, R.WALL_H - 0.05)
    corner_brackets(f"bracket-{world}", bounds, mats["bracket"])
    title, sub = P.TAG[world]
    idx = [w for row in P.GRID for w in row].index(world) + 1
    text(f"tag-{world}", f"W{idx}  {title}", x0 - 0.3, y0 - 0.5, 0.42, mats["tag"], z=-0.15)
    text(f"sub-{world}", sub, x0 - 0.3, y0 - 1.05, 0.28, mats["sub"], z=-0.15)
    info["tag"] = [x1 + 0.3, y1 + 0.3, R.WALL_H]  # top corner, for the overlay's live counter
    # move the whole room onto its cell
    root = bpy.data.objects.new(f"room-{world}", None)
    R._link(root)
    for o in bpy.data.objects:
        if o not in before and o.parent is None and o is not root:
            o.parent = root
    return info


def ground(mats):
    me = bpy.data.meshes.new("ground")
    s = (70.0, 50.0)
    me.from_pydata(
        [(-s[0], -s[1], 0), (s[0], -s[1], 0), (s[0], s[1], 0), (-s[0], s[1], 0)], [], [(0, 1, 2, 3)]
    )
    o = R._link(bpy.data.objects.new("ground", me))
    o.location = (0, 0, -0.165)
    m = bpy.data.materials.new("grid")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = nt.nodes.new
    L = nt.links.new
    tc = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(tc.outputs["Object"], sep.inputs[0])

    def line(u, period, w0, w1):
        a = N("ShaderNodeMath")
        a.operation = "MULTIPLY_ADD"
        a.inputs[1].default_value = 1.0 / period
        a.inputs[2].default_value = 0.5
        L(u, a.inputs[0])
        fr = N("ShaderNodeMath")
        fr.operation = "FRACT"
        L(a.outputs[0], fr.inputs[0])
        sub = N("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        L(fr.outputs[0], sub.inputs[0])
        ab = N("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        L(sub.outputs[0], ab.inputs[0])
        mr = N("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = w0 / period
        mr.inputs["From Max"].default_value = w1 / period
        mr.inputs["To Min"].default_value = 1.0
        mr.inputs["To Max"].default_value = 0.0
        L(ab.outputs[0], mr.inputs["Value"])
        return mr.outputs["Result"]

    def mx(a, b):
        n = N("ShaderNodeMath")
        n.operation = "MAXIMUM"
        L(a, n.inputs[0])
        L(b, n.inputs[1])
        return n.outputs[0]

    minor = mx(line(sep.outputs["X"], 1.0, 0.006, 0.018), line(sep.outputs["Y"], 1.0, 0.006, 0.018))
    major = mx(line(sep.outputs["X"], 5.0, 0.015, 0.035), line(sep.outputs["Y"], 5.0, 0.015, 0.035))
    wsum = N("ShaderNodeMath")
    wsum.operation = "MULTIPLY_ADD"
    wsum.inputs[1].default_value = 0.22
    L(minor, wsum.inputs[0])
    L(major, wsum.inputs[2])
    em = N("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.12, 0.45, 0.62, 1)
    L(wsum.outputs[0], em.inputs["Strength"])
    base = N("ShaderNodeBsdfDiffuse")
    base.inputs["Color"].default_value = (0.012, 0.016, 0.024, 1)
    add = N("ShaderNodeAddShader")
    L(base.outputs[0], add.inputs[0])
    L(em.outputs[0], add.inputs[1])
    out = N("ShaderNodeOutputMaterial")
    L(add.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    mats["grid"] = m


def world_and_sun():
    sc = bpy.context.scene
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    lp = nt.nodes.new("ShaderNodeLightPath")
    lit = nt.nodes.new("ShaderNodeBackground")
    lit.inputs["Color"].default_value = (0.78, 0.8, 0.84, 1)
    lit.inputs["Strength"].default_value = 0.3
    dark = nt.nodes.new("ShaderNodeBackground")
    dark.inputs["Color"].default_value = (0.006, 0.008, 0.013, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(lit.outputs[0], mix.inputs[1])
    nt.links.new(dark.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy, ld.color, ld.angle = 1.6, (1.0, 0.95, 0.88), math.radians(6)
    o = R._link(bpy.data.objects.new("sun", ld))
    o.rotation_euler = (math.radians(38), math.radians(-22), math.radians(-30))


def look(target, az, el, dist):
    az, el = math.radians(az), math.radians(el)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    return target + d * dist, d


def fit(cam, target, corners, az, el, span, x_at=0.5):
    """Distance (and a sideways target shift) so the corners fill `span` of the frame, centred
    at x_at of the width."""
    from bpy_extras.object_utils import world_to_camera_view

    scn = bpy.context.scene
    dist = 20.0
    t = target.copy()
    for _ in range(60):
        loc, d = look(t, az, el, dist)
        cam.location = loc
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        pts = [world_to_camera_view(scn, cam, p) for p in corners]
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        sx = max(xs) - min(xs)
        sy = max(ys) - min(ys)
        s = max(sx, sy)
        cx = (max(xs) + min(xs)) / 2
        right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
        width = 2 * dist * math.tan(cam.data.angle_x / 2)
        t = t + right * (cx - x_at) * width * 0.6
        if abs(s - span) < 0.002 and abs(cx - x_at) < 0.003:
            break
        dist *= 1 + (s - span) * 0.8
    return t, dist


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def camera(worlds: dict, offsets: dict):
    cd = bpy.data.cameras.new("cam")
    cd.lens = 32
    cd.clip_end = 400
    cam = R._link(bpy.data.objects.new("cam", cd))
    bpy.context.scene.camera = cam
    allc = []
    for w, inf in worlds.items():
        x0, x1, y0, y1 = inf["bounds"]
        ox, oy = offsets[w]
        allc += [
            Vector((x + ox, y + oy, z))
            for x in (x0, x1)
            for y in (y0 - 1.1, y1)
            for z in (0, R.WALL_H)
        ]
    c = sum(allc, Vector()) / len(allc)
    wide_t, wide_d = fit(cam, Vector((c.x, c.y, 0)), allc, -22, 56, 0.86, x_at=0.43)
    x0, x1, y0, y1 = worlds[P.HERO]["bounds"]
    ox, oy = offsets[P.HERO]
    hc = [Vector((x + ox, y + oy, z)) for x in (x0, x1) for y in (y0, y1) for z in (0, R.WALL_H)]
    ht = Vector(((x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy - 0.1, 0.35))
    close_t, close_d = fit(cam, ht, hc, -18, 64, 0.9, x_at=0.42)
    keys = []
    f0, f1 = P.FLY
    for f in range(1, P.FRAMES + 1):
        if f <= f0:  # slow drift over the canvas
            u = (f - 1) / (f0 - 1)
            t, az, el, dist = wide_t, -26 + 4 * u, 56, wide_d * (1.04 - 0.04 * u)
        elif f < f1:
            u = smooth((f - f0) / (f1 - f0))
            t = wide_t.lerp(close_t, u)
            az = -22 + 4 * u
            el = 56 + 8 * u
            dist = math.exp(math.log(wide_d) * (1 - u) + math.log(close_d) * u)
        else:
            u = (f - f1) / (P.FRAMES - f1)
            t, az, el, dist = close_t, -18 + 3 * u, 64, close_d * (1 - 0.03 * u)
        loc, d = look(t, az, el, dist)
        cam.location = loc
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        bpy.context.view_layer.update()
        mv = cam.matrix_world.inverted()
        proj = cam.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), x=R.W, y=R.H, scale_x=1, scale_y=1
        )
        full = proj @ mv
        keys.append([[round(full[i][j], 6) for j in range(4)] for i in range(4)])
    return keys


def build(a: dict):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.system.use_gpu_subdivision = False
    R.COLL = bpy.context.scene.collection
    R.W, R.H = (int(v) for v in a.get("res", "1280x720").split("x"))
    M = R.build_materials()
    mats = {
        "trail_i0": emit_mat("trail_i0", AMBER, 3.0),
        "trail_i6": emit_mat("trail_i6", CYAN, 3.0),
        "hit": emit_mat("hit", RED, 4.0),
        "goal": emit_mat("goal", (0.3, 1.0, 0.55), 3.0),
        "bracket": emit_mat("bracket", CYAN, 2.5),
        "tag": emit_mat("tag", (0.9, 0.95, 1.0), 3.0),
        "sub": emit_mat("sub", CYAN, 1.4),
    }
    wins = P.windows()
    worlds = {}
    bounds = {}
    for row in P.GRID:
        for w in row:
            info = build_world(w, [x for x in wins if x.world == w], M, mats)
            worlds[w] = info
            bounds[w] = info["bounds"]
            # off the origin before the next room is built there (posing works in world space)
            bpy.data.objects[f"room-{w}"].location = (*P.offsets(bounds)[w], 0.0)
    offsets = P.offsets(bounds)
    for w in worlds:
        bpy.data.objects[f"room-{w}"].location = (*offsets[w], 0.0)
    ground(mats)
    world_and_sun()
    scn = bpy.context.scene
    scn.render.resolution_x, scn.render.resolution_y = R.W, R.H
    keys = camera(worlds, offsets)
    R.setup_render(int(a.get("samples", 8)), 100)
    if hasattr(scn.eevee, "shadow_pool_size"):
        scn.eevee.shadow_pool_size = "1024"
    scn.frame_start, scn.frame_end = 1, P.FRAMES
    out = Path(a["out"])
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "w": R.W,
        "h": R.H,
        "fps": P.FPS,
        "cams": keys,
        "offsets": offsets,
        "worlds": worlds,
    }
    (out / "canvas.json").write_text(json.dumps(meta, separators=(",", ":")))
    if a.get("blend"):
        bpy.ops.wm.save_as_mainfile(filepath=a["blend"])


def render(a: dict):
    lo, hi = (int(v) for v in a["render"].split("-"))
    scn = bpy.context.scene
    scn.frame_start, scn.frame_end = lo, hi
    scn.render.image_settings.file_format = "PNG"
    d = Path(a["out"]) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(d / "#####")
    scn.render.use_overwrite = False
    bpy.ops.render.render(animation=True)


def main():
    a = args()
    if "render" in a:
        render(a)
    else:
        build(a)


if __name__ == "__main__":
    main()
