"""The parallel-walks shot: one bedroom per seed laid out in a grid, every walker replaying its own
solved walk at once, iteration after iteration, under one camera that opens close on a hero room
and pulls back over the field.

SIMULATED USER MODEL, NOT A HUMAN STUDY. Each tile is a real run: the walk exported by
scripts/export_sim_scene.py and solved by `blender_sim_room.py --track`. The room is the same in
every tile (the runs vary the sensing and walker noise, not the furniture). Rows are the belt
layouts: eight motors on top, the four built diagonals below.

    blender -b --factory-startup --python scripts/blender_sim_grid.py -- \
        --grid data/demo-takes/sim/grid --timeline data/demo-takes/sim/grid/timeline.json \
        --still 300 --out data/demo-takes/sim/preview/grid.png

`--frames DIR` (with `--first`/`--last`) renders the shot instead. The timeline comes from
scripts/build_sim_grid.py, which also composites the counters; this script writes the camera of
every frame (cams.json) and the screen box of every tile for it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_sim_room as R  # noqa: E402  (sits beside this script)

GAP_M = 1.5  # floor between neighbouring rooms
RING_OUT_M = 0.22  # the outcome ring runs this far outside the walls


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out: dict = {"samples": 24, "scale": 100}
    k = 0
    while k < len(a):
        out[a[k].lstrip("-")] = a[k + 1]
        k += 2
    return out


def coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def move_tree(root, dst):
    for o in [root, *root.children_recursive]:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        dst.objects.link(o)


def clone_walker(tpl, dst):
    """A walker sharing the template's mesh and armature data, with its own pose and pods."""
    root = tpl.root.copy()
    dst.objects.link(root)
    arm = tpl.arm.copy()
    arm.animation_data_clear()
    dst.objects.link(arm)
    arm.parent = root
    copies = {}
    for ch in tpl.arm.children:
        c = ch.copy()
        c.animation_data_clear()
        dst.objects.link(c)
        c.parent = arm
        for md in c.modifiers:
            if md.type == "ARMATURE":
                md.object = arm
        copies[ch] = c
    return root, arm, [copies[p] for p in tpl.pods]


def ring_mesh(name, x0, x1, y0, y1, w=0.07):
    """A flat rounded-rectangle band on the floor around one room."""
    import bmesh

    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    r = 0.35
    pts_out, pts_in = [], []
    cx = [(x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)]
    for px, py, a0 in cx:
        for s in range(9):
            a = math.radians(a0 + s * 90 / 8)
            pts_out.append((px + (r + w / 2) * math.cos(a), py + (r + w / 2) * math.sin(a)))
            pts_in.append((px + (r - w / 2) * math.cos(a), py + (r - w / 2) * math.sin(a)))
    vo = [bm.verts.new((x, y, 0.004)) for x, y in pts_out]
    vi = [bm.verts.new((x, y, 0.004)) for x, y in pts_in]
    n = len(vo)
    for k in range(n):
        bm.faces.new((vo[k], vo[(k + 1) % n], vi[(k + 1) % n], vi[k]))
    bm.to_mesh(me)
    bm.free()
    return me


def ring_material():
    m = bpy.data.materials.new("outcome_ring")
    m.use_nodes = True
    nt = m.node_tree
    b = R._bsdf(m)
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1)
    info = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(info.outputs["Color"], b.inputs["Emission Color"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 7.0
    nt.links.new(info.outputs["Alpha"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], b.inputs["Emission Strength"])
    return m


def ground_material():
    m = bpy.data.materials.new("void_floor")
    m.use_nodes = True
    nt = m.node_tree
    b = R._bsdf(m)
    b.inputs["Roughness"].default_value = 0.55
    # a faint metre grid on a dark slate floor, as in a simulator's ground plane
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    lines = []
    for ax in ("X", "Y"):
        fr = nt.nodes.new("ShaderNodeMath")
        fr.operation = "PINGPONG"
        fr.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs[ax], fr.inputs[0])
        st = nt.nodes.new("ShaderNodeMath")
        st.operation = "LESS_THAN"
        st.inputs[1].default_value = 0.012
        nt.links.new(fr.outputs["Value"], st.inputs[0])
        lines.append(st)
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    nt.links.new(lines[0].outputs["Value"], mx.inputs[0])
    nt.links.new(lines[1].outputs["Value"], mx.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.045, 0.048, 0.055, 1)
    mix.inputs["B"].default_value = (0.11, 0.115, 0.125, 1)
    nt.links.new(mx.outputs["Value"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    return m


def phase_rows(tr, start, nxt):
    """(absolute frame, value) rows of one solved walk placed at `start`, held until `nxt`."""
    out = {}
    for key, vals in tr["chans"].items():
        b, p = key.split("|")
        out[(None if b == "root" else b, p)] = held([tuple(v) for v in vals], start, nxt)
    return out


def merge(dst, src):
    for k, rows in src.items():
        dst.setdefault(k, []).extend(rows)


def held(vals, start, nxt):
    """Rows from `start`, cut before the next phase (a timed-out walk) or held up to it."""
    rows = [(start + k, v) for k, v in enumerate(vals) if nxt is None or start + k < nxt]
    if nxt is not None and rows[-1][0] < nxt - 1:
        rows.append((nxt - 1, rows[-1][1]))
    return rows


def cam_pose(cam, tgt, az, el, dist, lens):
    d = Vector(
        (
            math.sin(az) * math.cos(el),
            -math.cos(az) * math.cos(el),
            math.sin(el),
        )
    )
    cam.location = tgt + d * dist
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens


def fit_dist(cam, tgt, az, el, lens, corners, span_want=0.485):
    from bpy_extras.object_utils import world_to_camera_view

    dist = 10.0
    for _ in range(60):
        cam_pose(cam, tgt, az, el, dist, lens)
        bpy.context.view_layer.update()
        pts = [world_to_camera_view(bpy.context.scene, cam, p) for p in corners]
        span = max(max(abs(p.x - 0.5), abs(p.y - 0.5)) for p in pts)
        if abs(span - span_want) < 0.002:
            break
        dist *= 1 + (span - span_want)
    return dist


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def main():
    a = args()
    G = Path(a["grid"])
    tl = json.loads(Path(a["timeline"]).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.system.use_gpu_subdivision = False
    scene = bpy.context.scene
    room_c = coll("room")
    chair_c = coll("chair")
    tiles_c = coll("tiles")
    tpl_c = coll("templates")
    R.COLL = room_c
    first = json.loads((G / "scenes" / f"{tl['template']['eight']}.json").read_text())
    M = R.build_materials()
    bounds = R.room_bounds(first["walls"])
    R.build_room(first, M, ("S", "E"))
    chairs = R.build_furniture(first, M, bounds)
    seat = R.seat_of(first, chairs)
    croot, _ = chairs[first["target"]]
    move_tree(croot, chair_c)
    chair_c.instance_offset = (seat[0], seat[1], 0.0)
    R.setup_light(bounds)
    sun = bpy.data.objects["sun"]
    for c in list(sun.users_collection):
        c.objects.unlink(sun)
    scene.collection.objects.link(sun)
    sun.data.energy = 2.2
    bg = next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.5, 0.53, 0.58, 1)
    bg.inputs["Strength"].default_value = 0.35

    R.COLL = tpl_c
    tpl = {}
    for lay in ("eight", "diag4"):
        sc = json.loads((G / "scenes" / f"{tl['template'][lay]}.json").read_text())
        w = R.Walker(sc["motor_deg"], M)
        w.mesh.modifiers["smooth"].show_viewport = True
        for p in w.pods:
            p.animation_data_clear()
        w.arm.animation_data.action = None
        tpl[lay] = w
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.collection in (room_c, chair_c, tpl_c):
            lc.exclude = True

    x0, x1, y0, y1 = bounds
    pitch_x = (x1 - x0) + GAP_M
    pitch_y = (y1 - y0) + GAP_M
    cols, rows = tl["cols"], tl["rows"]
    ring_m = ring_material()
    ground = bpy.data.objects.new("ground", bpy.data.meshes.new("ground"))
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=400)
    bm.to_mesh(ground.data)
    bm.free()
    ground.data.materials.append(ground_material())
    ground.location = (0, 0, -0.004)
    scene.collection.objects.link(ground)

    tiles = tl["tiles"]  # [{"id": "eight-1002", "layout": "eight", "col": c, "row": r}]
    starts = [p["start"] for p in tl["phases"]]
    boxes = {}
    for t in tiles:
        off = Vector(
            ((t["col"] - (cols - 1) / 2) * pitch_x, -(t["row"] - (rows - 1) / 2) * pitch_y, 0)
        )
        ox, oy = off.x - (x0 + x1) / 2, off.y - (y0 + y1) / 2
        base = Matrix.Translation((ox, oy, 0))
        inst = bpy.data.objects.new(f"room_{t['id']}", None)
        inst.instance_type = "COLLECTION"
        inst.instance_collection = room_c
        inst.matrix_world = base
        tiles_c.objects.link(inst)
        ch = bpy.data.objects.new(f"chair_{t['id']}", None)
        ch.instance_type = "COLLECTION"
        ch.instance_collection = chair_c
        ch.location = (ox + seat[0], oy + seat[1], 0)
        tiles_c.objects.link(ch)
        holder = bpy.data.objects.new(f"tile_{t['id']}", None)
        holder.matrix_world = base
        tiles_c.objects.link(holder)
        root, arm, pods = clone_walker(tpl[t["layout"]], tiles_c)
        root.parent = holder
        if t["id"] != tl["hero"]:  # 60 subdivided bodies stall EEVEE; only the hero is seen close
            for c in arm.children:
                for md in [md for md in c.modifiers if md.type == "SUBSURF"]:
                    c.modifiers.remove(md)
        ring = bpy.data.objects.new(
            f"ring_{t['id']}",
            ring_mesh(
                f"ring_{t['id']}",
                x0 - R.WALL_T - RING_OUT_M,
                x1 + R.WALL_T + RING_OUT_M,
                y0 - R.WALL_T - RING_OUT_M,
                y1 + R.WALL_T + RING_OUT_M,
            ),
        )
        ring.data.materials.append(ring_m)
        ring.parent = holder
        ring.color = (0, 0, 0, 0)
        tiles_c.objects.link(ring)
        boxes[t["id"]] = [ox + x0, ox + x1, oy + y0, oy + y1]

        root_ch, bone_ch, chair_rows, ring_rows = {}, {}, [], []
        pod_rows = [[] for _ in pods]
        for pi, ph in enumerate(tl["phases"]):
            start = ph["start"] + ph.get("delay", {}).get(t["id"], 0)
            nxt = starts[pi + 1] if pi + 1 < len(starts) else None
            tr = json.loads((G / "tracks" / f"{ph['arm']}-{t['id']}.json").read_text())
            rows_ = phase_rows(tr, start, nxt)
            merge(root_ch, {k: v for k, v in rows_.items() if k[0] is None})
            merge(bone_ch, {k: v for k, v in rows_.items() if k[0] is not None})
            if start > ph["start"]:  # a late starter waits at the start pose
                for d in (root_ch, bone_ch):
                    for k in d:
                        first_row = next(r for r in d[k] if r[0] == start)
                        d[k].append((ph["start"], first_row[1]))
                        d[k].sort(key=lambda r: r[0])
            chair_rows += held([(0.0, 0.0, c) for c in tr["chair_turn"]], start, nxt)
            for k, p in enumerate(pod_rows):
                p += held([(f[k], 0.0, 0.0, 1.0) for f in tr["felt"]], start, nxt)
            ring_rows += [(f, tuple(c)) for f, c in tl["rings"][t["id"]][pi]]
        R.bake(root, f"root_{t['id']}", root_ch)
        R.bake(arm, f"pose_{t['id']}", bone_ch)
        R.bake(ch, f"chair_{t['id']}", {(None, "rotation_euler"): chair_rows})
        for p, rws in zip(pods, pod_rows, strict=True):
            R.bake(p, f"pod_{t['id']}_{p.name}", {(None, "color"): rws})
        R.bake(ring, f"ring_{t['id']}", {(None, "color"): sorted(ring_rows)})

    # camera: close on the hero room, out over the field, and back in
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.dof.use_dof = True
    hb = boxes[tl["hero"]]
    hero_t = Vector(((hb[0] + hb[1]) / 2, (hb[2] + hb[3]) / 2 - 0.1, 0.35))
    hero_c = [Vector((x, y, z)) for x in hb[:2] for y in hb[2:] for z in (0.0, R.WALL_H)]
    all_x = [v for b in boxes.values() for v in b[:2]]
    all_y = [v for b in boxes.values() for v in b[2:]]
    wide_t = Vector(((min(all_x) + max(all_x)) / 2, (min(all_y) + max(all_y)) / 2, 0.0))
    wide_c = [
        Vector((x, y, 0.0)) for x in (min(all_x), max(all_x)) for y in (min(all_y), max(all_y))
    ]
    shots = {}
    for name, (tgt, az, el, lens, corners, span) in {
        "close": (hero_t, -18.0, 66.0, 32.0, hero_c, 0.485),
        "wide": (wide_t, -14.0, 52.0, 40.0, wide_c, 0.47),
    }.items():
        az, el = math.radians(az), math.radians(el)
        shots[name] = (tgt, az, el, fit_dist(cam, tgt, az, el, lens, corners, span), lens)
    keys = tl["camera"]  # [[frame, shot name], ...]: hold, then ease to the next key

    def at(f):
        for (f0, s0), (f1, s1) in zip(keys, keys[1:], strict=False):
            if f0 <= f <= f1:
                u = ease((f - f0) / max(f1 - f0, 1))
                a0, a1 = shots[s0], shots[s1]
                tgt = a0[0].lerp(a1[0], u)
                az = a0[1] + (a1[1] - a0[1]) * u
                el = a0[2] + (a1[2] - a0[2]) * u
                dist = math.exp(math.log(a0[3]) + (math.log(a1[3]) - math.log(a0[3])) * u)
                lens = a0[4] + (a1[4] - a0[4]) * u
                return tgt, az, el, dist, lens
        s = shots[keys[-1][1]]
        return s

    focus = bpy.data.objects.new("focus", None)
    scene.collection.objects.link(focus)
    cam.data.dof.focus_object = focus
    f_first, f_last = 1, tl["frames"]
    cams = {}
    for f in range(f_first, f_last + 1):
        tgt, az, el, dist, lens = at(f)
        cam_pose(cam, tgt, az, el, dist, lens)
        focus.location = tgt
        # shallow enough that the far rows soften once the field is in view
        cam.data.dof.aperture_fstop = max(1.2, 0.09 * dist)
        for o, paths in ((cam, ("location", "rotation_euler")), (focus, ("location",))):
            for p in paths:
                o.keyframe_insert(p, frame=f)
        cam.data.keyframe_insert("lens", frame=f)
        cam.data.dof.keyframe_insert("aperture_fstop", frame=f)
    R.setup_render(int(a["samples"]), int(a["scale"]))
    for f in range(f_first, f_last + 1):
        if a.get("still") is not None and f != int(a["still"]):
            continue
        if a.get("frames") and not (int(a.get("first", 1)) <= f <= int(a.get("last", f_last))):
            continue
        scene.frame_set(f)
        mv = cam.matrix_world.inverted()
        proj = cam.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), x=R.W, y=R.H, scale_x=1, scale_y=1
        )
        full = proj @ mv
        cams[f] = [[full[i][j] for j in range(4)] for i in range(4)]
    meta = {"w": R.W, "h": R.H, "boxes": boxes, "ndc_from_world": cams}
    if a.get("blend"):
        bpy.ops.wm.save_as_mainfile(filepath=a["blend"])
    if a.get("still") is not None:
        out = Path(a["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        scene.frame_set(int(a["still"]))
        scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        out.with_suffix(".cams.json").write_text(json.dumps(meta))
        return
    d = Path(a["frames"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"cams-{a.get('first', 1)}.json").write_text(json.dumps(meta))
    scene.frame_start = int(a.get("first", 1))
    scene.frame_end = int(a.get("last", f_last))
    scene.render.filepath = str(d / "#####")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
