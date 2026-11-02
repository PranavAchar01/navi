"""Render a simulated walk in Pranav's bedroom with Blender (Eevee): the room as an architectural
set, furniture built to the twin's measured boxes, and a rigged walker with blindfold, glasses
and belt. Reads the scene file from scripts/export_sim_scene.py.

    blender -b --factory-startup --python scripts/blender_sim_room.py -- \
        --scene data/demo-takes/sim/scenes/avoid_chair-eight-1001.json \
        --still 60 --out data/demo-takes/sim/preview/hero_base.png

`--frames DIR` renders every frame to DIR/%05d.png instead. Either way the camera is written
next to the output as cam.json (world-to-pixel matrix) for the overlay pass.

Look references: his footage (white walls, light grey laminate, beige upholstered bed, black
standing desk with an orange mat, white folding table under the whiteboard, black mesh chair).
Assets and licences: data/assets/sim-hifi/PROVENANCE.md.
Desk, monitor, keyboard, laptop and chair are Sketchfab models via Objaverse, CC-BY: "Standing
Desk" by Ryan_Nein, "Monitor" by microsoft, "Wireless keyboard" by dmitriyslavyanskiy, "Laptop"
by Aullwen, "Office Chair" by FangedAnteater (data/assets/sim-hifi/objaverse/PROVENANCE.json).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "data" / "assets" / "sim-hifi"
WALL_T = 0.12
WALL_H = 2.6
CUT_H = 1.05  # the walls nearest the camera are cut here so the room reads from above
DOOR_H = 2.08
W, H = 1920, 1080
RAYTRACE = False


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out: dict = {"still": None, "frames": None, "out": None, "samples": 48, "scale": 100}
    k = 0
    while k < len(a):
        key = a[k].lstrip("-")
        out[key] = a[k + 1]
        k += 2
    return out


# ---------------------------------------------------------------- materials


def _bsdf(m):
    return next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def mat(name, rgb, rough=0.6, metal=0.0, coat=0.0, sheen=0.0, emit=None, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = _bsdf(m)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Sheen Weight"].default_value = sheen
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[0], 1)
        b.inputs["Emission Strength"].default_value = emit[1]
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


def textured(name, stem, tint, scale_m, rough_mul=1.0, sat=1.0, val=1.0, normal=1.0):
    """A Poly Haven PBR set (diff, nor_gl, rough) mapped in object space at real-world scale,
    with the colour pulled toward his room (`tint` multiplies, sat/val adjust)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = _bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale_m, 1 / scale_m, 1 / scale_m)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def img(kind, colour):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(ASSETS / "tex" / f"{stem}_{kind}_2k.jpg"))
        if not colour:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = "BOX"
        n.projection_blend = 0.2
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img("diff", True)
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = sat
    hsv.inputs["Value"].default_value = val
    nt.links.new(d.outputs["Color"], hsv.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs["B"].default_value = (*tint, 1)
    nt.links.new(hsv.outputs["Color"], mix.inputs["A"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    r = img("rough", False)
    rm = nt.nodes.new("ShaderNodeMath")
    rm.operation = "MULTIPLY"
    rm.inputs[1].default_value = rough_mul
    nt.links.new(r.outputs["Color"], rm.inputs[0])
    nt.links.new(rm.outputs["Value"], b.inputs["Roughness"])
    n = img("nor_gl", False)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal
    nt.links.new(n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


def build_materials() -> dict:
    M = {}
    M["floor"] = textured(
        "floor", "laminate_floor_02", (0.9, 0.89, 0.87), 1.7, sat=0.22, val=1.05, rough_mul=0.9
    )
    M["fabric"] = textured("fabric", "cotton_jersey", (0.83, 0.72, 0.55), 0.35, sat=0.0, val=1.3)
    M["duvet"] = textured("duvet", "cotton_jersey", (0.74, 0.69, 0.6), 0.3, sat=0.0, val=1.0)
    M["sheet"] = textured("sheet", "cotton_jersey", (0.86, 0.76, 0.6), 0.3, sat=0.0, val=1.1)
    M["wall"] = mat("wall", (0.88, 0.875, 0.86), 0.85)
    M["cap"] = mat("cap", (0.16, 0.16, 0.17), 0.9)
    M["trim"] = mat("trim", (0.92, 0.92, 0.9), 0.35, coat=0.2)
    M["door"] = mat("door", (0.9, 0.9, 0.88), 0.4, coat=0.15)
    M["black"] = mat("black", (0.025, 0.025, 0.028), 0.45)
    M["black_matte"] = mat("black_matte", (0.035, 0.035, 0.04), 0.75)
    M["metal"] = mat("metal", (0.55, 0.56, 0.58), 0.35, metal=1.0)
    M["grey_metal"] = mat("grey_metal", (0.32, 0.33, 0.35), 0.4, metal=0.8)
    M["white_plastic"] = mat("white_plastic", (0.86, 0.86, 0.84), 0.35)
    M["orange"] = mat("orange", (0.85, 0.33, 0.06), 0.8)
    M["screen"] = mat("screen", (0.02, 0.025, 0.035), 0.15, emit=((0.32, 0.42, 0.62), 0.6))
    M["laptop"] = mat("laptop", (0.62, 0.63, 0.65), 0.3, metal=0.9)
    M["mesh"] = mat("mesh", (0.04, 0.04, 0.045), 0.8, sheen=0.4)
    M["wood_dark"] = mat("wood_dark", (0.12, 0.08, 0.05), 0.55)
    M["cloth_grey"] = mat("cloth_grey", (0.32, 0.32, 0.33), 0.9, sheen=0.6)
    M["cloth_dark"] = mat("cloth_dark", (0.06, 0.06, 0.07), 0.9, sheen=0.6)
    M["cloth_red"] = mat("cloth_red", (0.55, 0.06, 0.07), 0.85, sheen=0.5)
    M["cloth_white"] = mat("cloth_white", (0.8, 0.8, 0.78), 0.9, sheen=0.5)
    M["cardboard"] = mat("cardboard", (0.55, 0.38, 0.22), 0.85)
    M["glass"] = mat("glass", (0.8, 0.88, 0.95), 0.05, emit=((0.85, 0.9, 1.0), 1.2))
    M["board"] = mat("board", (0.95, 0.95, 0.95), 0.2, coat=0.6)
    M["rug"] = rug_material()
    M["backdrop"] = mat("backdrop", (0.42, 0.43, 0.45), 1.0)
    return M


def rug_material():
    """His round black rug with the white smiley ring, drawn in the shader."""
    m = bpy.data.materials.new("rug")
    m.use_nodes = True
    nt = m.node_tree
    b = _bsdf(m)
    b.inputs["Roughness"].default_value = 1.0
    b.inputs["Sheen Weight"].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    ring = nt.nodes.new("ShaderNodeMapRange")  # white band 0.62..0.74 of the radius
    ring.interpolation_type = "LINEAR"
    nt.links.new(ln.outputs["Value"], ring.inputs["Value"])
    ring.inputs["From Min"].default_value = 0.0
    ring.inputs["From Max"].default_value = 1.0
    cr = nt.nodes.new("ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    els[0].position, els[0].color = 0.6, (0.02, 0.02, 0.025, 1)
    els[1].position, els[1].color = 0.62, (0.85, 0.85, 0.82, 1)
    e2 = els.new(0.73)
    e2.color = (0.85, 0.85, 0.82, 1)
    e3 = els.new(0.75)
    e3.color = (0.02, 0.02, 0.025, 1)
    cr.color_ramp.interpolation = "CONSTANT"
    nt.links.new(ring.outputs["Result"], cr.inputs["Fac"])
    # only the lower half of the ring (the smile) stays white
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "LESS_THAN"
    gt.inputs[1].default_value = -0.1
    nt.links.new(sep.outputs["Y"], gt.inputs[0])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.inputs["A"].default_value = (0.02, 0.02, 0.025, 1)
    nt.links.new(gt.outputs["Value"], mx.inputs["Factor"])
    nt.links.new(cr.outputs["Color"], mx.inputs["B"])
    nt.links.new(mx.outputs["Result"], b.inputs["Base Color"])
    return m


# ---------------------------------------------------------------- geometry helpers

COLL = None


def _link(o):
    COLL.objects.link(o)
    return o


def box(name, c, half, m, yaw=0.0, bevel=0.006, seg=3, subdiv=0):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    me.transform(Matrix.Diagonal((2 * half[0], 2 * half[1], 2 * half[2], 1.0)))
    o = _link(bpy.data.objects.new(name, me))
    o.location = c
    o.rotation_euler = (0, 0, yaw)
    me.materials.append(m)
    if bevel > 0:
        bv = o.modifiers.new("bevel", "BEVEL")
        bv.width = min(bevel, 0.49 * min(2 * x for x in half))
        bv.segments = seg
        bv.harden_normals = subdiv == 0
    if subdiv:
        sd = o.modifiers.new("sub", "SUBSURF")
        sd.levels = sd.render_levels = subdiv
    for p in me.polygons:
        p.use_smooth = True
    return o


def cyl(name, x, y, z0, z1, r, m, verts=24, rot=None):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r, radius2=r, depth=z1 - z0)
    bm.to_mesh(me)
    bm.free()
    o = _link(bpy.data.objects.new(name, me))
    o.location = (x, y, (z0 + z1) / 2)
    if rot is not None:
        o.rotation_euler = rot
    me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = True
    return o


def blob(name, c, half, m, seed=0, strength=0.25, yaw=0.0):
    """A soft heap (clothes, a bag): a subdivided sphere squashed to the box, then noise."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    o = _link(bpy.data.objects.new(name, me))
    o.scale = half
    o.location = c
    o.rotation_euler = (0, 0, yaw)
    me.materials.append(m)
    tx = bpy.data.textures.new(f"{name}_n", "CLOUDS")
    tx.noise_scale = 0.45
    tx.noise_depth = 2
    d = o.modifiers.new("disp", "DISPLACE")
    d.texture = tx
    d.strength = strength
    d.texture_coords = "GLOBAL"
    for p in me.polygons:
        p.use_smooth = True
    return o


def rot2(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c - y * s, x * s + y * c


# ---------------------------------------------------------------- the room


def room_bounds(walls):
    xs = [b for b in walls if b[3] < b[4]]  # thin in x: west and east walls
    ys = [b for b in walls if b[3] >= b[4]]
    x0 = min(b[0] + b[3] for b in xs)
    x1 = max(b[0] - b[3] for b in xs)
    y0 = min(b[1] + b[4] for b in ys)
    y1 = max(b[1] - b[4] for b in ys)
    return x0, x1, y0, y1


def build_room(sc, M, cut_sides):
    x0, x1, y0, y1 = room_bounds(sc["walls"])
    t = WALL_T
    # floor slab with a dark plinth showing the cut
    box(
        "floor",
        ((x0 + x1) / 2, (y0 + y1) / 2, -0.01),
        ((x1 - x0) / 2, (y1 - y0) / 2, 0.01),
        M["floor"],
        bevel=0,
    )
    box(
        "slab",
        ((x0 + x1) / 2, (y0 + y1) / 2, -0.09),
        ((x1 - x0) / 2 + t, (y1 - y0) / 2 + t, 0.07),
        M["cap"],
        bevel=0.004,
    )
    box("backdrop", ((x0 + x1) / 2, (y0 + y1) / 2, -0.17), (40, 40, 0.01), M["backdrop"], bevel=0)
    # openings per side: (lo, hi, z0, z1), along the wall's running axis
    gaps = {"S": [], "N": [], "W": [], "E": []}
    for f in sc["flush"]:
        if f["cls"] != "door":
            continue
        cx, cy, _ = f["centre"]
        sx, sy, _ = f["size"]
        if sy < sx:
            side = "N" if cy > (y0 + y1) / 2 else "S"
            gaps[side].append((cx - sx / 2, cx + sx / 2, 0.0, DOOR_H, f))
        else:
            side = "E" if cx > (x0 + x1) / 2 else "W"
            gaps[side].append((cy - sy / 2, cy + sy / 2, 0.0, DOOR_H, f))
    bed = next((o for o in sc["objects"] if o["label"] == "bed"), None)
    if bed is not None:  # the window above his headboard
        wx = bed["obb"][0]
        gaps["S"].append((wx - 0.6, wx + 0.6, 0.95, 2.15, None))
    sides = {
        "S": (x0 - t, x1 + t, y0 - t / 2, "x"),
        "N": (x0 - t, x1 + t, y1 + t / 2, "x"),
        "W": (y0, y1, x0 - t / 2, "y"),
        "E": (y0, y1, x1 + t / 2, "y"),
    }
    for side, (a0, a1, fixed, axis) in sides.items():
        top = CUT_H if side in cut_sides else WALL_H
        gs = sorted(gaps[side])
        pieces = []  # (lo, hi, z0, z1)
        cur = a0
        for lo, hi, z0, z1, _ in gs:
            pieces.append((cur, lo, 0.0, top))
            if z0 > 0:
                pieces.append((lo, hi, 0.0, min(z0, top)))
            if z1 < top:
                pieces.append((lo, hi, z1, top))
            cur = hi
        pieces.append((cur, a1, 0.0, top))
        for k, (lo, hi, z0, z1) in enumerate(pieces):
            if hi - lo < 1e-3 or z1 - z0 < 1e-3:
                continue
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            c = (mid, fixed, (z0 + z1) / 2) if axis == "x" else (fixed, mid, (z0 + z1) / 2)
            hs = (half, t / 2, (z1 - z0) / 2) if axis == "x" else (t / 2, half, (z1 - z0) / 2)
            box(f"wall_{side}{k}", c, hs, M["wall"], bevel=0.002)
            if z1 == top and side in cut_sides:  # dark section cap
                cc = (c[0], c[1], top + 0.004)
                ch = (hs[0], hs[1], 0.004)
                box(f"cap_{side}{k}", cc, ch, M["cap"], bevel=0)
            if z0 == 0.0:  # baseboard on the room side
                inward = {"S": 1, "N": -1, "W": 1, "E": -1}[side]
                off = t / 2 + 0.006
                if axis == "x":
                    bc, bh = (mid, fixed + inward * off, 0.05), (half, 0.006, 0.05)
                else:
                    bc, bh = (fixed + inward * off, mid, 0.05), (0.006, half, 0.05)
                box(f"base_{side}{k}", bc, bh, M["trim"], bevel=0.002)
        for lo, hi, z0, z1, f in gs:
            inward = {"S": 1, "N": -1, "W": 1, "E": -1}[side]
            mid = (lo + hi) / 2
            if f is None:  # window: sill, frame and the shutter slats, lit from outside
                build_window(side, axis, fixed, inward, lo, hi, z0, min(z1, top), M)
            else:
                build_door(side, axis, fixed, inward, lo, hi, min(z1, top), f["open"], M)
    return x0, x1, y0, y1


def _place(axis, fixed, along, normal_off, z):
    return (along, fixed + normal_off, z) if axis == "x" else (fixed + normal_off, along, z)


def _half(axis, along, normal, z):
    return (along, normal, z) if axis == "x" else (normal, along, z)


def build_door(side, axis, fixed, inward, lo, hi, top, is_open, M):
    w = hi - lo
    mid = (lo + hi) / 2
    # casing around the opening, on the room side
    for k, (a, hw) in enumerate(((lo - 0.035, 0.035), (hi + 0.035, 0.035))):
        box(
            f"case_{side}{lo:.2f}_{k}",
            _place(axis, fixed, a, inward * (WALL_T / 2 + 0.01), top / 2),
            _half(axis, hw, 0.01, top / 2),
            M["trim"],
            bevel=0.003,
        )
    if top >= DOOR_H:
        box(
            f"head_{side}{lo:.2f}",
            _place(axis, fixed, mid, inward * (WALL_T / 2 + 0.01), DOOR_H + 0.04),
            _half(axis, w / 2 + 0.07, 0.01, 0.04),
            M["trim"],
            bevel=0.003,
        )
    leaf_h = min(DOOR_H - 0.01, top)
    if is_open:  # swung 95 degrees into the room against the wall
        hinge = lo
        n = inward * (WALL_T / 2 + w / 2)
        c = _place(axis, fixed, hinge + 0.03, n, leaf_h / 2)
        box(f"leaf_{side}", c, _half(axis, 0.02, w / 2 - 0.01, leaf_h / 2), M["door"], bevel=0.004)
        return
    c = _place(axis, fixed, mid, 0.0, leaf_h / 2)
    box(
        f"leaf_{side}{lo:.2f}",
        c,
        _half(axis, w / 2 - 0.005, 0.02, leaf_h / 2),
        M["door"],
        bevel=0.004,
    )
    for k, (z0, z1) in enumerate(((0.12, 0.95), (1.1, min(1.95, leaf_h - 0.08)))):
        if z1 <= z0:
            continue
        pc = _place(axis, fixed, mid, inward * 0.024, (z0 + z1) / 2)
        box(
            f"panel_{side}{lo:.2f}_{k}",
            pc,
            _half(axis, w / 2 - 0.1, 0.006, (z1 - z0) / 2),
            M["door"],
            bevel=0.008,
        )
    if leaf_h > 1.0:
        kc = _place(axis, fixed, hi - 0.08, inward * 0.05, 1.0)
        cyl(f"knob_{side}{lo:.2f}", kc[0], kc[1], 0.985, 1.015, 0.03, M["metal"])


def build_window(side, axis, fixed, inward, lo, hi, z0, z1, M):
    mid = (lo + hi) / 2
    w = hi - lo
    sc_ = _place(axis, fixed, mid, inward * (WALL_T / 2 + 0.03), z0 - 0.015)
    box(f"sill_{side}", sc_, _half(axis, w / 2 + 0.06, 0.05, 0.015), M["trim"], bevel=0.004)
    if z1 - z0 < 0.3:  # cut away: only the sill shows
        return
    gc = _place(axis, fixed, mid, -inward * 0.03, (z0 + z1) / 2)
    box(f"glass_{side}", gc, _half(axis, w / 2, 0.005, (z1 - z0) / 2), M["glass"], bevel=0)
    n = int((z1 - z0) / 0.09)
    for k in range(n):  # plantation shutter louvres
        z = z0 + 0.05 + k * 0.09
        lc = _place(axis, fixed, mid, inward * 0.01, z)
        o = box(
            f"louvre_{side}{k}", lc, _half(axis, w / 2 - 0.03, 0.035, 0.006), M["trim"], bevel=0.003
        )
        o.rotation_euler = (math.radians(35), 0, 0) if axis == "x" else (0, math.radians(35), 0)


# ---------------------------------------------------------------- furniture


def build_bed(o, M, wall_y0):
    cx, cy, _, hx, hy, hz, yaw = o["obb"]
    top = 2 * hz
    head_s = -1 if abs((cy - hy) - wall_y0) < abs(cy + hy - wall_y0) else 1  # headboard side
    head_y = cy + head_s * (hy - 0.05)
    # upholstered frame: rails, legs, headboard
    box("bed_base", (cx, cy, 0.24), (hx, hy, 0.12), M["fabric"], bevel=0.03, seg=4)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(
                "bed_leg",
                cx + sx * (hx - 0.08),
                cy + sy * (hy - 0.08),
                0.0,
                0.12,
                0.025,
                M["black"],
            )
    box(
        "headboard",
        (cx, head_y, top / 2 + 0.06),
        (hx + 0.01, 0.05, top / 2 - 0.06),
        M["fabric"],
        bevel=0.045,
        seg=5,
    )
    # mattress and fitted sheet
    mh = (cx, cy - head_s * 0.03, 0.47)
    box("mattress", mh, (hx - 0.04, hy - 0.1, 0.11), M["sheet"], bevel=0.06, seg=5)
    # duvet over the foot two thirds: a sheet draped over the mattress edges, then wrinkled
    mt = 0.6
    span_y = hy * 1.25
    foot_y = cy - head_s * (hy - 0.02)
    dy = foot_y + head_s * span_y / 2
    me = bpy.data.meshes.new("duvet")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=70, y_segments=70, size=0.5)
    sx_, sy_ = 2 * hx + 0.5, span_y + 0.25
    for v in bm.verts:
        x, y = v.co.x * sx_, v.co.y * sy_
        edge_x = hx + 0.015
        over_x = max(0.0, abs(x) - edge_x)
        over_y = max(0.0, (foot_y - (dy + y)) * head_s)  # past the foot end
        drop = min(0.3, over_x + over_y)
        x = math.copysign(min(abs(x), edge_x), x)
        if over_y > 0:
            y = foot_y - head_s * 0.015 - dy
        v.co = Vector((x, y, mt - drop))
    bm.to_mesh(me)
    bm.free()
    d = _link(bpy.data.objects.new("duvet", me))
    d.location = (cx, dy, 0.0)
    me.materials.append(M["duvet"])
    tx = bpy.data.textures.new("duvet_n", "CLOUDS")
    tx.noise_scale = 0.11
    tx.noise_depth = 3
    dm = d.modifiers.new("crumple", "DISPLACE")
    dm.texture = tx
    dm.strength = 0.045
    dm.mid_level = 0.0
    dm.texture_coords = "GLOBAL"
    sd = d.modifiers.new("sub", "SUBSURF")
    sd.levels = sd.render_levels = 1
    for p in me.polygons:
        p.use_smooth = True
    # pillows at the head
    for k, px in enumerate((-hx / 2 + 0.09, hx / 2 - 0.09)):
        p = box(
            f"pillow{k}",
            (cx + px, head_y - head_s * 0.24, 0.62),
            (hx / 2 - 0.16, 0.13, 0.055),
            M["sheet"],
            bevel=0.06,
            seg=4,
            subdiv=1,
        )
        p.rotation_euler = (math.radians(-head_s * 12), 0, math.radians(3 - 6 * k))


def glb(stem, at, yaw, scale, front):
    """Import a sourced model (data/assets/sim-hifi/objaverse, CC-BY, see PROVENANCE.json) under
    one empty: footprint centred on at[0:2], bottom on at[2], turned so its front (model-frame
    angle `front`) points along `yaw`. `scale` is one factor or per model axis. Returns the
    empty and the placed world box (lo, hi)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(ASSETS / "objaverse" / f"{stem}.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        COLL.objects.link(o)
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in new if o.type == "MESH" for c in o.bound_box]
    lo = Vector((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)))
    hi = Vector((max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))
    sc = (scale,) * 3 if isinstance(scale, (int, float)) else scale
    root = _link(bpy.data.objects.new(stem, None))
    for o in new:
        if o.parent is None:
            o.parent = root
    root.matrix_world = (
        Matrix.Translation(at)
        @ Matrix.Rotation(yaw - front, 4, "Z")
        @ Matrix.Diagonal((*sc, 1.0))
        @ Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    )
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in new if o.type == "MESH" for c in o.bound_box]
    wlo = Vector((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)))
    whi = Vector((max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))
    return root, (wlo, whi), (hi - lo)


def build_desk_black(o, M, wall_x0):
    """His black standing desk (sourced model) at the twin's box, with a monitor, keyboard and
    laptop on it, all facing into the room."""
    cx, cy, _, hx, hy, hz, _ = o["obb"]
    top = 2 * hz
    back = -1 if abs((cx - hx) - wall_x0) < abs((cx + hx) - wall_x0) else 1
    front = -back
    face = 0.0 if front > 0 else math.pi
    long_y = hy >= hx
    w, d = (2 * hy, 2 * hx) if long_y else (2 * hx, 2 * hy)
    # the model is 1.60 x 0.78 x 1.10 m, long along x, front toward -y
    glb("desk", (cx, cy, 0.0), face, (w / 1.601, d / 0.775, top / 1.098), -math.pi / 2)
    _, (mlo, mhi), _ = glb(
        "monitor", (cx + back * (hx - 0.17), cy - 0.15, top), face, 0.0125, -math.pi / 2
    )
    # a lit screen just proud of the panel, so the monitor reads as one from above
    fx = mhi.x if front > 0 else mlo.x
    box(
        "mon_screen",
        (fx + front * 0.004, cy - 0.15, mlo.z + 0.62 * (mhi.z - mlo.z)),
        (0.0005, 0.29, 0.155),
        M["screen"],
        bevel=0,
    )
    glb("keyboard", (cx + front * 0.1, cy - 0.15, top), face, 0.0095, -math.pi / 2)
    glb("laptop", (cx + front * 0.02, cy + 0.42, top), face + 0.25 * front, 0.011, -math.pi / 2)


def build_laptop(o, M, z_on=None, open_=True):
    cx, cy, cz, hx, hy, hz, yaw = o["obb"]
    w, d = 0.31, 0.22
    z = z_on if z_on is not None else max(cz - hz, 0) + 0.008
    base = box(
        f"{o['name']}_base", (cx, cy, z), (d / 2, w / 2, 0.007), M["laptop"], yaw=yaw, bevel=0.004
    )
    _ = base
    if open_:
        lx, ly = rot2(-d / 2, 0, yaw)
        lid = box(
            f"{o['name']}_lid",
            (cx + lx, cy + ly, z + d / 2 * 0.95),
            (0.005, w / 2, d / 2),
            M["laptop"],
            yaw=yaw,
            bevel=0.004,
        )
        lid.rotation_euler = Euler((0, math.radians(-15), yaw), "ZYX")
        sx, sy = rot2(0.006, 0, yaw)
        scr = box(
            f"{o['name']}_scr",
            (cx + lx + sx, cy + ly + sy, z + d / 2 * 0.95),
            (0.0005, w / 2 - 0.01, d / 2 - 0.012),
            M["screen"],
            yaw=yaw,
            bevel=0,
        )
        scr.rotation_euler = Euler((0, math.radians(-15), yaw), "ZYX")


def build_folding_table(o, M):
    """The white folding table under the whiteboard, with his robot arms on it."""
    cx, cy, _, hx, hy, hz, _ = o["obb"]
    top = 2 * hz + 0.1
    box("table_top", (cx, cy, top - 0.02), (hx, hy, 0.02), M["white_plastic"], bevel=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(
                "table_leg",
                cx + sx * (hx - 0.06),
                cy + sy * (hy - 0.05),
                0.0,
                top - 0.04,
                0.013,
                M["grey_metal"],
            )
        box(
            "table_rail",
            (cx + sx * (hx - 0.06), cy, 0.12),
            (0.01, hy - 0.05, 0.01),
            M["grey_metal"],
        )
    for k, ax in enumerate((-0.45, 0.05, 0.42)):  # SO-101 arms: base, shoulder, two links
        bx, by = cx + ax, cy - 0.05 + 0.04 * (k % 2)
        cyl("arm_base", bx, by, top, top + 0.04, 0.05, M["black"])
        box("arm_l1", (bx, by, top + 0.13), (0.022, 0.022, 0.09), M["black"], bevel=0.006)
        box(
            "arm_l2",
            (bx + 0.07, by, top + 0.24),
            (0.08, 0.02, 0.02),
            M["white_plastic"],
            bevel=0.006,
        )
        box("arm_grip", (bx + 0.15, by, top + 0.2), (0.012, 0.02, 0.05), M["black"], bevel=0.004)
    box("table_box", (cx + 0.25, cy + 0.12, top + 0.04), (0.08, 0.06, 0.04), M["cardboard"])


def build_whiteboard(f, M):
    cx, cy, cz = f["centre"]
    sx, sy, sz = f["size"]
    box("wb_frame", (cx, cy - 0.025, cz), (sx / 2, 0.012, sz / 2), M["metal"], bevel=0.004)
    box("wb", (cx, cy - 0.038, cz), (sx / 2 - 0.015, 0.002, sz / 2 - 0.015), M["board"], bevel=0)


def build_office_chair(o, M):
    """His black mesh task chair (sourced model), seat facing away from the twin's back box.
    A touch larger and a shade lighter than true black so it reads from above: it is the
    hero object of the find-the-chair scenes."""
    cx, cy = o["obb"][0], o["obb"][1]
    backs = [b for b in o["boxes"] if b[5] > 0.15 and b[2] > 0.5]
    if backs:
        bx, by = backs[0][0] - cx, backs[0][1] - cy
        face = math.atan2(-by, -bx)
    else:
        face = 0.0
    root, _, _ = glb("chair", (cx, cy, 0.0), face, 0.0105, math.pi)  # model is in cm, faces -x
    for ch in root.children_recursive:
        for sl in getattr(ch, "material_slots", []):
            m = sl.material
            b = m and m.node_tree and _bsdf(m)
            if b and not b.inputs["Base Color"].is_linked:
                b.inputs["Base Color"].default_value = (0.045, 0.047, 0.052, 1.0)
                b.inputs["Roughness"].default_value = 0.42


def build_nightstand(o, M):
    cx, cy, _, hx, hy, hz, yaw = o["obb"]
    box("ns_body", (cx, cy, hz + 0.04), (hx, hy, hz - 0.04), M["wood_dark"], yaw=yaw, bevel=0.006)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = rot2(sx * (hx - 0.03), sy * (hy - 0.03), yaw)
            cyl("ns_leg", cx + lx, cy + ly, 0.0, 0.05, 0.012, M["black"])


def build_tower_fan(o, M):
    cx, cy, _, hx, hy, hz, yaw = o["obb"]
    h = 2 * hz + 0.5  # his tower fan stands taller than the scan caught
    box(
        "fan_body",
        (cx, cy, h / 2 + 0.03),
        (0.07, 0.07, h / 2),
        M["black"],
        yaw=yaw,
        bevel=0.05,
        seg=6,
    )
    cyl("fan_foot", cx, cy, 0.0, 0.03, 0.15, M["black"])


def build_clutter(o, M, k):
    cx, cy, cz, hx, hy, hz, yaw = o["obb"]
    hz = max(hz, 0.03)
    pal = [M["cloth_grey"], M["cloth_dark"], M["cloth_white"]]
    if o["name"] == "box":  # the shoebox in the aisle
        box("shoebox", (cx, cy, 0.06), (hx, hy, 0.06), M["cardboard"], yaw=yaw, bevel=0.004)
        box(
            "shoebox_lid",
            (cx, cy, 0.122),
            (hx + 0.005, hy + 0.005, 0.01),
            M["orange"],
            yaw=yaw,
            bevel=0.003,
        )
        return
    if hz > 0.2:  # a tall heap is his backpack
        blob(
            f"{o['name']}",
            (cx, cy, hz * 0.9),
            (max(hx, 0.15), max(hy, 0.12), hz * 0.9),
            M["cloth_dark"],
            seed=k,
            strength=0.08,
            yaw=yaw,
        )
        return
    blob(
        f"{o['name']}",
        (cx, cy, min(hz, 0.04) * 0.5),
        (max(hx, 0.1), max(hy, 0.08), min(hz, 0.04)),
        pal[k % len(pal)],
        seed=k,
        strength=0.05,
        yaw=yaw,
    )


def support_z(sc, o):
    """Top of what a small thing rests on: the duvet on the bed, the desktop on a desk."""
    cx, cy = o["obb"][0], o["obb"][1]
    best = 0.0
    for p in sc["objects"]:
        if p is o or p["label"] not in ("bed", "desk"):
            continue
        px, py, _, phx, phy, phz, _ = p["obb"]
        if abs(cx - px) <= phx and abs(cy - py) <= phy:
            top = 0.66 if p["label"] == "bed" else 2 * phz
            if p["label"] == "desk" and phz * 2 <= 0.75:
                top = 2 * phz + 0.1
            best = max(best, top)
    return best + 0.008


def build_furniture(sc, M, bounds):
    x0, x1, y0, y1 = bounds
    k = 0
    for o in sc["objects"]:
        lab, name = o["label"], o["name"]
        if lab == "bed":
            build_bed(o, M, y0)
        elif lab == "desk" and o["obb"][5] * 2 > 0.75:
            build_desk_black(o, M, x0)
        elif lab == "desk":
            build_folding_table(o, M)
        elif lab == "chair":
            build_office_chair(o, M)
        elif lab == "nightstand":
            build_nightstand(o, M)
        elif lab == "object" and name.startswith("desk_"):
            build_tower_fan(o, M)
        elif lab == "laptop":
            build_laptop(o, M, z_on=support_z(sc, o))
        else:
            build_clutter(o, M, k)
            k += 1
    for f in sc["flush"]:
        if f["cls"] == "whiteboard":
            build_whiteboard(f, M)
    bed = next((o for o in sc["objects"] if o["label"] == "bed"), None)
    if bed is not None:  # rug at the foot of the bed
        bx, by, _, bhx, bhy, _, _ = bed["obb"]
        r = 0.72
        ry = by + bhy + r - 0.25
        me = bpy.data.meshes.new("rug")
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, segments=96, radius=1.0)
        bm.to_mesh(me)
        bm.free()
        rug = _link(bpy.data.objects.new("rug", me))
        rug.scale = (r, r, 1)
        rug.location = (bx - 0.1, min(ry, y1 - r - 0.05), 0.006)
        me.materials.append(M["rug"])
        so = rug.modifiers.new("thick", "SOLIDIFY")
        so.thickness = 0.012


# ---------------------------------------------------------------- the walker


class Walker:
    """The CC0 Quaternius 'Man' with its walk and sit cycles, scaled to 1.75 m, plus a
    blindfold, glasses and the belt with one lit pod per motor."""

    HEIGHT = 1.75
    STRIDE_M = 1.25  # metres covered by one walk cycle

    def __init__(self, motor_deg, M):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(ASSETS / "human" / "HMnuH5geEG.glb"))
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            if o.name.startswith("Icosphere"):
                bpy.data.objects.remove(o)
        self.arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        self.mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == self.arm)
        for o in list(self.arm.users_collection):
            o.objects.unlink(self.arm)
        COLL.objects.link(self.arm)
        for ch in self.arm.children_recursive:
            for c in list(ch.users_collection):
                c.objects.unlink(ch)
            COLL.objects.link(ch)
        zs = [(self.mesh.matrix_world @ v.co).z for v in self.mesh.data.vertices]
        self.k = self.HEIGHT / (max(zs) - min(zs))
        self.root = _link(bpy.data.objects.new("walker", None))
        self.arm.parent = self.root
        self.arm.scale = tuple(s * self.k for s in self.arm.scale)
        self.restyle()
        self.walk = bpy.data.actions.get("HumanArmature|Man_Walk")
        self.sit = bpy.data.actions.get("HumanArmature|Man_Sitting")
        self.idle = bpy.data.actions.get("HumanArmature|Man_Idle")
        self.arm.animation_data_create()
        self.arm.animation_data.action = self.walk
        self.props(motor_deg, M)

    def restyle(self):
        cols = {
            "Shirt": ((0.42, 0.43, 0.44), 0.85),
            "Pants": ((0.05, 0.05, 0.06), 0.8),
            "Skin": ((0.42, 0.27, 0.18), 0.55),
            "Hair": ((0.03, 0.025, 0.02), 0.7),
            "Socks": ((0.85, 0.85, 0.83), 0.8),
        }
        for slot in self.mesh.material_slots:
            m = slot.material
            if m is None:
                continue
            key = next((k for k in cols if m.name.startswith(k)), None)
            if key is None or not m.use_nodes:
                continue
            b = _bsdf(m)
            b.inputs["Base Color"].default_value = (*cols[key][0], 1)
            b.inputs["Roughness"].default_value = cols[key][1]
            b.inputs["Sheen Weight"].default_value = 0.3 if key in ("Shirt", "Pants") else 0.0
        for slot in self.mesh.material_slots:
            if slot.material is not None:
                slot.material.use_backface_culling = False
        bpy.context.view_layer.objects.active = self.mesh
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        # the export splits every face apart (flat shading); weld it so subdivision rounds the
        # body instead of each face
        bm = bmesh.new()
        bm.from_mesh(self.mesh.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
        bm.to_mesh(self.mesh.data)
        bm.free()
        sd = self.mesh.modifiers.new("smooth", "SUBSURF")
        sd.levels = sd.render_levels = 1
        for p in self.mesh.data.polygons:
            p.use_smooth = True

    def _bone_world(self, name):
        bpy.context.view_layer.update()
        pb = self.arm.pose.bones[name]
        return self.arm.matrix_world @ pb.head, self.arm.matrix_world @ pb.tail

    def _attach(self, o, bone):
        """Parent `o` to a bone keeping its current world transform."""
        bpy.context.view_layer.update()
        mw = o.matrix_world.copy()
        o.parent = self.arm
        o.parent_type = "BONE"
        o.parent_bone = bone
        bpy.context.view_layer.update()
        o.matrix_world = mw

    def props(self, motor_deg, M):
        """Blindfold, glasses and belt, sized from the rest pose and parented to the bones."""
        self.arm.animation_data.action = None
        for pb in self.arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        hh, ht = self._bone_world("Head")
        eye_z = hh.z + 0.42 * (ht.z - hh.z)
        vs = [self.mesh.matrix_world @ v.co for v in self.mesh.data.vertices]
        ring = [v for v in vs if abs(v.z - eye_z) < 0.02]
        hx = max(abs(v.x - hh.x) for v in ring)
        hy_f = max(v.y - hh.y for v in ring)
        hy_b = min(v.y - hh.y for v in ring)
        cy = hh.y + (hy_f + hy_b) / 2
        ry = (hy_f - hy_b) / 2
        self.front = -1 if abs(hy_b) > abs(hy_f) else 1  # which y the face looks along
        band = self._ring(
            "blindfold", (hh.x, cy, eye_z), hx + 0.006, ry + 0.006, 0.022, M["cloth_dark"]
        )
        self._attach(band, "Head")
        gy = cy + self.front * (ry + 0.014)
        g = box(
            "glasses", (hh.x, gy, eye_z + 0.012), (hx * 0.92, 0.008, 0.02), M["black"], bevel=0.006
        )
        for s in (-1, 1):
            box(
                "temple",
                (hh.x + s * hx * 0.95, cy, eye_z + 0.02),
                (0.004, ry, 0.004),
                M["black"],
                bevel=0,
            )
        cam = box(
            "glasses_cam",
            (hh.x + hx * 0.75, gy + self.front * 0.008, eye_z + 0.02),
            (0.008, 0.006, 0.008),
            M["metal"],
            bevel=0.002,
        )
        for o in [g, cam] + [o for o in bpy.data.objects if o.name.startswith("temple")]:
            self._attach(o, "Head")
        # belt at the navel
        bh, bt = self._bone_world("Abdomen")
        wz = bh.z + 0.15 * (bt.z - bh.z)
        ring = [v for v in vs if abs(v.z - wz) < 0.02 and abs(v.x - bh.x) < 0.3]
        wx = max(abs(v.x - bh.x) for v in ring)
        wyf = max(v.y - bh.y for v in ring)
        wyb = min(v.y - bh.y for v in ring)
        wcy = bh.y + (wyf + wyb) / 2
        wry = (wyf - wyb) / 2
        belt = self._ring("belt", (bh.x, wcy, wz), wx + 0.012, wry + 0.012, 0.045, M["cloth_dark"])
        self._attach(belt, "Abdomen")
        self.pods = []
        pod_m = bpy.data.materials.new("pod")
        pod_m.use_nodes = True
        nt = pod_m.node_tree
        b = _bsdf(pod_m)
        b.inputs["Base Color"].default_value = (0.08, 0.08, 0.09, 1)
        info = nt.nodes.new("ShaderNodeObjectInfo")
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(info.outputs["Color"], sep.inputs["Color"])
        b.inputs["Emission Color"].default_value = (1.0, 0.42, 0.12, 1)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 8.0
        nt.links.new(sep.outputs["Red"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], b.inputs["Emission Strength"])
        for k, deg in enumerate(motor_deg):
            # motor bearings are clockwise from ahead; ahead is self.front along y here
            a = math.radians(-deg)
            fx, fy = rot2(0.0, self.front * 1.0, a)
            px, py = bh.x + fx * (wx + 0.03), wcy + fy * (wry + 0.03)
            p = box(f"pod{k}", (px, py, wz), (0.022, 0.022, 0.026), pod_m, bevel=0.01)
            p.rotation_euler = (0, 0, math.atan2(fy, fx))
            p.color = (0, 0, 0, 1)
            self._attach(p, "Abdomen")
            self.pods.append(p)
        self.arm.animation_data.action = self.walk

    def _ring(self, name, c, rx, ry, h, m):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=48, radius1=1.0, radius2=1.0, depth=1.0)
        bm.to_mesh(me)
        bm.free()
        me.transform(Matrix.Diagonal((rx, ry, h, 1.0)))  # baked in: bone parents can't shear
        o = _link(bpy.data.objects.new(name, me))
        o.location = c
        me.materials.append(m)
        so = o.modifiers.new("thick", "SOLIDIFY")
        so.thickness = 0.008
        so.offset = 1.0
        for p in me.polygons:
            p.use_smooth = True
        return o

    def pose(self, fr, dist_m, frame_no):
        """Place the walker for one video frame, keyframed so a whole clip renders in one go."""
        sit = fr["sit"]
        yaw = fr["heading"] - (math.pi / 2 if self.front < 0 else -math.pi / 2)
        self.root.location = (fr["x"], fr["y"], 0.0)
        self.root.rotation_euler = (0, 0, yaw)
        self.root.keyframe_insert("location", frame=frame_no)
        self.root.keyframe_insert("rotation_euler", frame=frame_no)
        ad = self.arm.animation_data
        if sit > 0 and self.sit is not None:
            ad.action = self.sit
            t = min(1.0, sit) * 60.0  # the sitting-down part of the cycle
        elif not fr["walking"] and self.idle is not None:
            ad.action = self.idle
            t = frame_no % 100
        else:
            ad.action = self.walk
            t = (dist_m / self.STRIDE_M * 25.0) % 25.0
        for pb in self.arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        _apply_action(self.arm, ad.action, t)
        for pb in self.arm.pose.bones:
            for path in ("location", "rotation_quaternion", "scale"):
                pb.keyframe_insert(path, frame=frame_no)
        for k, p in enumerate(self.pods):
            v = fr["felt"][k] if k < len(fr["felt"]) else 0.0
            p.color = (v, 0, 0, 1)
            p.keyframe_insert("color", frame=frame_no)


def _apply_action(arm, action, t):
    """Evaluate `action` at time `t` onto the pose bones (Blender 4.4+ layered actions)."""
    slot = action.slots[0] if hasattr(action, "slots") and action.slots else None
    curves = []
    if slot is not None and action.layers:
        for layer in action.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot)
                if cb is not None:
                    curves += list(cb.fcurves)
    else:
        curves = list(action.fcurves)
    for fc in curves:
        dp = fc.data_path
        if not dp.startswith("pose.bones["):
            continue
        bone = dp.split('"')[1]
        prop = dp.rsplit(".", 1)[1]
        pb = arm.pose.bones.get(bone)
        if pb is None:
            continue
        getattr(pb, prop)[fc.array_index] = fc.evaluate(t)


# ---------------------------------------------------------------- camera, light, render


def setup_camera(bounds, az_deg=-18.0, el_deg=66.0, lens=32.0, focus=None):
    x0, x1, y0, y1 = bounds
    target = Vector(((x0 + x1) / 2, (y0 + y1) / 2 - 0.1, 0.35))
    if focus is not None:  # a close look at the walker, for checking the props
        x0, x1, y0, y1 = focus[0] - 0.5, focus[0] + 0.5, focus[1] - 0.5, focus[1] + 0.5
        target = Vector((focus[0], focus[1], 1.0))
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = lens
    cam = _link(bpy.data.objects.new("cam", cam_d))
    bpy.context.scene.camera = cam
    top = 1.8 if focus is not None else WALL_H
    corners = [Vector((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (0.0, top)]
    from bpy_extras.object_utils import world_to_camera_view

    az, el = math.radians(az_deg), math.radians(el_deg)
    direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    dist = 4.0
    for _ in range(40):
        cam.location = target + direction * dist
        cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        pts = [world_to_camera_view(bpy.context.scene, cam, p) for p in corners]
        span = max(max(abs(p.x - 0.5), abs(p.y - 0.5)) for p in pts)
        if abs(span - 0.485) < 0.003:
            break
        dist *= 1 + (span - 0.485)
    return cam


def setup_light(bounds):
    x0, x1, y0, y1 = bounds
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.78, 0.8, 0.84, 1)
    bg.inputs["Strength"].default_value = 0.3
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def light(name, kind, loc, rot, energy, col, **kw):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = col
        for k, v in kw.items():
            setattr(ld, k, v)
        o = _link(bpy.data.objects.new(name, ld))
        o.location = loc
        o.rotation_euler = rot
        return o

    # warm ceiling fixture, a soft daylight through the window, and a low sun for shape
    light(
        "ceiling",
        "AREA",
        (cx, cy, WALL_H - 0.05),
        (0, 0, 0),
        230,
        (1.0, 0.86, 0.72),
        shape="DISK",
        size=1.6,
    )
    light(
        "window",
        "AREA",
        (3.4, y0 + 0.25, 1.6),
        (math.radians(72), 0, 0),
        110,
        (0.86, 0.92, 1.0),
        shape="RECTANGLE",
        size=1.2,
        size_y=1.3,
    )
    light(
        "sun",
        "SUN",
        (0, 0, 5),
        (math.radians(38), math.radians(-22), math.radians(-30)),
        1.5,
        (1.0, 0.95, 0.88),
        angle=math.radians(6),
    )


def setup_render(samples: int, scale: int):
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    ee = sc.eevee
    ee.taa_render_samples = samples
    for attr, val in (
        ("use_raytracing", RAYTRACE),
        ("use_shadows", True),
        ("shadow_ray_count", 2),
        ("shadow_step_count", 8),
        ("use_fast_gi", True),
        ("fast_gi_distance", 1.0),
    ):
        if hasattr(ee, attr):
            setattr(ee, attr, val)
    if hasattr(ee, "ray_tracing_options"):
        ee.ray_tracing_options.resolution_scale = "1"
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = scale
    sc.render.fps = 30
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.view_settings.exposure = -0.1


def write_cam(cam, path: Path):
    """World point -> pixel: P = K [R|t] as a 3x4 row-major matrix for the overlay pass."""
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    mv = cam.matrix_world.inverted()
    proj = cam.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=W, y=H, scale_x=1, scale_y=1
    )
    full = proj @ mv
    rows = [[full[i][j] for j in range(4)] for i in range(4)]
    path.write_text(json.dumps({"w": W, "h": H, "ndc_from_world": rows, "fps": sc.render.fps}))


def main():
    global COLL
    a = args()
    sc = json.loads(Path(a["scene"]).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.system.use_gpu_subdivision = False  # Metal GPU subdiv speckles
    COLL = bpy.context.scene.collection
    M = build_materials()
    bounds = room_bounds(sc["walls"])
    cut = ("S", "E")
    build_room(sc, M, cut)
    build_furniture(sc, M, bounds)
    walker = Walker(sc["motor_deg"], M)
    frames = sc["frames"]
    dist = 0.0
    for k, fr in enumerate(frames):
        if k:
            dist += math.hypot(fr["x"] - frames[k - 1]["x"], fr["y"] - frames[k - 1]["y"])
        fr["_dist"] = dist
    setup_light(bounds)
    focus = None
    if a.get("view") == "walker" and a["still"] is not None:
        f = frames[int(a["still"])]
        focus = (f["x"], f["y"])
    cam = setup_camera(bounds, focus=focus)
    setup_render(int(a["samples"]), int(a["scale"]))
    scn = bpy.context.scene
    if a["still"] is not None:
        k = int(a["still"])
        walker.pose(frames[k], frames[k]["_dist"], 1)
        scn.frame_set(1)
        out = Path(a["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        scn.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        write_cam(cam, out.with_suffix(".cam.json"))
        if a.get("blend"):
            bpy.ops.wm.save_as_mainfile(filepath=a["blend"])
        return
    d = Path(a["frames"])
    d.mkdir(parents=True, exist_ok=True)
    for k, fr in enumerate(frames):
        walker.pose(fr, fr["_dist"], k + 1)
    scn.frame_start, scn.frame_end = 1, len(frames)
    scn.render.filepath = str(d / "#####")
    write_cam(cam, d / "cam.json")
    bpy.ops.render.render(animation=True)


main()
