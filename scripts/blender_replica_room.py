"""Build the photo replica of Pranav's bedroom as data/rooms/bedroom/replica/room.blend.

    blender -b --factory-startup --python scripts/blender_replica_room.py

Everything sits in the sim frame blender_sim_room.py renders in (metres, z up):
    X = mesh x + 1.6,  Y = -mesh z + 1.5,  Z = mesh y - floor_y
so a walker posed by the sim lands in the right place without a transform.

Dimensions: walls, doors and ceiling from data/rooms/bedroom/floorplan.json; furniture at the
twin boxes in recon-v2/v2.0/objects.json (the set the sim walks; v2.1 splits the standing desk
in two). Look: his footage (data/walks), textures cut from it by scripts/replica_textures.py.
Assets and licences: data/rooms/bedroom/replica/PROVENANCE.md.

Collections (link or append them by name from room.blend):
    replica_architecture  floor, walls, trim, doors, window, outlets
    replica_ceiling       ceiling and its fittings (hide for the dollhouse view)
    replica_furniture     bed, standing desk, folding table, built-in, chair, small furniture
    replica_props         screens, keyboard, cables, posters, clothes, clutter
    replica_lights        sun, window portal, downlights
Scene "replica" renders with Cycles; scene "replica_eevee" links the same collections for Eevee.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
ROOM = ROOT / "data" / "rooms" / "bedroom"
OUT = ROOM / "replica"
TEX = OUT / "tex"
HIFI = ROOT / "data" / "assets" / "sim-hifi"
MATCH_FRAME = ("20261028-135040-iphone", "1793220651337-204.jpg")  # the compare frame

OX, OY = 1.6, 1.5  # metasight.simwalk.OFFSET
FP = json.loads((ROOM / "floorplan.json").read_text())
OBJ = {
    o["id"]: o
    for o in json.loads((ROOM / "recon-v2" / "v2.0" / "objects.json").read_text())["objects"]
}
FLOOR_Y = FP["floor_y"]
X0, X1 = FP["room_xz"]["x"][0] + OX, FP["room_xz"]["x"][1] + OX
Y0, Y1 = -FP["room_xz"]["z"][1] + OY, -FP["room_xz"]["z"][0] + OY
CEIL = FP["ceiling_height_m"]
WALL_T = 0.12
DOOR_H = 2.2  # the three scanned door heights (2.16-2.4) are one standard door
rnd = random.Random(11)

C: dict[str, bpy.types.Collection] = {}
M: dict[str, bpy.types.Material] = {}


def W(x, z, h=0.0):
    """Mesh-frame plan point (x, z) and height above the floor to the sim frame."""
    return Vector((x + OX, -z + OY, h))


def obb(oid):
    """A v2.0 twin box in the sim frame: centre (x, y), half sizes (hx, hy, hz), yaw."""
    b = OBJ[oid]["obb"]
    (px, _, pz), (sx, sy, sz) = b["center"], b["size"]
    return (px + OX, -pz + OY), (sx / 2, sz / 2, sy / 2), b.get("yaw_rad", 0.0)


# ---------------------------------------------------------------- materials


def _bsdf(m):
    return next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def mat(
    name,
    rgb,
    rough=0.5,
    metal=0.0,
    coat=0.0,
    sheen=0.0,
    emit=None,
    trans=0.0,
    rvar=0.0,
    bump=0.0,
    bump_scale=300.0,
):
    """Principled material; `rvar` adds smudgy roughness variation, `bump` a fine surface grain
    (paint orange peel, plastic stipple)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = _bsdf(m)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Transmission Weight"].default_value = trans
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[0], 1)
        b.inputs["Emission Strength"].default_value = emit[1]
    if rvar or bump:
        tc = nt.nodes.new("ShaderNodeTexCoord")
    if rvar:
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 6.0
        nz.inputs["Detail"].default_value = 8.0
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = max(0.02, rough - rvar)
        mr.inputs["To Max"].default_value = min(1.0, rough + rvar)
        mr.inputs["From Min"].default_value = 0.3
        mr.inputs["From Max"].default_value = 0.7
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    if bump:
        nb = nt.nodes.new("ShaderNodeTexNoise")
        nb.inputs["Scale"].default_value = bump_scale
        nb.inputs["Detail"].default_value = 4.0
        nt.links.new(tc.outputs["Object"], nb.inputs["Vector"])
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.0005
        nt.links.new(nb.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    M[name] = m
    return m


def image_mat(name, path, rough=0.45, emit=0.0, alpha=False, sheen=0.0):
    """An image on UVs: footage cut-outs, posters, screens (emit > 0 lights it)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = _bsdf(m)
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(str(path))
    t.interpolation = "Cubic"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Sheen Weight"].default_value = sheen
    if emit:
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit
        b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
        nt.links.remove(b.inputs["Base Color"].links[0])
        b.inputs["Coat Weight"].default_value = 1.0
        b.inputs["Coat Roughness"].default_value = 0.08
    if alpha:
        nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
    M[name] = m
    return m


def polyhaven(name, stem, tint, scale_m, rough_mul=1.0, sat=1.0, val=1.0, normal=1.0):
    """A Poly Haven PBR set from data/assets/sim-hifi/tex in object space at real scale,
    pulled toward his room's colour (`tint` multiplies, sat/val adjust)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = _bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale_m,) * 3
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def img(kind, colour):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(HIFI / "tex" / f"{stem}_{kind}_2k.jpg"))
        if not colour:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = "BOX"
        n.projection_blend = 0.2
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = sat
    hsv.inputs["Value"].default_value = val
    nt.links.new(img("diff", True).outputs["Color"], hsv.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs["B"].default_value = (*tint, 1)
    nt.links.new(hsv.outputs["Color"], mix.inputs["A"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    rm = nt.nodes.new("ShaderNodeMath")
    rm.operation = "MULTIPLY"
    rm.inputs[1].default_value = rough_mul
    nt.links.new(img("rough", False).outputs["Color"], rm.inputs[0])
    nt.links.new(rm.outputs["Value"], b.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal
    nt.links.new(img("nor_gl", False).outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    M[name] = m
    return m


def rug_material():
    """His round black rug with white rings (seen beside the bed)."""
    m = mat("rug", (0.02, 0.02, 0.02), rough=0.95, sheen=1.0)
    nt = m.node_tree
    b = _bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    ring = nt.nodes.new("ShaderNodeValToRGB")
    els = ring.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.02, 0.02, 0.02, 1)
    els[1].position, els[1].color = 1.0, (0.02, 0.02, 0.02, 1)
    for p0, p1 in ((0.18, 0.26), (0.46, 0.52), (0.84, 0.9)):
        e = els.new(p0)
        e.color = (0.02, 0.02, 0.02, 1)
        e = els.new(p0 + 0.005)
        e.color = (0.75, 0.74, 0.72, 1)
        e = els.new(p1)
        e.color = (0.75, 0.74, 0.72, 1)
        e = els.new(p1 + 0.005)
        e.color = (0.02, 0.02, 0.02, 1)
    nt.links.new(ln.outputs["Value"], ring.inputs["Fac"])
    nt.links.new(ring.outputs["Color"], b.inputs["Base Color"])
    return m


def blanket_material():
    """The grey and white patterned throw bunched on his bed: a scalloped cell print."""
    m = mat("throw", (0.6, 0.6, 0.6), rough=0.9, sheen=0.8)
    nt = m.node_tree
    b = _bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.feature = "DISTANCE_TO_EDGE"
    v.inputs["Scale"].default_value = 9.0
    nt.links.new(tc.outputs["UV"], v.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.015
    cr.color_ramp.elements[0].color = (0.36, 0.36, 0.37, 1)
    cr.color_ramp.elements[1].position = 0.035
    cr.color_ramp.elements[1].color = (0.7, 0.69, 0.66, 1)
    nt.links.new(v.outputs["Distance"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], b.inputs["Base Color"])
    return m


def build_materials():
    mat("wall", (0.80, 0.80, 0.785), rough=0.72, rvar=0.06, bump=0.08, bump_scale=900)
    mat("ceiling", (0.82, 0.82, 0.81), rough=0.9, bump=0.12, bump_scale=500)
    mat("trim", (0.84, 0.84, 0.82), rough=0.3, coat=0.15, rvar=0.06, bump=0.02)
    mat("cap", (0.06, 0.06, 0.065), rough=0.8)
    polyhaven("floor", "laminate_floor_02", (0.92, 0.9, 0.87), 1.7, 0.85, sat=0.25, val=1.05)
    polyhaven("upholstery", "cotton_jersey", (0.8, 0.68, 0.52), 0.25, sat=0.0, val=1.25, normal=1.5)
    polyhaven("sheet", "cotton_jersey", (0.86, 0.75, 0.59), 0.3, sat=0.0, val=1.2)
    polyhaven("duvet", "cotton_jersey", (0.83, 0.73, 0.58), 0.35, sat=0.0, val=1.22)
    polyhaven("pillow", "cotton_jersey", (0.82, 0.73, 0.6), 0.3, sat=0.0, val=1.2)
    polyhaven("hoodie_red", "cotton_jersey", (0.55, 0.05, 0.05), 0.25, sat=0.0, val=1.0)
    polyhaven("cloth_grey", "cotton_jersey", (0.32, 0.32, 0.33), 0.3, sat=0.0, val=1.0)
    polyhaven("denim", "cotton_jersey", (0.62, 0.68, 0.74), 0.2, sat=0.0, val=1.25, normal=2.0)
    mat("desk_black", (0.012, 0.012, 0.013), rough=0.18, coat=0.4, rvar=0.12)
    mat("powder_black", (0.02, 0.02, 0.022), rough=0.45, metal=0.2, rvar=0.08)
    mat("mat_orange", (0.62, 0.2, 0.05), rough=0.8, sheen=0.6, rvar=0.06)
    mat("alu", (0.82, 0.82, 0.83), rough=0.3, metal=1.0, rvar=0.05)
    mat("alu_dark", (0.3, 0.3, 0.32), rough=0.35, metal=1.0)
    mat("chrome", (0.9, 0.9, 0.9), rough=0.12, metal=1.0)
    mat("nickel", (0.7, 0.68, 0.64), rough=0.28, metal=1.0, rvar=0.06)
    mat("plastic_black", (0.025, 0.025, 0.027), rough=0.45, rvar=0.08)
    mat("keycap", (0.06, 0.06, 0.065), rough=0.55, bump=0.05, bump_scale=2000)
    mat("plastic_white", (0.78, 0.78, 0.76), rough=0.5, bump=0.15, bump_scale=600)
    mat("table_white", (0.8, 0.79, 0.76), rough=0.55, bump=0.3, bump_scale=250)
    mat("steel_grey", (0.42, 0.42, 0.43), rough=0.4, metal=0.9)
    mat("screen_off", (0.005, 0.005, 0.006), rough=0.06, coat=1.0)
    mat("glass", (0.95, 0.97, 0.97), rough=0.0, trans=1.0)
    mat("whiteboard_frame", (0.83, 0.81, 0.76), rough=0.35, coat=0.2)
    mat("rim_orange", (0.75, 0.18, 0.02), rough=0.35, metal=0.6)
    mat("net", (0.85, 0.85, 0.85), rough=0.8, sheen=0.5)
    mat("acrylic", (0.9, 0.92, 0.93), rough=0.05, trans=0.9)
    mat("wood", (0.28, 0.16, 0.08), rough=0.6, rvar=0.1, bump=0.1, bump_scale=60)
    mat("leg_dark", (0.03, 0.025, 0.02), rough=0.5)
    mat("pink", (0.75, 0.52, 0.85), rough=0.4)
    mat("teal", (0.0, 0.25, 0.25), rough=0.3, coat=0.5)
    mat("gold", (0.85, 0.62, 0.2), rough=0.25, metal=1.0)
    mat("green_bag", (0.28, 0.4, 0.06), rough=0.6, sheen=0.3)
    mat("red_paper", (0.65, 0.05, 0.04), rough=0.5)
    mat("yellow_box", (0.8, 0.62, 0.12), rough=0.5)
    mat("cable_white", (0.82, 0.82, 0.8), rough=0.35)
    mat("cable_black", (0.02, 0.02, 0.02), rough=0.4)
    mat("fan_grille", (0.03, 0.03, 0.03), rough=0.5)
    mat("light_emit", (1, 1, 1), emit=((1.0, 0.86, 0.68), 12.0))
    mat("phone", (0.02, 0.02, 0.03), rough=0.05, coat=1.0)
    mat("speaker_fabric", (0.08, 0.08, 0.09), rough=0.95, sheen=0.5, bump=0.4, bump_scale=1500)
    mat("servo", (0.02, 0.02, 0.02), rough=0.4)
    mat("print_white", (0.85, 0.85, 0.85), rough=0.6)
    mat("blue_cap", (0.05, 0.1, 0.3), rough=0.8, sheen=0.4)
    mat("basket_white", (0.82, 0.82, 0.8), rough=0.45)
    mat("paper", (0.85, 0.84, 0.8), rough=0.8)
    rug_material()
    blanket_material()
    for n in (
        "whiteboard",
        "wood_rail",
        "switch_plate",
        "wave_print",
        "poster_collage",
        "poster_comic",
        "poster_purple",
        "poster_mountain",
        "door_poster",
        "print_a",
        "print_b",
        "print_c",
    ):
        image_mat(n, TEX / f"{n}.png", rough=0.5 if n != "whiteboard" else 0.15)
    image_mat("jersey", TEX / "jersey.png", rough=0.7, alpha=True, sheen=0.5)
    image_mat("door_garments", TEX / "door_garments.png", rough=0.85, alpha=True, sheen=0.4)
    image_mat("screen_curved", TEX / "screen_curved.png", emit=2.2)
    image_mat("screen_laptop", TEX / "screen_laptop.png", emit=1.6)


# ---------------------------------------------------------------- mesh helpers


def link(o, coll):
    C[coll].objects.link(o)
    return o


def finish(o, m, bevel=0.0, seg=3, smooth=True, angle=35.0):
    if isinstance(m, str):
        m = M[m]
    if m is not None:
        o.data.materials.append(m)
        if _bsdf(m).inputs["Transmission Weight"].default_value > 0.5:
            o.visible_shadow = False  # caustics are off: clear things must not cast shadows
    if bevel > 0:
        bv = o.modifiers.new("bevel", "BEVEL")
        bv.width = bevel
        bv.segments = seg
        bv.limit_method = "ANGLE"
        bv.harden_normals = True
        bv.miter_outer = "MITER_ARC"
    if smooth:
        for p in o.data.polygons:
            p.use_smooth = True
        o.data.set_sharp_from_angle(angle=math.radians(angle))
    return o


def box(name, coll, c, size, m, bevel=0.003, seg=3, yaw=0.0, rot=None):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    o.rotation_euler = rot if rot is not None else (0, 0, yaw)
    return finish(o, m, bevel, seg)


def cyl(name, coll, c, r, h, m, verts=32, rot=(0, 0, 0), bevel=0.0, r2=None):
    """Cylinder (or cone with r2) standing on c, along local z."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=verts, radius1=r, radius2=r if r2 is None else r2, depth=h
    )
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    o.rotation_euler = rot
    return finish(o, m, bevel, 2, angle=50)


def sphere(name, coll, c, r, m, scale=(1, 1, 1)):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    o.scale = scale
    return finish(o, m, angle=80)


def slab(name, coll, pts, thick, mw, m, bevel=0.002, seg=3):
    """A flat shape: polygon `pts` (u, v) in local XZ, `thick` along local -Y, placed by mw."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new((u, 0.0, v)) for u, v in pts]
    f = bm.faces.new(vs)
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0, -thick, 0), verts=nv)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.matrix_world = mw
    return finish(o, m, bevel, seg)


def plane(name, coll, c, w, h, m, normal="-Y", sub=0, wobble=0.0):
    """A w x h picture facing `normal` (the room side), UVs 0..1; `wobble` bows paper."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=max(1, sub), y_segments=max(1, sub), size=0.5)
    uv = bm.loops.layers.uv.new()
    for f in bm.faces:
        for lp in f.loops:
            lp[uv].uv = (lp.vert.co.x + 0.5, lp.vert.co.y + 0.5)
    for v in bm.verts:
        x, y = v.co.x, v.co.y
        v.co = Vector(
            (x * w, y * h, wobble * (math.sin(7 * x + 3 * y) * 0.5 + (abs(x) + abs(y)) ** 2))
        )
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    face = {
        "-Y": (math.pi / 2, 0, 0),
        "+Y": (math.pi / 2, 0, math.pi),
        "+X": (math.pi / 2, 0, math.pi / 2),
        "-X": (math.pi / 2, 0, -math.pi / 2),
        "+Z": (0, 0, 0),
    }[normal]
    o.rotation_euler = face  # the grid's +z is its front, its +x reads left to right
    return finish(o, m, angle=80)


def cable(name, coll, pts, r, m, res=10):
    """A smooth cable through `pts` (world)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    cu.bevel_resolution = 3
    cu.resolution_u = res
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts, strict=True):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    cu.use_fill_caps = True
    o = link(bpy.data.objects.new(name, cu), coll)
    cu.materials.append(M[m] if isinstance(m, str) else m)
    return o


def sag(a, b, drop, n=6, jitter=0.01):
    """Points of a cable hanging from a to b."""
    a, b = Vector(a), Vector(b)
    out = []
    for k in range(n + 1):
        t = k / n
        p = a.lerp(b, t)
        p.z -= drop * 4 * t * (1 - t)
        if 0 < k < n:
            p += Vector((rnd.uniform(-jitter, jitter), rnd.uniform(-jitter, jitter), 0))
        out.append(p)
    return out


def mesh_from(name, coll, verts, faces, m, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = link(bpy.data.objects.new(name, me), coll)
    return finish(o, m, smooth=smooth, angle=80)


def soft_blob(name, coll, c, half, m, seed, lumps=0.25, flat_bottom=True, sub=4):
    """A soft heap (backpack, folded hoodie): a subdivided box pushed toward an ellipsoid and
    dented with noise."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=sub, use_grid_fill=True)
    r = random.Random(seed)
    ph = [r.uniform(0, 6.28) for _ in range(6)]
    for v in bm.verts:
        d = v.co.normalized()
        s = 0.55 * v.co + 0.45 * d * 1.3
        n = 1 + lumps * (
            math.sin(3 * d.x + ph[0]) * math.sin(2 * d.y + ph[1]) * 0.5
            + 0.5 * math.sin(5 * d.z + ph[2]) * math.cos(4 * d.x + ph[3])
        )
        s *= n
        if flat_bottom and s.z < -0.85:
            s.z = -0.85
        v.co = Vector((s.x * half[0], s.y * half[1], (s.z + 0.85) * half[2] / 1.85 * 2 - half[2]))
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    sd = o.modifiers.new("smooth", "SUBSURF")
    sd.levels = sd.render_levels = 1
    return finish(o, m, angle=180)


def pillow(name, coll, c, w, d, h, m, yaw=0.0, tilt=0.0, seed=0):
    """A stuffed pillow: thin at the seams, full in the middle, slightly lumpy."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=28, y_segments=20, size=1.0)
    top = list(bm.faces)
    r = random.Random(seed)
    a, b = r.uniform(0, 6), r.uniform(0, 6)
    ext = bmesh.ops.extrude_face_region(bm, geom=top)
    bmesh.ops.reverse_faces(bm, faces=top)
    nv = {v for v in (e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert))}
    for v in bm.verts:
        x, y = v.co.x, v.co.y
        k = max(0.0, (1 - abs(x) ** 4) * (1 - abs(y) ** 4)) ** 0.5
        z = h / 2 * k * (1 + 0.08 * math.sin(4 * x + a) * math.sin(3 * y + b))
        v.co = Vector(
            (x * w / 2 * (1 - 0.05 * k), y * d / 2 * (1 - 0.06 * k), z if v in nv else -z * 0.85)
        )
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), coll)
    o.location = c
    o.rotation_euler = (tilt, 0, yaw)
    sd = o.modifiers.new("smooth", "SUBSURF")
    sd.levels = sd.render_levels = 1
    return finish(o, m, angle=180)


# ---------------------------------------------------------------- architecture


def wall_openings():
    """Openings per wall in the sim frame: (lo, hi, z0, z1, kind) along the wall's axis."""
    ops = {"N": [], "S": [], "E": [], "W": []}
    name = {"north": "N", "south": "S", "east": "E", "west": "W"}
    for w in FP["walls"]:
        side = name[w["name"]]
        (sx, sz), (ex, ez) = w["start_xz"], w["end_xz"]
        for op in w["openings"]:
            p0 = W(
                sx + (ex - sx) * op["from_m"] / w["length_m"],
                sz + (ez - sz) * op["from_m"] / w["length_m"],
            )
            p1 = W(
                sx + (ex - sx) * op["to_m"] / w["length_m"],
                sz + (ez - sz) * op["to_m"] / w["length_m"],
            )
            a, b = (p0.x, p1.x) if side in "NS" else (p0.y, p1.y)
            ops[side].append((min(a, b), max(a, b), 0.0, DOOR_H, op["source"]))
    # the built-in shelving sits in a recess left of the closet door (cabinet in the scan)
    cab = FP["objects"][[o["id"] for o in FP["objects"]].index("cabinet_23")]
    cx = cab["center"][0] + OX
    ops["N"].append((cx - 0.3, cx + 0.3, 0.0, DOOR_H, "niche"))
    # the shuttered window between the desk end and the bed, sill at 0.95 m
    ops["S"].append((1.05, 1.95, 0.95, 2.15, "window"))
    return ops


def build_walls():
    t = WALL_T
    box(
        "floor",
        "replica_architecture",
        ((X0 + X1) / 2, (Y0 + Y1) / 2, -0.01),
        (X1 - X0 + 0.02, Y1 - Y0 + 0.02, 0.02),
        "floor",
        bevel=0,
    )
    box(
        "slab",
        "replica_architecture",
        ((X0 + X1) / 2, (Y0 + Y1) / 2, -0.1),
        (X1 - X0 + 2 * t, Y1 - Y0 + 2 * t, 0.16),
        "cap",
        bevel=0.004,
    )
    ops = wall_openings()
    sides = {
        "S": (X0 - t, X1 + t, Y0 - t / 2, "x", 1),
        "N": (X0 - t, X1 + t, Y1 + t / 2, "x", -1),
        "W": (Y0, Y1, X0 - t / 2, "y", 1),
        "E": (Y0, Y1, X1 + t / 2, "y", -1),
    }
    for side, (a0, a1, fixed, axis, inward) in sides.items():
        gs = sorted(ops[side])
        pieces, cur = [], a0
        for lo, hi, z0, z1, _ in gs:
            pieces.append((cur, lo, 0.0, CEIL))
            if z0 > 0:
                pieces.append((lo, hi, 0.0, z0))
            pieces.append((lo, hi, z1, CEIL))
            cur = hi
        pieces.append((cur, a1, 0.0, CEIL))
        for k, (lo, hi, z0, z1) in enumerate(pieces):
            if hi - lo < 1e-3 or z1 - z0 < 1e-3:
                continue
            mid, half = (lo + hi) / 2, hi - lo
            c = (mid, fixed, (z0 + z1) / 2) if axis == "x" else (fixed, mid, (z0 + z1) / 2)
            s = (half, t, z1 - z0) if axis == "x" else (t, half, z1 - z0)
            box(f"wall_{side}{k}", "replica_architecture", c, s, "wall", bevel=0.001, seg=1)
            if z0 == 0.0 and z1 > 1.0:  # baseboard, stopping at the door casings
                lo2, hi2 = max(lo, X0 if axis == "x" else Y0), min(hi, X1 if axis == "x" else Y1)
                if hi2 - lo2 > 0.02:
                    m2 = (lo2 + hi2) / 2
                    off = fixed + inward * (t / 2 + 0.006)
                    bc = (m2, off, 0.045) if axis == "x" else (off, m2, 0.045)
                    bs = (hi2 - lo2, 0.012, 0.09) if axis == "x" else (0.012, hi2 - lo2, 0.09)
                    box(f"base_{side}{k}", "replica_architecture", bc, bs, "trim", bevel=0.004)
        for lo, hi, z0, z1, kind in gs:
            if kind == "window":
                build_window(lo, hi, z0, z1)
            else:
                build_door_frame(side, axis, fixed, inward, lo, hi, kind)


def wall_frame(side, along, z, out=0.0):
    """Matrix on a wall: local x runs along the wall (left to right seen from the room), local
    z up, local -y into the room; `out` moves toward the room from the wall face."""
    if side == "N":
        return Matrix.Translation((along, Y1 - out, z))
    if side == "S":
        return Matrix.Translation((along, Y0 + out, z)) @ Matrix.Rotation(math.pi, 4, "Z")
    if side == "W":
        return Matrix.Translation((X0 + out, along, z)) @ Matrix.Rotation(math.pi / 2, 4, "Z")
    return Matrix.Translation((X1 - out, along, z)) @ Matrix.Rotation(-math.pi / 2, 4, "Z")


def _u(side, along):
    """Wall-local u (left to right from the room) of a world coordinate along the wall."""
    return {"N": along, "S": -along, "W": along, "E": -along}[side]


def build_door_frame(side, axis, fixed, inward, lo, hi, kind):
    t = WALL_T
    w = hi - lo
    mid = (lo + hi) / 2
    # jamb liners through the wall thickness and the casing on the room face
    for k, (a, b) in enumerate(((lo, lo + 0.018), (hi - 0.018, hi))):
        m2 = (a + b) / 2
        c = (m2, fixed, DOOR_H / 2) if axis == "x" else (fixed, m2, DOOR_H / 2)
        s = (0.018, t, DOOR_H) if axis == "x" else (t, 0.018, DOOR_H)
        box(f"jamb_{kind}_{k}", "replica_architecture", c, s, "trim", bevel=0.001)
    c = (mid, fixed, DOOR_H - 0.009) if axis == "x" else (fixed, mid, DOOR_H - 0.009)
    s = (w, t, 0.018) if axis == "x" else (t, w, 0.018)
    box(f"jamb_{kind}_head", "replica_architecture", c, s, "trim", bevel=0.001)
    face = fixed + inward * (t / 2 + 0.008)
    cw = 0.07
    for k, a in enumerate((lo - cw / 2, hi + cw / 2)):
        c = (a, face, (DOOR_H + cw) / 2) if axis == "x" else (face, a, (DOOR_H + cw) / 2)
        s = (cw, 0.016, DOOR_H + cw) if axis == "x" else (0.016, cw, DOOR_H + cw)
        box(f"casing_{kind}_{k}", "replica_architecture", c, s, "trim", bevel=0.005, seg=4)
    c = (mid, face, DOOR_H + cw / 2) if axis == "x" else (face, mid, DOOR_H + cw / 2)
    s = (w + 2 * cw, 0.016, cw) if axis == "x" else (0.016, w + 2 * cw, cw)
    box(f"casing_{kind}_head", "replica_architecture", c, s, "trim", bevel=0.005, seg=4)
    if kind == "niche":
        build_niche(lo, hi)
        return
    # the doors: door_20 closed (bags on its hooks), door_02 closed (poster), door_21 ajar
    swing = {"door_21": math.radians(84)}.get(kind, 0.0)
    build_door(kind, side, lo, hi, swing)


def door_leaf(name, w, h, coll="replica_architecture"):
    """His two-panel arched-top door, built flat in local space: x across (0 at the hinge),
    z up, front face at y = 0, back at y = 0.035. Returns the parts."""
    parts = []
    th = 0.035
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(w, th, h), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(w / 2, th / 2, h / 2), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    leaf = link(bpy.data.objects.new(name, me), coll)
    finish(leaf, "trim", 0.003, 3)
    parts.append(leaf)
    st, rail = 0.115, 0.13
    pw = w - 2 * st
    # upper panel with an arched head, lower panel square; raised fields with sloped edges
    low_z0, low_z1 = rail + 0.02, 0.78
    up_z0, up_z1 = low_z1 + 0.13, h - rail
    arc_rise = 0.12
    up = [(st, up_z0), (st + pw, up_z0), (st + pw, up_z1 - arc_rise)]
    for k in range(1, 16):
        a = k / 16
        up.append((st + pw * (1 - a), up_z1 - arc_rise + arc_rise * math.sin(math.pi * a)))
    up.append((st, up_z1 - arc_rise))
    low = [(st, low_z0), (st + pw, low_z0), (st + pw, low_z1), (st, low_z1)]
    for face_y, sgn in ((0.0, 1), (th, -1)):
        for k, pts in enumerate((up, low)):
            # recess ring (dark line of the moulding) then the raised field inset 25 mm
            cx = sum(p[0] for p in pts) / len(pts)
            cz = sum(p[1] for p in pts) / len(pts)
            inner = [
                (
                    cx + (u - cx) * (1 - 0.05 / max(pw / 2, 0.1)),
                    cz
                    + (v - cz)
                    * (1 - 0.05 / max((up_z1 - up_z0 if k == 0 else low_z1 - low_z0) / 2, 0.1)),
                )
                for u, v in pts
            ]
            mw = Matrix.Translation((0, face_y + (0.0 if sgn > 0 else 0.0), 0))
            if sgn < 0:
                mw = Matrix.Translation((0, th, 0)) @ Matrix.Scale(-1, 4, (0, 1, 0))
            o = slab(
                f"{name}_mould{k}{sgn}",
                coll,
                pts,
                0.004,
                mw @ Matrix.Translation((0, 0.004, 0)),
                "trim",
                bevel=0.0025,
            )
            o2 = slab(
                f"{name}_field{k}{sgn}",
                coll,
                inner,
                0.009,
                mw @ Matrix.Translation((0, 0.006, 0)),
                "trim",
                bevel=0.006,
                seg=4,
            )
            parts += [o, o2]
    return parts


def build_door(kind, side, lo, hi, swing):
    w = hi - lo - 0.006
    parts = door_leaf(kind, w, DOOR_H - 0.012)
    hinge = bpy.data.objects.new(f"{kind}_hinge", None)
    link(hinge, "replica_architecture")
    for p in parts:
        p.parent = hinge
    # knob on both faces, opposite the hinge
    for y, s in ((-0.05, 1), (0.035 + 0.05, -1)):
        k = sphere(
            f"{kind}_knob{s}",
            "replica_architecture",
            (w - 0.065, y, 0.92),
            0.028,
            "nickel",
            scale=(1, 0.8, 1),
        )
        k.parent = hinge
        r = cyl(
            f"{kind}_rose{s}",
            "replica_architecture",
            (w - 0.065, y * 0.55 + (0.0 if s > 0 else 0.016), 0.92),
            0.03,
            0.012,
            "nickel",
            rot=(math.pi / 2, 0, 0),
        )
        r.parent = hinge
        st = cyl(
            f"{kind}_stem{s}",
            "replica_architecture",
            (w - 0.065, 0.0175, 0.92),
            0.009,
            0.1,
            "nickel",
            rot=(math.pi / 2, 0, 0),
        )
        st.location.y = 0.0175 + 0.05
        st.parent = hinge
    for z in (0.18, 1.05, DOOR_H - 0.3):
        hg = box(
            f"{kind}_hinge_{z:.2f}",
            "replica_architecture",
            (0.0, 0.0175, z),
            (0.006, 0.03, 0.09),
            "nickel",
            bevel=0.001,
        )
        hg.parent = hinge
    # hinge on the left end of the opening seen from the room, at the room face of the wall
    u_lo = min(_u(side, lo), _u(side, hi))
    a_hinge = lo if _u(side, lo) == u_lo else hi
    mw = wall_frame(side, a_hinge, 0.006, out=-0.01)  # leaf face 10 mm inside the casing
    hinge.matrix_world = mw @ Matrix.Rotation(-swing, 4, "Z")
    return hinge


def build_niche(lo, hi):
    """The recess with his built-in white shelving: four drawers, three shelves of caps and
    trophies (seen in the night walk and the iPhone walk)."""
    d = 0.36
    yb = Y1 + WALL_T / 2
    cx, w = (lo + hi) / 2, hi - lo
    coll = "replica_furniture"
    box("niche_back", coll, (cx, Y1 + d, DOOR_H / 2), (w, 0.02, DOOR_H), "trim", bevel=0)
    for k, x in enumerate((lo - 0.01, hi + 0.01)):
        box(
            f"niche_side{k}",
            coll,
            (x, yb + d / 2, DOOR_H / 2),
            (0.02, d + WALL_T, DOOR_H),
            "trim",
            bevel=0.001,
        )
    box("niche_head", coll, (cx, yb + d / 2, DOOR_H - 0.01), (w, d + WALL_T, 0.02), "trim", bevel=0)
    iw, idp = w - 0.03, d + WALL_T - 0.03
    ycen = Y1 + (d - WALL_T / 2) / 2 + 0.01
    # drawers, front face just inside the wall's room face
    fy = Y1 - 0.005
    for k in range(4):
        z0, z1 = 0.08 + k * 0.235, 0.08 + (k + 1) * 0.235 - 0.004
        box(
            f"drawer{k}",
            coll,
            (cx, fy + 0.01, (z0 + z1) / 2),
            (iw - 0.006, 0.02, z1 - z0),
            "trim",
            bevel=0.004,
            seg=4,
        )
        box(
            f"drawer{k}_body",
            coll,
            (cx, fy + idp / 2, (z0 + z1) / 2),
            (iw - 0.03, idp - 0.04, z1 - z0 - 0.02),
            "trim",
            bevel=0,
        )
        cyl(
            f"pull{k}",
            coll,
            (cx - 0.07, fy - 0.02, (z0 + z1) / 2 + 0.02),
            0.005,
            0.14,
            "chrome",
            rot=(0, math.pi / 2, 0),
        )
        for s in (-1, 1):
            cyl(
                f"pull{k}_post{s}",
                coll,
                (cx + s * 0.06, fy, (z0 + z1) / 2 + 0.02),
                0.004,
                0.02,
                "chrome",
                rot=(math.pi / 2, 0, 0),
            )
    box("toe_kick", coll, (cx, fy + 0.04, 0.04), (iw, 0.02, 0.08), "trim", bevel=0)
    box("counter", coll, (cx, ycen, 1.03), (iw + 0.02, idp, 0.025), "trim", bevel=0.003)
    shelves = (1.38, 1.72, 2.02)
    for k, z in enumerate(shelves):
        box(f"shelf{k}", coll, (cx, ycen + 0.02, z), (iw, idp - 0.04, 0.022), "trim", bevel=0.003)
    p = "replica_props"
    # caps on the counter and first shelf
    for k in range(3):
        for z in (1.043, 1.391):
            x = cx - 0.17 + k * 0.17
            col = ("cloth_grey", "plastic_black", "blue_cap")[(k + int(z)) % 3]
            sphere(
                f"cap_crown{k}_{z:.1f}",
                p,
                (x, ycen + 0.02, z + 0.005),
                0.085,
                col,
                scale=(1, 1.1, 0.75),
            )
            cyl(
                f"cap_brim{k}_{z:.1f}",
                p,
                (x, ycen - 0.09, z + 0.006),
                0.07,
                0.006,
                col,
                rot=(0.12, 0, 0),
            )
    # trophies on the upper shelves, a yellow box
    for k in range(4):
        x = cx - 0.2 + k * 0.13
        z = shelves[1] + 0.011
        box(f"trophy_base{k}", p, (x, ycen, z + 0.025), (0.06, 0.06, 0.05), "wood", bevel=0.003)
        cyl(f"trophy_col{k}", p, (x, ycen, z + 0.05), 0.012, 0.09 + 0.03 * (k % 2), "gold")
        sphere(
            f"trophy_fig{k}",
            p,
            (x, ycen, z + 0.17 + 0.03 * (k % 2)),
            0.025,
            "gold",
            scale=(0.8, 0.6, 1.6),
        )
    box(
        "yellow_box",
        p,
        (cx + 0.12, ycen, shelves[0] + 0.011 + 0.06),
        (0.18, 0.14, 0.12),
        "yellow_box",
        bevel=0.002,
    )
    for k in range(5):
        box(
            f"book{k}",
            p,
            (cx - 0.22 + k * 0.03, ycen + 0.03, shelves[2] + 0.011 + 0.1),
            (0.025, 0.17, 0.2 - 0.02 * (k % 3)),
            ("red_paper", "paper", "blue_cap", "cloth_grey", "yellow_box")[k],
            bevel=0.002,
        )


def build_window(lo, hi, z0, z1):
    """Shuttered window: jamb returns, a stool, the casing, glass at the outer face and two
    plantation shutter panels with their louvres tilted open (the stripes of sun on the floor
    and bed come through those)."""
    a = "replica_architecture"
    t = WALL_T
    w, h = hi - lo, z1 - z0
    cx = (lo + hi) / 2
    for k, x in enumerate((lo + 0.009, hi - 0.009)):
        box(f"win_jamb{k}", a, (x, Y0 - t / 2, (z0 + z1) / 2), (0.018, t, h), "trim", bevel=0.001)
    box("win_head", a, (cx, Y0 - t / 2, z1 - 0.009), (w, t, 0.018), "trim", bevel=0.001)
    box(
        "win_stool",
        a,
        (cx, Y0 - t / 2 + 0.03, z0 + 0.012),
        (w + 0.12, t + 0.06, 0.024),
        "trim",
        bevel=0.006,
        seg=4,
    )
    box("win_apron", a, (cx, Y0 + 0.008, z0 - 0.04), (w + 0.04, 0.016, 0.06), "trim", bevel=0.004)
    cw = 0.065
    for k, x in enumerate((lo - cw / 2, hi + cw / 2)):
        box(
            f"win_casing{k}",
            a,
            (x, Y0 + 0.008, (z0 + z1 + cw) / 2 + 0.012),
            (cw, 0.016, h + cw - 0.024),
            "trim",
            bevel=0.005,
            seg=4,
        )
    box(
        "win_casing_head",
        a,
        (cx, Y0 + 0.008, z1 + cw / 2),
        (w + 2 * cw, 0.016, cw),
        "trim",
        bevel=0.005,
        seg=4,
    )
    box("win_glass", a, (cx, Y0 - t + 0.01, (z0 + z1) / 2), (w, 0.006, h), "glass", bevel=0)
    box(
        "win_sash_mid",
        a,
        (cx, Y0 - t + 0.025, (z0 + z1) / 2),
        (w, 0.03, 0.03),
        "plastic_white",
        bevel=0.002,
    )
    # two shutter panels in the opening
    pw = w / 2 - 0.004
    for s, x_in in enumerate((lo + 0.002, cx + 0.002)):
        xc = x_in + pw / 2
        y = Y0 - 0.03
        for k, x in enumerate((x_in + 0.025, x_in + pw - 0.025)):
            box(
                f"shutter{s}_stile{k}",
                a,
                (x, y, (z0 + z1) / 2),
                (0.05, 0.03, h - 0.03),
                "trim",
                bevel=0.003,
            )
        for k, z in enumerate((z0 + 0.045, z1 - 0.045, (z0 + z1) / 2)):
            box(
                f"shutter{s}_rail{k}",
                a,
                (xc, y, z),
                (pw - 0.1, 0.03, 0.07 if k < 2 else 0.05),
                "trim",
                bevel=0.003,
            )
        n = int((h - 0.18) / 0.09)
        for k in range(n):
            z = z0 + 0.1 + (k + 0.5) * (h - 0.2) / n
            if abs(z - (z0 + z1) / 2) < 0.05:
                continue
            box(
                f"shutter{s}_louvre{k}",
                a,
                (xc, y, z),
                (pw - 0.1, 0.07, 0.011),
                "trim",
                bevel=0.004,
                seg=2,
                rot=(math.radians(-38), 0, 0),
            )
        box(
            f"shutter{s}_tiltrod",
            a,
            (xc, y - 0.04, (z0 + z1) / 2),
            (0.012, 0.01, h - 0.25),
            "trim",
            bevel=0.003,
        )


def build_wall_fittings():
    """Outlets, the switch plate under the whiteboard, the thermostat by the east door."""
    a = "replica_architecture"
    outlets = [
        ("W", 1.25, 0.32),
        ("S", 2.6, 0.32),
        ("E", 2.2, 0.32),
        ("N", 2.0, 0.32),
        ("E", 1.1, 0.32),
    ]
    for k, (side, along, z) in enumerate(outlets):
        mw = wall_frame(side, along, z, out=0.003)
        o = box(f"outlet{k}", a, (0, 0, 0), (0.07, 0.006, 0.115), "plastic_white", bevel=0.002)
        o.matrix_world = mw
        for j, dz in enumerate((-0.021, 0.021)):
            r = box(
                f"outlet{k}_rcpt{j}",
                a,
                (0, 0, 0),
                (0.035, 0.004, 0.03),
                "plastic_white",
                bevel=0.004,
            )
            r.matrix_world = mw @ Matrix.Translation((0, -0.004, dz))
            for s in (-1, 1):
                sl = box(
                    f"outlet{k}_slot{j}{s}",
                    a,
                    (0, 0, 0),
                    (0.002, 0.002, 0.007),
                    "plastic_black",
                    bevel=0,
                )
                sl.matrix_world = mw @ Matrix.Translation((s * 0.006, -0.0065, dz + 0.002))
    # the four-gang plate under the whiteboard (cut from the iPhone frame)
    p = plane("switch_plate", a, (0, 0, 0), 0.19, 0.105, M["switch_plate"], normal="-Y")
    p.matrix_world = wall_frame("N", 1.86, 1.14, out=0.004) @ Matrix.Rotation(math.pi / 2, 4, "X")
    sw = box("switch_plate_body", a, (0, 0, 0), (0.19, 0.005, 0.105), "plastic_white", bevel=0.002)
    sw.matrix_world = wall_frame("N", 1.86, 1.14, out=0.0015)
    for k in range(4):
        r = box(f"rocker{k}", a, (0, 0, 0), (0.022, 0.006, 0.06), "plastic_white", bevel=0.002)
        r.matrix_world = wall_frame("N", 1.86 + (k - 1.5) * 0.044, 1.14, out=0.006)
    t = box("thermostat", a, (0, 0, 0), (0.09, 0.02, 0.09), "plastic_white", bevel=0.008, seg=4)
    t.matrix_world = wall_frame("E", 2.75, 1.45, out=0.01)
    s2 = box("switch_east", a, (0, 0, 0), (0.075, 0.006, 0.12), "plastic_white", bevel=0.002)
    s2.matrix_world = wall_frame("E", 2.86, 1.2, out=0.003)
    r = box("switch_east_rocker", a, (0, 0, 0), (0.035, 0.007, 0.065), "plastic_white", bevel=0.003)
    r.matrix_world = wall_frame("E", 2.86, 1.2, out=0.006)


def build_ceiling():
    c = "replica_ceiling"
    box(
        "ceiling",
        c,
        ((X0 + X1) / 2, (Y0 + Y1) / 2, CEIL + 0.01),
        (X1 - X0 + 0.24, Y1 - Y0 + 0.24, 0.02),
        "ceiling",
        bevel=0,
    )
    for k, (x, y) in enumerate(((2.05, 2.55), (3.45, 1.75))):
        cyl(
            f"downlight{k}_trim",
            c,
            (x, y, CEIL - 0.006),
            0.085,
            0.006,
            "plastic_white",
            bevel=0.002,
            verts=48,
        )
        cyl(f"downlight{k}_lens", c, (x, y, CEIL - 0.0065), 0.062, 0.002, "light_emit", verts=48)
    vx, vy = 3.6, 3.55
    box("vent_frame", c, (vx, vy, CEIL - 0.006), (0.36, 0.2, 0.012), "plastic_white", bevel=0.003)
    for k in range(9):
        box(
            f"vent_slat{k}",
            c,
            (vx - 0.14 + k * 0.035, vy, CEIL - 0.012),
            (0.004, 0.16, 0.016),
            "plastic_white",
            bevel=0.001,
            rot=(0, math.radians(30), 0),
        )
    cyl(
        "smoke_alarm",
        c,
        (2.7, 3.2, CEIL - 0.035),
        0.065,
        0.035,
        "plastic_white",
        bevel=0.01,
        verts=48,
    )


# ---------------------------------------------------------------- furniture


def build_bed():
    """Beige upholstered platform bed, headboard on the south wall, fitted sheet, a duvet and
    the grey patterned throw left in a heap (cloth simulated onto the mattress)."""
    f = "replica_furniture"
    (cx, cy), (hx, hy, hz), _ = obb("bed_00")
    y_head = Y0 + 0.02
    y_foot = cy + hy
    yc = (y_head + y_foot) / 2
    L = y_foot - y_head
    rail_z0, rail_z1 = 0.11, 0.33
    box(
        "bed_rails",
        f,
        (cx, yc + 0.03, (rail_z0 + rail_z1) / 2),
        (2 * hx, L - 0.06, rail_z1 - rail_z0),
        "upholstery",
        bevel=0.03,
        seg=5,
    )
    box(
        "headboard",
        f,
        (cx, y_head + 0.05, (0.2 + 2 * hz) / 2),
        (2 * hx + 0.04, 0.1, 2 * hz - 0.2),
        "upholstery",
        bevel=0.035,
        seg=6,
    )
    # vertical channel seams on the headboard
    for k in range(1, 8):
        x = cx - hx + k * (2 * hx) / 8
        box(
            f"headboard_seam{k}",
            f,
            (x, y_head + 0.101, 0.6 + (2 * hz - 0.6) / 2),
            (0.006, 0.004, 2 * hz - 0.65),
            "upholstery",
            bevel=0.002,
        )
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(
                f"bed_leg{sx}{sy}",
                f,
                (cx + sx * (hx - 0.06), yc + sy * (L / 2 - 0.08), 0.0),
                0.022,
                0.11,
                "leg_dark",
                r2=0.018,
            )
    mz0, mz1 = 0.31, 0.53
    mw, ml = 2 * hx - 0.12, L - 0.18
    myc = y_head + 0.11 + ml / 2
    mattress = box(
        "mattress", f, (cx, myc, (mz0 + mz1) / 2), (mw, ml, mz1 - mz0), "sheet", bevel=0.05, seg=6
    )
    p = "replica_props"
    pillow(
        "pillow_left",
        p,
        (cx - 0.38, y_head + 0.3, mz1 + 0.08),
        0.68,
        0.46,
        0.17,
        "pillow",
        tilt=0.35,
        seed=1,
    )
    pillow(
        "pillow_right",
        p,
        (cx + 0.36, y_head + 0.28, mz1 + 0.085),
        0.68,
        0.46,
        0.17,
        "pillow",
        yaw=0.05,
        tilt=0.4,
        seed=2,
    )
    pillow(
        "pillow_throw",
        p,
        (cx + 0.1, y_head + 0.46, mz1 + 0.06),
        0.5,
        0.36,
        0.14,
        "throw",
        yaw=-0.2,
        tilt=0.2,
        seed=3,
    )
    drape(cx, myc, mw, ml, mz1, mattress)


def cloth_grid(name, w, length, nx, ny, z, c, wrinkle, seed):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=nx, y_segments=ny, size=0.5)
    uv = bm.loops.layers.uv.new()
    for fc in bm.faces:
        for lp in fc.loops:
            lp[uv].uv = (lp.vert.co.x + 0.5, lp.vert.co.y + 0.5)
    r = random.Random(seed)
    ph = [r.uniform(0, 6.3) for _ in range(8)]
    for v in bm.verts:
        x, y = v.co.x, v.co.y
        h = wrinkle * (
            math.sin(9 * x + ph[0]) * math.sin(7 * y + ph[1])
            + 0.6 * math.sin(17 * x + 11 * y + ph[2])
            + 0.4 * math.sin(23 * y - 5 * x + ph[3])
        )
        v.co = Vector((x * w, y * length, h))
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new(name, me), "replica_props")
    o.location = (c[0], c[1], z)
    return o


def drape(cx, myc, mw, ml, mz1, mattress):
    """Drop the duvet and the throw on the bed with Blender's cloth solver and apply them."""
    col_objs = [mattress, bpy.data.objects["bed_rails"], bpy.data.objects["headboard"]]
    for o in col_objs + [bpy.data.objects["pillow_left"], bpy.data.objects["pillow_right"]]:
        o.modifiers.new("collision", "COLLISION")
        o.collision.thickness_outer = 0.006
        o.collision.cloth_friction = 8.0
    floor = bpy.data.objects["floor"]
    floor.modifiers.new("collision", "COLLISION")
    # duvet: pushed toward the foot, half turned back, so the top of the mattress shows at the head
    duvet = cloth_grid(
        "duvet", mw + 0.55, ml * 0.78, 70, 64, mz1 + 0.12, (cx + 0.05, myc + ml * 0.16), 0.035, 4
    )
    throw = cloth_grid("throw", 1.25, 1.05, 50, 44, mz1 + 0.42, (cx - 0.2, myc + 0.1), 0.09, 9)
    throw.rotation_euler.z = 0.5
    sc = bpy.context.scene
    for o, mass, bend in ((duvet, 0.4, 0.6), (throw, 0.25, 0.3)):
        cl = o.modifiers.new("cloth", "CLOTH")
        s = cl.settings
        s.quality = 6
        s.mass = mass
        s.tension_stiffness = s.compression_stiffness = 12
        s.bending_stiffness = bend
        s.shrink_min = 0.06 if o is throw else 0.02
        cl.collision_settings.use_self_collision = o is throw
        cl.collision_settings.distance_min = 0.004
        cl.point_cache.frame_start = 1
        cl.point_cache.frame_end = 60
    sc.frame_start, sc.frame_end = 1, 60
    # the throw lands on the duvet: the duvet collides once it has settled
    for fr in range(1, 46):
        sc.frame_set(fr)
    for o in (duvet, throw):
        with bpy.context.temp_override(object=o, active_object=o):
            bpy.ops.object.modifier_apply(modifier="cloth")
    for o in col_objs + [floor, bpy.data.objects["pillow_left"], bpy.data.objects["pillow_right"]]:
        o.modifiers.remove(o.modifiers["collision"])
    sc.frame_set(1)
    for o, m, th in ((duvet, "duvet", 0.02), (throw, "throw", 0.008)):
        so = o.modifiers.new("thick", "SOLIDIFY")
        so.thickness = th
        so.offset = 0.0
        sd = o.modifiers.new("smooth", "SUBSURF")
        sd.levels = sd.render_levels = 1
        finish(o, m, angle=180)


def build_standing_desk():
    """His black standing desk on the west wall with the orange mat, curved monitor, a second
    flat monitor (off), Mac mini, speakers, keyboard, mouse, laptop, phone, pens and cables."""
    f, p = "replica_furniture", "replica_props"
    (cx, cy), (hx, hy, hz), _ = obb("desk_01")
    d = 0.76
    x0 = X0 + 0.03
    xc = x0 + d / 2
    top = 2 * hz
    L = 2 * hy
    box("desk_top", f, (xc, cy, top - 0.0125), (d, L, 0.025), "desk_black", bevel=0.008, seg=5)
    for s in (-1, 1):
        y = cy + s * (L / 2 - 0.2)
        box(
            f"desk_column_outer{s}",
            f,
            (xc, y, 0.28),
            (0.09, 0.06, 0.5),
            "powder_black",
            bevel=0.004,
        )
        box(
            f"desk_column_inner{s}",
            f,
            (xc, y, 0.5 + (top - 0.55) / 2),
            (0.075, 0.05, top - 0.55),
            "powder_black",
            bevel=0.004,
        )
        box(
            f"desk_foot{s}",
            f,
            (xc, y, 0.02),
            (d - 0.06, 0.07, 0.04),
            "powder_black",
            bevel=0.01,
            seg=4,
        )
        for e in (-1, 1):
            cyl(
                f"desk_glide{s}{e}",
                f,
                (xc + e * (d / 2 - 0.06), y, 0.0),
                0.016,
                0.006,
                "plastic_black",
            )
        box(
            f"desk_bracket{s}",
            f,
            (xc, y, top - 0.045),
            (d - 0.12, 0.05, 0.03),
            "powder_black",
            bevel=0.003,
        )
    box(
        "desk_crossbar", f, (xc, cy, top - 0.07), (0.05, L - 0.4, 0.05), "powder_black", bevel=0.004
    )
    box(
        "desk_keypad",
        f,
        (xc + d / 2 - 0.06, cy + L / 2 - 0.35, top - 0.04),
        (0.06, 0.13, 0.025),
        "plastic_black",
        bevel=0.006,
    )
    box(
        "cable_tray",
        f,
        (x0 + 0.12, cy, top - 0.12),
        (0.12, L - 0.5, 0.06),
        "powder_black",
        bevel=0.002,
    )
    z = top
    # orange mat across the front
    box(
        "desk_mat",
        p,
        (xc + 0.06, cy - 0.05, z + 0.0015),
        (0.42, 0.9, 0.003),
        "mat_orange",
        bevel=0.0012,
        seg=2,
    )
    # curved monitor (left from the chair = south) and the flat one (north)
    mon_y = cy - 0.3
    monitor("monitor_curved", (x0 + 0.2, mon_y, z), 0.71, 0.32, "screen_curved", curved=True)
    monitor(
        "monitor_flat", (x0 + 0.17, cy + 0.42, z), 0.6, 0.34, "screen_off", curved=False, yaw=-0.25
    )
    # Mac mini, speaker puck and a round speaker
    mm = box(
        "mac_mini",
        p,
        (x0 + 0.38, cy - 0.05, z + 0.025),
        (0.127, 0.127, 0.05),
        "alu",
        bevel=0.012,
        seg=5,
    )
    box(
        "mac_mini_foot",
        p,
        (mm.location.x, mm.location.y, z + 0.002),
        (0.1, 0.1, 0.004),
        "plastic_black",
        bevel=0.01,
    )
    cyl(
        "speaker_puck",
        p,
        (x0 + 0.25, cy - 0.55, z),
        0.05,
        0.042,
        "speaker_fabric",
        bevel=0.012,
        verts=48,
    )
    cyl(
        "speaker_round",
        p,
        (x0 + 0.22, cy + 0.08, z),
        0.07,
        0.11,
        "speaker_fabric",
        bevel=0.015,
        verts=48,
    )
    keyboard((xc + 0.12, cy + 0.05, z + 0.003), yaw=math.pi / 2 + 0.12)
    sphere(
        "mouse",
        p,
        (xc + 0.14, cy - 0.32, z + 0.016),
        0.035,
        "plastic_black",
        scale=(1.0, 1.7, 0.55),
    )
    box(
        "phone",
        p,
        (xc + 0.15, cy - 0.12, z + 0.0075),
        (0.072, 0.15, 0.008),
        "phone",
        bevel=0.009,
        seg=4,
        yaw=0.3,
    )
    laptop("laptop_desk", (xc + 0.06, cy + 0.6, z + 0.003), yaw=math.pi / 2 - 0.35)
    for k, (dx, dy, m) in enumerate(((0.2, -0.38, "teal"), (0.19, -0.35, "plastic_black"))):
        cyl(
            f"pen{k}",
            p,
            (xc + dx, cy + dy, z + 0.006),
            0.0055,
            0.14,
            m,
            rot=(math.pi / 2, 0, 0.5 + 0.1 * k),
        )
    pencil_cup((x0 + 0.1, cy + 0.72, z))
    box(
        "pink_case",
        p,
        (xc + 0.02, cy + 0.32, z + 0.025),
        (0.08, 0.17, 0.05),
        "pink",
        bevel=0.02,
        seg=5,
        yaw=0.2,
    )
    cyl("water_bottle", p, (x0 + 0.08, cy - 0.7, z), 0.035, 0.24, "glass", bevel=0.01)
    cyl(
        "water_bottle_cap",
        p,
        (x0 + 0.08, cy - 0.7, z + 0.24),
        0.022,
        0.03,
        "plastic_black",
        bevel=0.003,
    )
    # cables: Mac mini to the phone (white), monitor and speaker leads over the back edge
    cable(
        "cable_phone",
        p,
        [
            (mm.location.x + 0.06, mm.location.y, z + 0.012),
            (xc + 0.0, cy - 0.02, z + 0.004),
            (xc + 0.08, cy - 0.2, z + 0.004),
            (xc + 0.15, cy - 0.19, z + 0.006),
        ],
        0.0018,
        "cable_white",
    )
    for k, (a, b) in enumerate(
        (
            ((x0 + 0.2, mon_y, z + 0.05), (x0 + 0.05, mon_y + 0.2, z - 0.4)),
            ((x0 + 0.38, cy - 0.05, z + 0.02), (x0 + 0.04, cy, z - 0.5)),
            ((x0 + 0.22, cy + 0.08, z + 0.02), (x0 + 0.06, cy + 0.3, z - 0.6)),
            ((x0 + 0.17, cy + 0.42, z + 0.04), (x0 + 0.05, cy + 0.5, z - 0.45)),
        )
    ):
        mid = ((a[0] + 0.0) * 0.5 + x0 * 0.5, (a[1] + b[1]) / 2, z + 0.006)
        cable(
            f"cable_desk{k}",
            p,
            [
                a,
                mid,
                (x0 + 0.01, (a[1] + b[1]) / 2, z + 0.004),
                (x0 + 0.02, b[1], z - 0.05),
                b,
                (x0 + 0.1, b[1] + 0.1, 0.02),
                (x0 + 0.25, cy + 0.6 + 0.05 * k, 0.006),
            ],
            0.0025,
            "cable_black",
        )
    box(
        "power_strip",
        p,
        (x0 + 0.25, cy + 0.65, 0.02),
        (0.06, 0.3, 0.04),
        "plastic_white",
        bevel=0.006,
        yaw=0.1,
    )


def monitor(name, base, w, h, screen, curved, yaw=0.0):
    """A monitor on a V stand, panel facing +x (into the room)."""
    p = "replica_props"
    root = bpy.data.objects.new(name, None)
    link(root, p)
    parts = []
    parts.append(
        box(
            f"{name}_foot",
            p,
            (0.06, 0, 0.006),
            (0.2, 0.26, 0.012),
            "plastic_black",
            bevel=0.005,
            seg=3,
        )
    )
    parts.append(
        box(
            f"{name}_neck",
            p,
            (0.0, 0, 0.2),
            (0.03, 0.07, 0.38),
            "plastic_black",
            bevel=0.008,
            seg=3,
            rot=(0, math.radians(-8), 0),
        )
    )
    zc = 0.13 + h / 2
    bow = (lambda y: y * y / (2 * 1.8)) if curved else (lambda y: 0.0)
    me = bpy.data.meshes.new(f"{name}_screen")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=40 if curved else 1, y_segments=1, size=0.5)
    uv = bm.loops.layers.uv.new()
    for fc in bm.faces:
        for lp in fc.loops:
            lp[uv].uv = (0.5 - lp.vert.co.x, lp.vert.co.y + 0.5)
    for v in bm.verts:
        y = v.co.x * (w - 0.012)
        v.co = Vector((bow(y), y, v.co.y * (h - 0.008)))
    bm.to_mesh(me)
    bm.free()
    scr = link(bpy.data.objects.new(f"{name}_screen", me), p)
    scr.location = (0.0535, 0, zc)
    finish(scr, screen, angle=80)
    if me.polygons[0].normal.x < 0:  # the screen faces +x, into the room
        me.flip_normals()
    # the panel: the same bow, 6 mm bezel round the glass, 25 mm deep behind it
    pme = bpy.data.meshes.new(f"{name}_panel")
    pv = [
        (bow(y) - 0.0015, y, zz)
        for y in [(k / 40 - 0.5) * w for k in range(41)]
        for zz in (-(h + 0.012) / 2, (h + 0.012) / 2)
    ]
    pf = [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(40)]
    pme.from_pydata(pv, [], pf)
    panel = link(bpy.data.objects.new(f"{name}_panel", pme), p)
    panel.location = (0.0535, 0, zc)
    so = panel.modifiers.new("depth", "SOLIDIFY")
    so.thickness = 0.025
    so.offset = 1.0 if pme.polygons[0].normal.x < 0 else -1.0
    finish(panel, "plastic_black", 0.003, 2)
    parts += [panel, scr]
    for o in parts:
        o.parent = root
    root.location = base
    root.rotation_euler.z = yaw
    return root


def keyboard(c, yaw):
    """His 60% board: black case, dark grey caps with a spacebar, every key a bevelled cap."""
    p = "replica_props"
    root = link(bpy.data.objects.new("keyboard", None), p)
    pitch = 0.019
    rows = [
        [1] * 13 + [2],
        [1.5] + [1] * 12 + [1.5],
        [1.75] + [1] * 11 + [2.25],
        [2.25] + [1] * 10 + [2.75],
        [1.25] * 3 + [6.25] + [1.25] * 4,
    ]
    W_ = 15 * pitch
    case = box(
        "kb_case",
        p,
        (0, 0, 0.012),
        (W_ + 0.012, 5 * pitch + 0.012, 0.024),
        "plastic_black",
        bevel=0.004,
        seg=4,
    )
    case.parent = root
    caps = []
    for r, row in enumerate(rows):
        x = -W_ / 2
        y = (2 - r) * pitch
        for u in row:
            kw = u * pitch
            k = box(
                f"kb_key{r}_{len(caps)}",
                p,
                (x + kw / 2, y, 0.03),
                (kw - 0.0028, pitch - 0.0028, 0.009),
                "keycap",
                bevel=0.0018,
                seg=2,
            )
            k.parent = root
            caps.append(k)
            x += kw
    root.location = c
    root.rotation_euler = (math.radians(4), 0, yaw)
    return root


def laptop(name, c, yaw, screen="screen_laptop", open_deg=108):
    p = "replica_props"
    root = link(bpy.data.objects.new(name, None), p)
    w, d = 0.304, 0.215
    base = box(f"{name}_base", p, (0, 0, 0.0075), (w, d, 0.015), "alu", bevel=0.006, seg=4)
    deck = box(
        f"{name}_deck", p, (0, 0.035, 0.0152), (w - 0.04, 0.1, 0.0008), "plastic_black", bevel=0
    )
    pad = box(f"{name}_pad", p, (0, -0.06, 0.0152), (0.12, 0.075, 0.0008), "alu_dark", bevel=0.002)
    lid_root = link(bpy.data.objects.new(f"{name}_hinge", None), p)
    lid_root.location = (0, d / 2, 0.015)
    lid = box(f"{name}_lid", p, (0, -d / 2, 0.0035), (w, d, 0.007), "alu", bevel=0.005, seg=4)
    scr = plane(
        f"{name}_screen",
        p,
        (0, -d / 2 + 0.006, 0.0076),
        w - 0.02,
        d - 0.025,
        M[screen],
        normal="+Z",
    )
    for o in (lid, scr):
        o.parent = lid_root
    lid_root.rotation_euler.x = math.radians(180 - open_deg) + math.pi
    for o in (base, deck, pad, lid_root):
        o.parent = root
    root.location = c
    root.rotation_euler.z = yaw
    return root


def pencil_cup(c):
    p = "replica_props"
    me = bpy.data.meshes.new("pencil_cup")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=0.04, radius2=0.042, depth=0.1)
    bmesh.ops.subdivide_edges(
        bm, edges=[e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 0.05], cuts=12
    )
    bm.to_mesh(me)
    bm.free()
    o = link(bpy.data.objects.new("pencil_cup", me), p)
    o.location = (c[0], c[1], c[2] + 0.05)
    wf = o.modifiers.new("mesh", "WIREFRAME")
    wf.thickness = 0.0012
    finish(o, "plastic_black", smooth=False)
    cyl("pencil_cup_base", p, c, 0.04, 0.004, "plastic_black")
    for k in range(5):
        a = k * 1.3
        cyl(
            f"pencil{k}",
            p,
            (c[0] + 0.012 * math.cos(a), c[1] + 0.012 * math.sin(a), c[2] + 0.004),
            0.004,
            0.17,
            ("yellow_box", "plastic_black", "teal", "red_paper", "plastic_white")[k],
            rot=(0.18 * math.cos(a), 0.18 * math.sin(a), 0),
        )


def build_folding_table():
    """White folding table under the whiteboard with the two SO-101 arms, a cable heap, a
    power strip on the floor and a basketball hoop on wood rails over the whiteboard."""
    f, p = "replica_furniture", "replica_props"
    (cx, cy), (hx, hy, hz), _ = obb("desk_02")
    top = 2 * hz + 0.03
    # his is a 4 ft Lifetime table; the twin box runs long, so keep its east end
    w, d = 1.22, 2 * hy
    cx += hx - w / 2
    box("table_top", f, (cx, cy, top - 0.022), (w, d, 0.044), "table_white", bevel=0.012, seg=5)
    box(
        "table_apron",
        f,
        (cx, cy, top - 0.06),
        (w - 0.06, d - 0.06, 0.035),
        "steel_grey",
        bevel=0.003,
    )
    for sx in (-1, 1):
        xl = cx + sx * (w / 2 - 0.12)
        for sy in (-1, 1):
            yl = cy + sy * (d / 2 - 0.06)
            cyl(f"table_leg{sx}{sy}", f, (xl, yl, 0.02), 0.012, top - 0.08, "powder_black")
            cyl(f"table_foot{sx}{sy}", f, (xl, yl, 0.0), 0.016, 0.025, "plastic_black", bevel=0.004)
        cyl(
            f"table_brace{sx}",
            f,
            (xl, cy - d / 2 + 0.06, 0.18),
            0.008,
            d - 0.12,
            "steel_grey",
            rot=(-math.pi / 2, 0, 0),
        )
        box(
            f"table_hinge{sx}",
            f,
            (cx + sx * 0.18, cy, top - 0.09),
            (0.36, 0.02, 0.012),
            "steel_grey",
            bevel=0.002,
        )
    z = top
    for k, ax in enumerate((cx - 0.42, cx - 0.22)):
        so101(f"so101_{k}", (ax, cy + 0.08, z), yaw=math.radians(-90 + 25 * k), pose=k)
    # the tangle of cables, boards and adapters along the right of the table
    # chargers, hubs and servo boards heaped on the right half, cables looping over them
    for k in range(12):
        box(
            f"gadget{k}",
            p,
            (
                cx + rnd.uniform(0.0, 0.55),
                cy + rnd.uniform(-0.18, 0.18),
                z + 0.012 + 0.012 * (k % 3),
            ),
            (rnd.uniform(0.04, 0.12), rnd.uniform(0.03, 0.08), rnd.uniform(0.012, 0.03)),
            ("plastic_black", "plastic_white", "alu_dark", "plastic_black")[k % 4],
            bevel=0.004,
            yaw=rnd.uniform(-1.2, 1.2),
        )
    for k in range(26):  # each cable a random walk lying on the heap, 6-9 points
        pts = [Vector((cx + rnd.uniform(-0.05, 0.55), cy + rnd.uniform(-0.2, 0.2), z + 0.004))]
        h = rnd.uniform(0, 6.3)
        for _ in range(rnd.randint(5, 8)):
            h += rnd.uniform(-1.3, 1.3)
            st = rnd.uniform(0.03, 0.08)
            nxt = pts[-1] + Vector((math.cos(h) * st, math.sin(h) * st, 0))
            nxt.z = z + 0.004 + rnd.uniform(0.0, 0.035)
            pts.append(nxt)
        cable(
            f"tangle{k}",
            p,
            pts,
            rnd.choice((0.0018, 0.0022, 0.003)),
            "cable_black" if k % 3 else "cable_white",
        )
    for k in range(4):
        box(
            f"pcb{k}",
            p,
            (cx + 0.1 + k * 0.12, cy - 0.12 + 0.07 * (k % 2), z + 0.006),
            (0.07, 0.05, 0.012),
            ("teal", "plastic_black", "servo", "red_paper")[k],
            bevel=0.002,
            yaw=rnd.uniform(-0.5, 0.5),
        )
    box(
        "adapter_brick",
        p,
        (cx + 0.5, cy + 0.05, z + 0.016),
        (0.11, 0.05, 0.03),
        "plastic_black",
        bevel=0.006,
    )
    ps = box(
        "floor_power_strip",
        p,
        (cx - 0.1, cy + 0.05, 0.02),
        (0.32, 0.06, 0.04),
        "plastic_white",
        bevel=0.006,
        yaw=0.2,
    )
    for k in range(4):  # leads down from the table to the strip and a charger brick on the floor
        a = (cx + 0.2 + 0.08 * k, cy - d / 2 + 0.05, z)
        cable(
            f"drop{k}",
            p,
            [
                a,
                (a[0] - 0.02, a[1] - 0.04, z - 0.05),
                (a[0] - 0.05 * k, cy - 0.05, 0.35),
                (ps.location.x + 0.05 * k, ps.location.y + 0.15, 0.01),
                (ps.location.x - 0.1 + 0.07 * k, ps.location.y, 0.03),
            ],
            0.0022,
            "cable_black" if k % 2 else "cable_white",
        )
    box(
        "charger_floor",
        p,
        (cx - 0.25, cy - 0.15, 0.015),
        (0.1, 0.06, 0.03),
        "plastic_black",
        bevel=0.006,
        yaw=-0.4,
    )


def so101(name, base, yaw, pose):
    """An SO-101 arm, roughly: a printed base, five servos and links, a gripper."""
    p = "replica_props"
    root = link(bpy.data.objects.new(name, None), p)
    parts = [
        box(f"{name}_base", p, (0, 0, 0.02), (0.11, 0.09, 0.04), "servo", bevel=0.006),
        box(f"{name}_servo0", p, (0, 0, 0.06), (0.045, 0.025, 0.035), "servo", bevel=0.003),
    ]
    a1 = [0.15, 0.55][pose]
    a2 = [-0.9, -2.0][pose]
    pts = [Vector((0, 0, 0.08))]
    l1, l2 = 0.16, 0.17
    pts.append(pts[0] + Vector((math.sin(a1) * l1, 0, math.cos(a1) * l1)))
    pts.append(pts[1] + Vector((math.sin(a1 + a2) * l2, 0, math.cos(a1 + a2) * l2)))
    for k in range(2):
        a, b = pts[k], pts[k + 1]
        mid = (a + b) / 2
        ang = math.atan2(b.x - a.x, b.z - a.z)
        parts.append(
            box(
                f"{name}_link{k}",
                p,
                mid,
                (0.03, 0.035, (b - a).length),
                "servo",
                bevel=0.005,
                rot=(0, ang, 0),
            )
        )
        parts.append(
            box(
                f"{name}_servo{k + 1}",
                p,
                b,
                (0.035, 0.045, 0.025),
                "servo",
                bevel=0.003,
                rot=(0, ang, 0),
            )
        )
    tip = pts[2]
    for s in (-1, 1):
        parts.append(
            box(
                f"{name}_jaw{s}",
                p,
                tip + Vector((0.03, s * 0.012, -0.01)),
                (0.05, 0.008, 0.02),
                "servo",
                bevel=0.002,
            )
        )
    for o in parts:
        o.parent = root
    root.location = base
    root.rotation_euler.z = yaw
    return root


def build_whiteboard_and_hoop():
    p = "replica_props"
    (cx, cy), (hx, hy, hz), _ = obb("whiteboard_04")
    zc = OBJ["whiteboard_04"]["obb"]["center"][1] - FLOOR_Y
    w, h = 2 * hx, 2 * hz
    for k, (dx, dz, sw, sh) in enumerate(
        (
            (0, h / 2, w + 0.036, 0.036),
            (0, -h / 2, w + 0.036, 0.036),
            (-w / 2, 0, 0.036, h + 0.036),
            (w / 2, 0, 0.036, h + 0.036),
        )
    ):
        box(
            f"wb_frame{k}",
            p,
            (cx + dx, Y1 - 0.012, zc + dz),
            (sw, 0.016, sh),
            "whiteboard_frame",
            bevel=0.004,
        )
    plane("whiteboard", p, (cx, Y1 - 0.008, zc), w - 0.01, h - 0.01, M["whiteboard"], normal="-Y")
    box(
        "wb_tray",
        p,
        (cx, Y1 - 0.03, zc - h / 2 - 0.015),
        (w * 0.6, 0.04, 0.012),
        "whiteboard_frame",
        bevel=0.003,
    )
    cyl(
        "wb_marker",
        p,
        (cx + 0.2, Y1 - 0.035, zc - h / 2 - 0.005),
        0.007,
        0.12,
        "teal",
        rot=(0, math.pi / 2, 0),
    )
    # the hoop: two wood rails over the door frame style mount, an acrylic board, rim and net
    top_rail_z = zc + h / 2 + 0.36
    for k, z in enumerate((top_rail_z, top_rail_z - 0.16)):
        box(f"hoop_rail{k}", p, (cx, Y1 - 0.012, z), (0.62, 0.022, 0.05), "wood", bevel=0.004)
        if k == 0:
            plane(
                "hoop_rail_face", p, (cx, Y1 - 0.0235, z), 0.62, 0.05, M["wood_rail"], normal="-Y"
            )
    # clear acrylic board: white shooter's square and border lines, black corner clamps
    by, bz = Y1 - 0.038, top_rail_z - 0.08
    for k, (dx, dz, sw, sh) in enumerate(
        (
            (0, 0.13, 0.4, 0.008),
            (0, -0.13, 0.4, 0.008),
            (-0.2, 0, 0.008, 0.26),
            (0.2, 0, 0.008, 0.26),
            (0, 0.035, 0.17, 0.012),
            (-0.08, -0.03, 0.012, 0.13),
            (0.08, -0.03, 0.012, 0.13),
        )
    ):
        box(f"hoop_line{k}", p, (cx + dx, by, bz + dz), (sw, 0.001, sh), "plastic_white", bevel=0)
    for k, (dx, dz) in enumerate(((-0.19, 0.12), (0.19, 0.12), (-0.19, -0.12), (0.19, -0.12))):
        box(
            f"hoop_clamp{k}",
            p,
            (cx + dx, by - 0.004, bz + dz),
            (0.05, 0.012, 0.05),
            "plastic_black",
        )
    box(
        "hoop_board_glass",
        p,
        (cx, Y1 - 0.034, top_rail_z - 0.08),
        (0.42, 0.006, 0.27),
        "acrylic",
        bevel=0.003,
    )
    rim_z = top_rail_z - 0.18
    me = bpy.data.meshes.new("hoop_rim")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, segments=48, radius=0.085)
    bm.to_mesh(me)
    bm.free()
    rim = link(bpy.data.objects.new("hoop_rim", me), p)
    rim.location = (cx, Y1 - 0.13, rim_z)
    sk = rim.modifiers.new("tube", "SKIN")
    for v in me.skin_vertices[0].data:
        v.radius = (0.005, 0.005)
    finish(rim, "rim_orange", smooth=True, angle=180)
    sd = rim.modifiers.new("smooth", "SUBSURF")
    sd.levels = sd.render_levels = 1
    _ = sk
    box("hoop_bracket", p, (cx, Y1 - 0.05, rim_z), (0.08, 0.07, 0.02), "rim_orange", bevel=0.003)
    net = link(bpy.data.objects.new("hoop_net", bpy.data.meshes.new("hoop_net")), p)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=12, radius1=0.055, radius2=0.085, depth=0.12)
    bmesh.ops.subdivide_edges(
        bm, edges=[e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 0.05], cuts=3
    )
    for v in bm.verts:  # diamond mesh: twist each ring
        v.co.rotate(Matrix.Rotation(v.co.z * 4.0, 3, "Z"))
    bm.to_mesh(net.data)
    bm.free()
    net.location = (cx, Y1 - 0.13, rim_z - 0.06)
    wf = net.modifiers.new("cord", "WIREFRAME")
    wf.thickness = 0.003
    finish(net, "net", smooth=False)


def build_built_in_misc():
    """Laundry basket in the north-east corner, the poster on the closet door."""
    p = "replica_props"
    me = bpy.data.meshes.new("laundry_basket")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.24, radius2=0.27, depth=0.55)
    bmesh.ops.rotate(
        bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 4, 3, "Z"), verts=bm.verts
    )
    top = [f for f in bm.faces if f.normal.z > 0.9]
    bmesh.ops.delete(bm, geom=top, context="FACES")
    bm.to_mesh(me)
    bm.free()
    b = link(bpy.data.objects.new("laundry_basket", me), p)
    b.location = (X1 - 0.24, Y1 - 0.2, 0.275)
    so = b.modifiers.new("wall", "SOLIDIFY")
    so.thickness = 0.008
    bv = b.modifiers.new("bevel", "BEVEL")
    bv.width = 0.03
    bv.segments = 4
    finish(b, "basket_white")
    soft_blob("laundry", p, (X1 - 0.24, Y1 - 0.2, 0.5), (0.2, 0.18, 0.08), "cloth_grey", seed=5)
    # door poster on the closet door (door_02), slightly bowed paper
    d02 = [o for o in FP["objects"] if o["id"] == "door_02"][0]
    x = d02["center"][0] + OX
    plane(
        "door_poster",
        p,
        (x + 0.05, Y1 + 0.007, 1.55),
        0.3,
        0.45,
        M["door_poster"],
        normal="-Y",
        sub=6,
        wobble=0.003,
    )


def build_walls_art():
    """The wave print over the curved monitor, three framed prints by the coat stand, the
    poster wall on the east side and the shirt over the bed."""
    p = "replica_props"

    def on_wall(name, side, along, z, w, h, m, frame=None, sub=6, wobble=0.0025):
        mw = wall_frame(side, along, z, out=0.004 if frame is None else 0.022)
        o = plane(
            name,
            p,
            (0, 0, 0),
            w,
            h,
            M[m],
            normal="+Z",
            sub=sub,
            wobble=wobble if frame is None else 0.0,
        )
        o.matrix_world = mw @ Matrix.Rotation(math.pi / 2, 4, "X")
        if frame is not None:
            fw = 0.022
            for k, (dx, dz, sw, sh) in enumerate(
                (
                    (0, (h + fw) / 2 + 0.035, w + 0.07 + 2 * fw, fw),
                    (0, -(h + fw) / 2 - 0.035, w + 0.07 + 2 * fw, fw),
                    (-(w + fw) / 2 - 0.035, 0, fw, h + 0.07),
                    ((w + fw) / 2 + 0.035, 0, fw, h + 0.07),
                )
            ):
                fb = box(
                    f"{name}_frame{k}", p, (0, 0, 0), (sw, 0.028, sh), "plastic_black", bevel=0.003
                )
                fb.matrix_world = mw @ Matrix.Translation((dx, 0.008, dz))
            mat_ = box(
                f"{name}_mat", p, (0, 0, 0), (w + 0.07, 0.004, h + 0.07), "print_white", bevel=0
            )
            mat_.matrix_world = mw @ Matrix.Translation((0, 0.003, 0))
            gl = box(f"{name}_glass", p, (0, 0, 0), (w + 0.07, 0.002, h + 0.07), "glass", bevel=0)
            gl.matrix_world = mw @ Matrix.Translation((0, -0.006, 0))
        return o

    (dcx, dcy), _, _ = obb("desk_01")
    on_wall("wave_print", "W", dcy - 0.42, 1.62, 0.45, 0.66, "wave_print")
    for y, z, m, w, h in (
        (3.0, 1.6, "print_a", 0.24, 0.31),
        (3.44, 1.66, "print_b", 0.3, 0.23),
        (3.86, 1.6, "print_c", 0.24, 0.31),
    ):
        on_wall(f"framed_{m}", "W", y, z, w, h, m, frame=True)
    for name, y, z, w, h in (
        ("poster_mountain", 1.05, 1.62, 0.78, 0.54),
        ("poster_collage", 1.72, 1.66, 0.5, 0.42),
        ("poster_comic", 2.12, 1.66, 0.28, 0.45),
        ("poster_purple", 1.72, 1.2, 0.5, 0.4),
    ):
        on_wall(name, "E", y, z, w, h, name)
    on_wall("jersey", "S", 2.38, 1.55, 0.62, 0.62, "jersey", sub=8, wobble=0.004)
    # a small mirror under the mountain poster
    mr = box("mirror", p, (0, 0, 0), (0.28, 0.01, 0.26), "chrome", bevel=0.002)
    mr.matrix_world = wall_frame("E", 1.2, 1.12, out=0.006)


def build_small_furniture():
    f, p = "replica_furniture", "replica_props"
    # nightstand in the south-east corner
    (cx, cy), (hx, hy, hz), _ = obb("nightstand_15")
    box(
        "nightstand",
        f,
        (X1 - 0.21, Y0 + 0.24, 0.29),
        (0.4, 0.4, 0.58),
        "leg_dark",
        bevel=0.006,
        seg=3,
    )
    box(
        "nightstand_drawer",
        f,
        (X1 - 0.21, Y0 + 0.44, 0.45),
        (0.36, 0.01, 0.15),
        "leg_dark",
        bevel=0.003,
    )
    sphere("ns_knob", f, (X1 - 0.21, Y0 + 0.455, 0.45), 0.01, "chrome")
    cyl("ns_lamp_base", p, (X1 - 0.25, Y0 + 0.2, 0.58), 0.05, 0.02, "alu", bevel=0.005)
    cyl("ns_lamp_stem", p, (X1 - 0.25, Y0 + 0.2, 0.6), 0.006, 0.28, "alu")
    cyl("ns_lamp_shade", p, (X1 - 0.25, Y0 + 0.2, 0.82), 0.1, 0.16, "paper", r2=0.075)
    _ = (cx, cy, hx, hy, hz)
    # tower fan by the desk end
    (fx, fy), _, _ = obb("desk_18")
    cyl("fan_base", f, (fx, fy, 0.0), 0.15, 0.04, "plastic_black", bevel=0.01, verts=48)
    box("fan_body", f, (fx, fy, 0.56), (0.13, 0.14, 1.02), "plastic_black", bevel=0.05, seg=6)
    box("fan_grille", f, (fx + 0.07, fy, 0.52), (0.006, 0.08, 0.72), "fan_grille", bevel=0.002)
    for k in range(30):
        box(
            f"fan_slot{k}",
            f,
            (fx + 0.072, fy, 0.18 + k * 0.024),
            (0.004, 0.085, 0.008),
            "plastic_black",
            bevel=0.001,
        )
    # coat stand with the red hoodie beside the desk's north end
    sx, sy = X0 + 0.2, 2.95
    cyl("coat_pole", f, (sx, sy, 0.0), 0.016, 1.75, "wood", bevel=0.003)
    for k in range(3):
        a = k * 2.09 + 0.3
        cyl(
            f"coat_leg{k}",
            f,
            (sx, sy, 0.12),
            0.012,
            0.36,
            "wood",
            rot=(math.cos(a) * 1.15, math.sin(a) * 1.15, 0),
        )
        cyl(
            f"coat_hook{k}",
            f,
            (sx, sy, 1.62),
            0.008,
            0.16,
            "wood",
            rot=(math.cos(a + 1) * 0.9, math.sin(a + 1) * 0.9, 0),
        )
    sphere("coat_top", f, (sx, sy, 1.76), 0.022, "wood")
    garment("hoodie", (sx + 0.13, sy), 1.63, 0.66, 0.24, 0.47, "hoodie_red", seed=21)
    for k, side in enumerate((-1, 1)):  # the sleeves hang down each side, cuffs ribbed
        garment(
            f"hoodie_sleeve{k}",
            (sx + 0.15, sy + side * 0.2),
            1.47,
            0.56,
            0.06,
            0.085,
            "hoodie_red",
            seed=23 + k,
            depth=0.75,
        )
    soft_blob("hoodie_hood", p, (sx + 0.06, sy, 1.6), (0.12, 0.15, 0.11), "hoodie_red", seed=22)
    # office chair (sourced model) facing the desk, a grey throw over its back
    chair()
    # the tote, the red bag and the denim jacket on the arched door's hooks: one GrabCut card
    # cut from the iPhone frame, hung where that frame's solved camera saw it
    photo_card("door_garments", MATCH_FRAME, (0, 286, 142, 618), Y1 - 0.03)
    # floor: backpack under the desk, rug, green bag with a red folder, a shoebox
    (bx, by), _, _ = obb("floor_clutter_11")
    soft_blob("backpack", p, (bx + 0.1, by, 0.2), (0.17, 0.13, 0.22), "plastic_black", seed=41)
    me = bpy.data.meshes.new("rug")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=96, radius=0.55)
    bm.to_mesh(me)
    bm.free()
    rug = link(bpy.data.objects.new("rug", me), p)
    rug.location = (2.2, 1.25, 0.002)
    s = rug.modifiers.new("pile", "SOLIDIFY")
    s.thickness = 0.012
    finish(rug, "rug", 0.006, 3)
    rug.modifiers.move(rug.modifiers.find("bevel"), len(rug.modifiers) - 1)
    soft_blob("green_bag", p, (2.85, 2.9, 0.12), (0.18, 0.12, 0.12), "green_bag", seed=51)
    box(
        "red_folder",
        p,
        (2.85, 2.93, 0.2),
        (0.24, 0.02, 0.18),
        "red_paper",
        bevel=0.002,
        rot=(0.25, 0, 0.1),
    )


def garment(name, xy, top, length, w_top, w_hem, m, seed, depth=0.4):
    """A hanging garment: a flattened tube, narrow where it is gathered on the hook, widening to
    the hem, with vertical folds that deepen toward the bottom and a ribbed hem band."""
    nseg, nring = 48, 30
    r = random.Random(seed)
    ph = [r.uniform(0, 6.3) for _ in range(nring + 1)]
    verts, faces = [], []
    for j in range(nring + 1):
        t = j / nring
        z = top - length * t
        half_w = (w_top + (w_hem - w_top) * (1 - (1 - t) ** 2)) / 2
        fold = 0.02 + 0.09 * t
        for i in range(nseg):
            a = 2 * math.pi * i / nseg
            k = 1 + fold * math.sin(7 * a + ph[j] * 0.3 + 2.0 * t) + 0.04 * math.sin(13 * a + ph[0])
            if j >= nring - 2:  # hem band pulls in
                k *= 0.94
            verts.append(
                (xy[0] + math.cos(a) * half_w * depth * k, xy[1] + math.sin(a) * half_w * k, z)
            )
    for j in range(nring):
        for i in range(nseg):
            a0, a1 = j * nseg + i, j * nseg + (i + 1) % nseg
            faces.append((a0, a1, a1 + nseg, a0 + nseg))
    faces.append(tuple(range(nring * nseg, (nring + 1) * nseg))[::-1])
    o = mesh_from(name, "replica_props", verts, faces, m)
    sd = o.modifiers.new("smooth", "SUBSURF")
    sd.levels = sd.render_levels = 1
    return o


def photo_card(name, frame, rect, plane_y):
    """An RGBA cut-out from `frame` (walk, file) laid on the plane Y = plane_y exactly where the
    frame's solved camera saw it: rays through the rect corners, a bilinear grid between them."""
    row = next(
        q
        for q in map(json.loads, (ROOM / "recon-v2" / "v2.1" / "cams.jsonl").open())
        if (q["walk"], q["file"]) == frame
    )
    k = row["k"]
    sx, sy = 504 / (2 * k[0][2]), 896 / (2 * k[1][2])  # K is for the downscaled solve image
    a = Matrix(((1, 0, 0, OX), (0, 0, -1, OY), (0, 1, 0, -FLOOR_Y), (0, 0, 0, 1)))
    cam = a @ Matrix(row["pose"])
    org = cam.translation

    def hit(u, v):
        d = cam.to_3x3() @ Vector(((u / sx - k[0][2]) / k[0][0], (v / sy - k[1][2]) / k[1][1], 1.0))
        return org + d * ((plane_y - org.y) / d.y)

    x0, y0, x1, y1 = rect
    bl, br, tr, tl = hit(x0, y1), hit(x1, y1), hit(x1, y0), hit(x0, y0)
    n = 8
    verts, uvs, faces = [], [], []
    for j in range(n + 1):
        for i in range(n + 1):
            u, v = i / n, j / n
            verts.append(bl.lerp(br, u).lerp(tl.lerp(tr, u), v))
            uvs.append((u, v))
    for j in range(n):
        for i in range(n):
            q = j * (n + 1) + i
            faces.append((q, q + 1, q + n + 2, q + n + 1))
    o = mesh_from(name, "replica_props", verts, faces, M[name])
    uv = o.data.uv_layers.new()
    for lp in o.data.loops:
        uv.data[lp.index].uv = uvs[lp.vertex_index]
    return o


def chair():
    (cx, cy), _, _ = obb("chair_12")
    p = "replica_furniture"
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(HIFI / "objaverse" / "chair.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        C[p].objects.link(o)
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in new if o.type == "MESH" for c in o.bound_box]
    lo = Vector((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)))
    hi = Vector((max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))
    root = link(bpy.data.objects.new("office_chair", None), p)
    for o in new:
        if o.parent is None:
            o.parent = root
        for sl in getattr(o, "material_slots", []):
            m = sl.material
            b = m and m.node_tree and _bsdf(m)
            if b and not b.inputs["Base Color"].is_linked:
                b.inputs["Base Color"].default_value = (0.02, 0.021, 0.024, 1.0)
                b.inputs["Roughness"].default_value = 0.5
    face = math.pi + 0.35  # the seat looks toward the desk (west), turned a little
    root.matrix_world = (
        Matrix.Translation((cx, cy, 0.0))
        @ Matrix.Rotation(face - math.pi, 4, "Z")
        @ Matrix.Diagonal((0.0105, 0.0105, 0.0105, 1.0))
        @ Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    )


# ---------------------------------------------------------------- lights, cameras, render


def build_lights():
    lc = "replica_lights"
    # late-afternoon sun from the south-west through the shutters
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.0
    sun.angle = math.radians(0.7)
    sun.color = (1.0, 0.9, 0.79)
    so = link(bpy.data.objects.new("sun", sun), lc)
    d = Vector((0.18, 1.0, -0.66)).normalized()
    so.rotation_euler = (-d).to_track_quat("Z", "Y").to_euler()
    # skylight through the window, as a portal so Cycles aims its samples there
    for k, portal in enumerate((True, False)):
        a = bpy.data.lights.new(f"window_{'portal' if portal else 'fill'}", "AREA")
        a.shape = "RECTANGLE"
        a.size, a.size_y = 0.9, 1.2
        a.energy = 0 if portal else 18.0
        a.color = (0.78, 0.86, 1.0)
        if portal:
            a.cycles.is_portal = True
        ao = link(bpy.data.objects.new(a.name, a), lc)
        ao.location = (1.5, Y0 - WALL_T + 0.02 + 0.02 * k, 1.55)
        ao.rotation_euler = (-math.pi / 2, 0, 0)
    for k, (x, y) in enumerate(((2.05, 2.55), (3.45, 1.75))):
        s = bpy.data.lights.new(f"downlight{k}", "SPOT")
        s.energy = 22.0
        s.spot_size = math.radians(110)
        s.spot_blend = 0.6
        s.shadow_soft_size = 0.05
        s.color = (1.0, 0.78, 0.55)
        o = link(bpy.data.objects.new(s.name, s), lc)
        o.location = (x, y, CEIL - 0.02)
    # monitor spill on the wall behind the screens
    (dcx, dcy), _, _ = obb("desk_01")
    a = bpy.data.lights.new("monitor_glow", "AREA")
    a.size, a.size_y = 0.68, 0.3
    a.energy = 6.0
    a.color = (0.8, 0.9, 1.0)
    o = link(bpy.data.objects.new("monitor_glow", a), lc)
    o.location = (X0 + 0.27, dcy - 0.3, 2 * obb("desk_01")[1][2] + 0.29)
    o.rotation_euler = (0, -math.pi / 2, 0)


def world():
    w = bpy.data.worlds.new("replica_sky")
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    sky = nt.nodes.new("ShaderNodeTexSky")
    types = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = next(
        (t for t in ("NISHITA", "MULTIPLE_SCATTERING", "SINGLE_SCATTERING") if t in types), types[0]
    )
    try:
        sky.sun_elevation = math.radians(28)
        sky.sun_rotation = math.radians(200)
    except AttributeError:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6
    return w


def setup_render(sc, engine):
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    try:
        sc.view_settings.view_transform = "AgX"
        looks = [i.identifier for i in sc.view_settings.bl_rna.properties["look"].enum_items]
        sc.view_settings.look = next((x for x in looks if "Medium High" in x), "None")
    except TypeError:
        pass
    sc.view_settings.exposure = 0.7
    if engine == "CYCLES":
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 512
        sc.cycles.adaptive_threshold = 0.008
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 10
        sc.cycles.diffuse_bounces = 6
        sc.cycles.glossy_bounces = 4
        sc.cycles.transmission_bounces = 8
        sc.cycles.sample_clamp_indirect = 8.0
        sc.cycles.blur_glossy = 0.5
        sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    else:
        engines = [i.identifier for i in sc.render.bl_rna.properties["engine"].enum_items]
        sc.render.engine = next(e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if e in engines)
        ee = sc.eevee
        for attr, val in (
            ("use_raytracing", True),
            ("taa_render_samples", 64),
            ("use_shadows", True),
            ("use_fast_gi", True),
            ("fast_gi_method", "GLOBAL_ILLUMINATION"),
        ):
            if hasattr(ee, attr):
                try:
                    setattr(ee, attr, val)
                except TypeError:
                    pass


def cameras():
    c = "replica_cameras"
    # eye level from the east door, looking west over the bed to the desk and window
    cd = bpy.data.cameras.new("cam_eye")
    cd.lens = 22
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = 4.0
    cd.dof.focus_distance = 2.6
    eye = link(bpy.data.objects.new("cam_eye", cd), c)
    eye.location = (X1 - 0.18, 3.35, 1.62)
    target = Vector((1.0, 1.7, 1.0))
    eye.rotation_euler = (target - eye.location).to_track_quat("-Z", "Y").to_euler()
    # dollhouse: straight down through the missing ceiling
    ct = bpy.data.cameras.new("cam_top")
    ct.type = "ORTHO"
    ct.ortho_scale = max(X1 - X0, Y1 - Y0) * 1.95
    top = link(bpy.data.objects.new("cam_top", ct), c)
    top.location = ((X0 + X1) / 2, (Y0 + Y1) / 2, 9.0)
    sc = bpy.context.scene
    sc.camera = eye


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = "replica"
    for n in (
        "replica_architecture",
        "replica_ceiling",
        "replica_furniture",
        "replica_props",
        "replica_lights",
        "replica_cameras",
    ):
        C[n] = bpy.data.collections.new(n)
        sc.collection.children.link(C[n])
    C["replica_architecture"]["frame"] = (
        "sim frame: X = mesh x + 1.6, Y = -mesh z + 1.5, Z = mesh y - floor_y (0.103)"
    )
    build_materials()
    build_walls()
    build_wall_fittings()
    build_ceiling()
    build_standing_desk()
    build_folding_table()
    build_whiteboard_and_hoop()
    build_built_in_misc()
    build_walls_art()
    build_small_furniture()
    build_bed()
    build_lights()
    sc.world = world()
    cameras()
    setup_render(sc, "CYCLES")
    ev = bpy.data.scenes.new("replica_eevee")
    for coll in C.values():
        ev.collection.children.link(coll)
    ev.world = sc.world
    ev.camera = sc.camera
    setup_render(ev, "EEVEE")
    bpy.context.window_manager.windows[0].scene = sc if bpy.context.window_manager.windows else None
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "room.blend"), compress=True)


if __name__ == "__main__":
    main()
