"""Render the bedroom replica: the hero stills, the photo match and the flythrough frames.

    blender -b data/rooms/bedroom/replica/room.blend --python scripts/blender_replica_render.py \
        -- --shot eye|top|match|fly [--samples N] [--scale 1.0]

eye    hero-eye.png, eye level from the east door (camera cam_eye)
top    hero-top.png, dollhouse straight down with replica_ceiling hidden (camera cam_top)
match  match-render.png, from the solved pose of his iPhone frame (cams.jsonl) at its intrinsics;
       scripts/replica_compare.py puts it beside the real frame
fly    data/demo-takes/replica/fly/####.png, 7 s in Cycles from straight above down to eye
       level; skips frames already on disk, so a stopped run resumes; ffmpeg makes the mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
ROOM = ROOT / "data" / "rooms" / "bedroom"
OUT = ROOM / "replica"
MATCH = ("20261028-135040-iphone", "1793220651337-204.jpg")
OX, OY = 1.6, 1.5
FLOOR_Y = json.loads((ROOM / "floorplan.json").read_text())["floor_y"]
CEILING = 2.821
TAKES = ROOT / "data" / "demo-takes" / "replica"


def gpu(sc):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("METAL", "OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
        return kind
    return "CPU"


def match_camera(sc):
    """A camera at the PnP pose of the iPhone frame: OpenCV cam-to-world in the mesh frame,
    moved into the sim frame (X = x + 1.6, Y = -z + 1.5, Z = y - floor_y)."""
    row = next(
        r
        for r in map(json.loads, (ROOM / "recon-v2" / "v2.1" / "cams.jsonl").open())
        if (r["walk"], r["file"]) == MATCH
    )
    pose, k = Matrix(row["pose"]), row["k"]
    a = Matrix(((1, 0, 0, OX), (0, 0, -1, OY), (0, 1, 0, -FLOOR_Y), (0, 0, 0, 1)))
    cv_to_bl = Matrix.Diagonal((1, -1, -1, 1))
    cd = bpy.data.cameras.new("cam_match")
    w, h = 2 * k[0][2], 2 * k[1][2]
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.lens = k[0][0] / w * 36.0
    cam = bpy.data.objects.new("cam_match", cd)
    sc.collection.objects.link(cam)
    cam.matrix_world = a @ pose @ cv_to_bl
    sc.render.resolution_x, sc.render.resolution_y = 504, round(504 * h / w)
    return cam


def fly_path(sc, cam, seconds=7.0, fps=30):
    """One unbroken move from straight above the room down to eye level at the east door.

    The camera keeps lens 22 the whole way (no zoom), position eases on smootherstep and the
    view turns by quaternion slerp from looking straight down to the cam_eye framing. Image-up
    at the top is already the direction the camera will face, so the tilt reads as one gesture.
    The ceiling is invisible to camera rays while the camera is above it and visible again
    once it drops below, so the light in the room never changes; at that moment the view is
    still pitched well down and the ceiling is out of frame."""
    n = round(seconds * fps)
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, n
    end_loc = cam.location.copy()
    end_q = cam.matrix_world.to_quaternion()
    look = Vector((1.0, 1.7, 1.0)) - end_loc
    up = Vector((look.x, look.y, 0)).normalized()
    start_q = Matrix((up.cross(Vector((0, 0, 1))), up, Vector((0, 0, 1)))).transposed()
    start_q = start_q.to_quaternion()
    if start_q.dot(end_q) < 0:
        start_q.negate()
    start_loc = Vector((2.6, 2.4, 4.4))
    cam.rotation_mode = "QUATERNION"
    cam.data.lens = 22
    ceiling = list(bpy.data.collections["replica_ceiling"].all_objects)
    below = False
    for f in range(1, n + 1):
        t = (f - 1) / (n - 1)
        s = t * t * t * (t * (6 * t - 15) + 10)
        cam.location = start_loc.lerp(end_loc, s)
        cam.rotation_quaternion = start_q.slerp(end_q, s)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        if f == 1 or (cam.location.z < CEILING - 0.05) != below:
            below = cam.location.z < CEILING - 0.05
            for ob in ceiling:
                ob.visible_camera = below
                ob.keyframe_insert("visible_camera", frame=f)


TOUR = (  # (camera position, look-at point): east door, along to the desk, round to the bed, up
    ((4.15, 3.45, 1.60), (1.4, 1.9, 1.0)),
    ((3.2, 3.2, 1.60), (1.1, 1.9, 0.95)),
    ((2.2, 2.9, 1.45), (1.0, 1.7, 0.9)),
    ((1.9, 2.6, 1.55), (1.1, 1.6, 0.9)),
    ((2.0, 3.3, 1.70), (3.2, 1.9, 0.45)),
    ((2.7, 3.4, 1.85), (3.4, 1.7, 0.4)),
    ((2.8, 2.9, 2.5), (3.0, 2.1, 0.3)),
    ((2.54, 2.37, 5.2), (2.54, 2.37, 0.0)),
)


def catmull(pts, u):
    n = len(pts) - 1
    i = min(int(u * n), n - 1)
    f = u * n - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
    return 0.5 * (
        2 * p1
        + (p2 - p0) * f
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
        + (3 * p1 - p0 - 3 * p2 + p3) * f**3
    )


def tour_path(sc, cam, seconds=12.0, fps=24):
    """Door to desk to bed, then rising to straight down; lens 22 throughout. The ceiling is
    invisible to camera rays once the camera is above it."""
    n = round(seconds * fps)
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, n
    locs = [Vector(p) for p, _ in TOUR]
    tgts = [Vector(t) for _, t in TOUR]
    cam.rotation_mode = "QUATERNION"
    cam.data.lens = 22
    ceiling = list(bpy.data.collections["replica_ceiling"].all_objects)
    below = None
    prev = None
    for f in range(1, n + 1):
        t = (f - 1) / (n - 1)
        s = t * t * t * (t * (6 * t - 15) + 10) * 0.6 + t * 0.4  # eased, never fully stops mid-way
        loc, tgt = catmull(locs, s), catmull(tgts, s)
        d = tgt - loc
        q = d.to_track_quat("-Z", "Y")
        if s > 0.93:  # final approach: image-up is +Y, as in cam_top
            k = (s - 0.93) / 0.07
            top = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1))).to_quaternion()
            q = q.slerp(top, k * k * (3 - 2 * k))
        if prev is not None and q.dot(prev) < 0:
            q.negate()
        prev = q
        cam.location, cam.rotation_quaternion = loc, q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        b = loc.z < CEILING - 0.05
        if b != below:
            below = b
            for ob in ceiling:
                ob.visible_camera = b
                ob.keyframe_insert("visible_camera", frame=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True, choices=("eye", "top", "match", "fly", "tour"))
    ap.add_argument("--samples", type=int, default=384)
    ap.add_argument("--scale", type=float, default=1.0)
    # Metal can stall in Session::wait while other renders hold the GPU; CPU always finishes
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--part", help="tour/fly: I/N renders every Nth frame from I (parallel)")
    ap.add_argument("--frames", help="fly only: comma list of frames to render as a test")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    sc = bpy.data.scenes["replica"]
    sc.cycles.samples = args.samples
    print("device", "CPU" if args.cpu else gpu(sc))
    sc.render.resolution_percentage = round(100 * args.scale)
    sc.render.image_settings.file_format = "PNG"
    if args.shot == "eye":
        sc.camera = bpy.data.objects["cam_eye"]
        sc.render.filepath = str(OUT / "hero-eye.png")
    elif args.shot == "top":
        sc.camera = bpy.data.objects["cam_top"]
        sc.render.resolution_x = sc.render.resolution_y = 1600
        for c in ("replica_ceiling",):
            bpy.data.collections[c].hide_render = True
        bpy.data.objects["cam_top"].data.ortho_scale = 4.6
        sc.render.filepath = str(OUT / "hero-top.png")
    elif args.shot == "match":
        sc.camera = match_camera(sc)
        sc.camera.data.dof.use_dof = False
        # his frame was shot with no direct sun in the room: skylight and downlights only
        bpy.data.objects["sun"].hide_render = True
        sc.view_settings.exposure += 0.4
        sc.render.filepath = str(OUT / "match-render.png")
    else:
        sc.camera = bpy.data.objects["cam_eye"]
        sc.camera.data.dof.use_dof = False
        (tour_path if args.shot == "tour" else fly_path)(sc, sc.camera)
        # the same noise pattern every frame: the denoiser sees steady input, so no shimmer
        sc.cycles.use_animated_seed = False
        sc.cycles.seed = 7
        (TAKES / "fly").mkdir(parents=True, exist_ok=True)
        frames = [int(f) for f in args.frames.split(",")] if args.frames else None
        if args.part:
            i, n = map(int, args.part.split("/"))
            frames = frames or list(range(sc.frame_start + i, sc.frame_end + 1, n))
            skip = True
        else:
            skip = not frames
        for f in frames or range(sc.frame_start, sc.frame_end + 1):
            out = TAKES / "fly" / (f"t{f:04d}.png" if args.shot == "tour" else f"{f:04d}.png")
            if out.exists() and skip:
                continue
            sc.frame_set(f)
            sc.render.filepath = str(out)
            bpy.ops.render.render(write_still=True, scene=sc.name)
        return
    bpy.ops.render.render(write_still=True, scene=sc.name)


if __name__ == "__main__":
    main()
