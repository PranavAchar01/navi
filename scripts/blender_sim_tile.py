"""One room of the parallel-walks reel as its own small clip: every iteration's walk of one tile,
rendered at tile size from the scene files in the grid folder. Tiles are independent, so
scripts/run_sim_tiles.sh runs many of them at once; scripts/build_sim_reel.py lays them out.

SIMULATED USER MODEL, NOT A HUMAN STUDY.

    blender -b --factory-startup --python scripts/blender_sim_tile.py -- \
        --grid data/demo-takes/sim/grid --tile eight-1000 --out data/demo-takes/sim/tiles \
        [--arms I0,I3,I5,I6 --cap 330 --step 2 --scale 20 --samples 4 --first 0 --last N
         --res 576x400]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_sim_room as R  # noqa: E402
import sim_motion  # noqa: E402


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1 :]
    return {a[k].lstrip("-"): a[k + 1] for k in range(0, len(a), 2)}


def render_arm(a: dict, arm: str):
    sc = json.loads((Path(a["grid"]) / "scenes" / f"{arm}-{a['tile']}.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.system.use_gpu_subdivision = False
    R.COLL = bpy.context.scene.collection
    M = R.build_materials()
    bounds = R.room_bounds(sc["walls"])
    R.build_room(sc, M, ("S", "E"))
    chairs = R.build_furniture(sc, M, bounds)
    walker = R.Walker(sc["motor_deg"], M)
    frames = sc["frames"]
    seat = R.seat_of(sc, chairs)
    rects = [
        (o["obb"][0], o["obb"][1], o["obb"][3], o["obb"][4], o["obb"][6]) for o in sc["objects"]
    ]
    plan = sim_motion.plan(frames, sc["motor_deg"], seat=seat and seat[:3], rects=rects)
    if seat is not None and any(abs(m.chair_turn) > 1e-4 for m in plan):
        R.swivel(chairs[sc["target"]][0], seat[0], seat[1], [m.chair_turn for m in plan])
    last = min(int(a.get("last", len(frames) - 1)), int(a.get("cap", 330)) - 1, len(frames) - 1)
    first = int(a.get("first", 0))
    step = int(a.get("step", 1))
    for k in range(0, last + 1):
        walker.pose(plan[k], frames[k], k + 1, seat and seat[3])
    R.finish_walker(walker)
    R.setup_light(bounds)
    if "res" in a:  # a tile of its own shape, e.g. 576x400: the room fills more of it
        R.W, R.H = (int(v) for v in a["res"].split("x"))
        rs = bpy.context.scene.render
        rs.resolution_x, rs.resolution_y = R.W, R.H  # the camera fit reads the aspect
    cam = R.setup_camera(bounds)
    R.setup_render(int(a.get("samples", 4)), int(a.get("scale", 20)))
    scn = bpy.context.scene
    scn.render.image_settings.file_format = "JPEG"
    scn.render.image_settings.quality = 92
    ee = scn.eevee
    if hasattr(ee, "shadow_pool_size"):
        ee.shadow_pool_size = "512"
    scn.frame_start, scn.frame_end, scn.frame_step = first + 1, last + 1, step
    d = Path(a["out"]) / a["tile"] / arm
    d.mkdir(parents=True, exist_ok=True)
    R.write_cam(cam, d / "cam.json")  # world -> pixel, for the overlay pass
    scn.render.filepath = str(d / "#####")
    bpy.ops.render.render(animation=True)


def main():
    a = args()
    for arm in a.get("arms", "I0,I3,I5,I6").split(","):
        render_arm(a, arm)


if __name__ == "__main__":
    main()
