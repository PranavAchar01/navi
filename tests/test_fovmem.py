import math

import numpy as np

from metasight.fovmem import (
    HazardMemory,
    PitchFuse,
    band_name,
    floor_pitch,
    gate_memory_cues,
    level_floor,
    level_points,
    loose_floor,
    pitch_from_gravity,
    prior_cells,
)

HFOV = 44.8
EYE = 1.6
W, H = 108, 192  # portrait, as the glasses stream


def render(pitch, wall=None, box=None, eye=EYE):
    """Z-depth of a floor seen `pitch` degrees down, plus a wall `wall` m ahead and a box
    (distance, half width, top height) standing on the floor straight ahead."""
    f = 0.5 * W / math.tan(math.radians(HFOV) / 2)
    u, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    p = math.radians(pitch)
    down = math.sin(p) + math.cos(p) * v / f  # per unit z
    ahead = math.cos(p) - math.sin(p) * v / f
    z = np.where(down > 1e-6, eye / np.maximum(down, 1e-6), np.inf)
    if wall is not None:
        z = np.minimum(z, np.where(ahead > 1e-6, wall / np.maximum(ahead, 1e-6), np.inf))
    if box is not None:
        dist, half, top = box
        zb = np.where(ahead > 1e-6, dist / np.maximum(ahead, 1e-6), np.inf)
        hb = eye - zb * down
        right = zb * u / f
        hit = (hb >= 0) & (hb <= top) & (np.abs(right) <= half)
        z = np.where(hit & (zb < z), zb, z)
    return np.where(np.isfinite(z) & (z < 15), z, 0.0).astype(np.float32)


def test_pitch_from_gravity_level_and_down():
    assert abs(pitch_from_gravity([-9.8, 0, 0])) < 1e-6
    g = [-9.8 * math.cos(math.radians(30)), 0, -9.8 * math.sin(math.radians(30))]
    assert abs(pitch_from_gravity(g) - 30) < 1e-6


def test_floor_pitch_matches_level_points():
    p = math.radians(25)
    assert abs(floor_pitch(np.array([0, -math.cos(p), -math.sin(p)])) - 25) < 1e-6
    _, fwd, hgt = level_points(render(25), HFOV, 25, EYE)
    floor = render(25) > 0
    assert np.abs(hgt[floor]).max() < 1e-3
    assert fwd[floor].min() > 0


def test_level_floor_ignores_a_box_and_recovers_from_a_bad_guess():
    z = render(15, box=(1.4, 0.15, 0.12))
    for guess in (9, 15, 22):
        p, h = level_floor(z, HFOV, guess, EYE)
        assert abs(p - 15) < 0.2 and abs(h - EYE) < 0.02
    # Scaled depth: the camera height comes back scaled, which is the depth scale's ruler.
    assert abs(level_floor(z * 1.1, HFOV, 15, EYE)[1] - 1.1 * EYE) < 0.03


def test_level_floor_refuses_a_short_strip_of_floor():
    # Head near level, wall 2.4 m ahead: the floor shows only from about 1.9 m to the wall.
    assert level_floor(render(5, wall=2.4), HFOV, 5, EYE, min_span=0.8) is None
    p, _ = level_floor(render(30, wall=2.4), HFOV, 30, EYE, min_span=0.8)
    assert abs(p - 30) < 0.5


def test_level_floor_none_without_floor():
    assert level_floor(render(-20, wall=1.0), HFOV, -20, EYE) is None


def test_pitch_fuse_learns_offset_and_gates_outliers():
    fuse = PitchFuse(snap=False)
    n = lambda deg: np.array([0, -math.cos(math.radians(deg)), -math.sin(math.radians(deg))])  # noqa: E731
    for _ in range(10):
        fuse.update(12.0, n(15.0))  # the accelerometer reads 3 degrees low
    assert abs(fuse.offset - 3.0) < 1e-6
    fuse.update(12.0, n(24.0))  # a tilted fit (box in view): refused
    assert abs(fuse.offset - 3.0) < 1e-6
    assert abs(fuse.update(40.0) - 43.0) < 1e-6  # floor out of view: accelerometer plus offset
    for _ in range(25):  # glasses moved on the nose: a steady new offset is taken afresh
        fuse.update(12.0, n(20.0))
    assert abs(fuse.offset - 8.0) < 0.5


def test_pitch_fuse_snap_returns_the_floor():
    fuse = PitchFuse()
    p = math.radians(15)
    assert abs(fuse.update(13.0, np.array([0, -math.cos(p), -math.sin(p)])) - 15) < 1e-6


def test_band_names():
    assert band_name(0.1, EYE) == "low"
    assert band_name(0.8, EYE) == "hip"
    assert band_name(1.5, EYE) == "head"


def test_memory_holds_a_box_after_it_drops_under_the_view():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    box_at = 1.7
    out = None
    for i, x in enumerate(np.arange(0.0, 1.3, 0.1)):
        out = mem.update(i * 0.1, render(15, box=(box_at - x, 0.15, 0.3)), 15.0, 0.0, (0.0, x))
    # At 0.5 m the box is under the bottom edge of the view; memory still has it ahead.
    assert out.hidden[0] < 0.6 and abs(out.hidden[0] - 0.5) < 0.12
    assert out.path_m < 0.6
    assert band_name(out.hidden_h[0], EYE) == "low"


def test_memory_drops_a_cell_seen_through():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    for i in range(3):
        mem.update(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    held = mem.update(0.3, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0).cells
    assert held > 0
    out = mem.update(0.4, render(15), 15.0, 0.0)  # the box was carried away: floor behind it
    assert out.cleared > 0 and out.cells < held


def test_memory_forgets_after_keep_s_and_needs_min_hits():
    mem = HazardMemory(HFOV, EYE, keep_s=1.0, min_hits=2, low_slope=0.0)
    out = mem.update(0.0, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    assert math.isinf(out.path_m)  # one sighting is not enough
    mem.update(0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    # Turn 90 degrees to the right: the box is held out of view on the left.
    out = mem.update(0.2, render(15), 15.0, -math.pi / 2)
    assert out.hidden[6] < 1.6
    out = mem.update(1.5, render(15), 15.0, -math.pi / 2)
    assert out.cells == 0


def test_memory_keys_dedupe():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    z = render(15, box=(1.5, 0.15, 0.3))
    mem.update(0.0, z, 15.0, 0.0)
    n = len(mem.key)
    mem.update(0.1, z, 15.0, 0.0)
    assert len(mem.key) == n and np.all(np.diff(mem.key) > 0) and mem.hits.max() == 2


def test_memory_gate_moved_needs_head_motion():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0, gate="moved")
    for i in range(2):
        mem.update(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    # Same pose, frame lost (blank depth): nothing out of view the head's motion explains.
    out = mem.update(0.2, np.zeros((H, W), np.float32), 15.0, 0.0)
    assert math.isinf(out.path_m)
    out = mem.update(0.3, render(15), 15.0, -math.pi / 2)  # a 90 degree turn explains it
    assert out.hidden[6] < 1.6


def test_fov_loop_shadow_record():
    from types import SimpleNamespace

    from metasight.fovmem import FovLoop

    loop = FovLoop(HFOV, EYE, low_slope=0.0)
    head = SimpleNamespace(yaw=None, turn=0.0, pos=None)
    p = math.radians(15)
    grav = [[-9.8 * math.cos(p) + 0.0, 0.0, -9.8 * math.sin(p)]]  # glasses axes, 15 down
    grid = np.full((H, W), 0, np.uint8)  # every pixel an obstacle candidate
    person = SimpleNamespace(label="person", box=(0.0, 0.0, 1.0, 1.0))
    out = loop.step(0.0, render(15, box=(1.5, 0.15, 0.3)), grid, True, [person], grav, head)
    assert out["floor"] and abs(out["pitch"] - 15) < 0.5 and out["cells"] == 0  # all masked
    for i in range(1, 3):
        out = loop.step(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), grid, True, [], grav, head)
    assert out["cells"] > 0
    head.turn = -math.pi / 2  # turned right: the box is out of view on the left
    out = loop.step(0.3, render(15), grid, True, [], grav, head)
    assert out["hidden"][6] is not None and out["band"][6] == "low"


def render_drop(pitch, edge, step=0.18, eye=EYE):
    """Z-depth of a floor that ends `edge` m ahead in a step down of `step` metres."""
    f = 0.5 * W / math.tan(math.radians(HFOV) / 2)
    _, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    p = math.radians(pitch)
    down = math.sin(p) + math.cos(p) * v / f
    ahead = math.cos(p) - math.sin(p) * v / f
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = eye / down
        z = np.where(upper * ahead <= edge, upper, (eye + step) / down)
    return np.where((down > 1e-6) & (z < 15), z, 0.0).astype(np.float32)


def test_drop_memory_holds_an_edge_after_it_leaves_the_view():
    from metasight.fovmem import DropMemory

    mem = DropMemory(HFOV, EYE)
    seen = math.inf
    for i in range(10):  # walking up from 3.5 m to the edge, the head level (10 degrees)
        x = 0.1 * i
        seen = mem.update(i / 13, render_drop(10, 3.5 - x), 10.0, 0.0, (0.0, x))
    assert math.isinf(seen)  # still 2.6 m off: beyond the 2 m the path looks
    for i in range(10, 22):  # from 1.5 m the view's bottom (46 degrees) no longer holds the edge
        x = 0.1 * i
        reported = mem.update(i / 13, render_drop(10, 3.5 - x), 10.0, 0.0, (0.0, x))
    assert abs(reported - (3.5 - 2.1)) < 0.3


def test_drop_memory_stays_quiet_on_plain_floor_and_a_low_box():
    from metasight.fovmem import DropMemory

    mem = DropMemory(HFOV, EYE)
    for i in range(10):
        assert math.isinf(mem.update(i / 13, render(10), 10.0, 0.0, (0.0, 0.0)))
    for i in range(10):  # the far edge of a 12 cm box is not a drop
        out = mem.update(i / 13, render(15, box=(1.7, 0.15, 0.12)), 15.0, 0.0, (0.0, 0.0))
        assert math.isinf(out)
    assert len(mem.key) == 0


def test_drop_memory_does_not_add_without_a_trusted_pitch():
    from metasight.fovmem import DropMemory

    mem = DropMemory(HFOV, EYE)
    for i in range(5):
        out = mem.update(i / 13, render_drop(10, 2.0), 10.0, 0.0, (0.0, 0.0), add=False)
    assert math.isinf(out) and len(mem.key) == 0


def test_memory_does_not_store_the_slab_when_the_floor_fit_sits_on_the_lower_floor():
    # Heights measured from the lower floor lift the slab the wearer stands on by the step.
    mem = HazardMemory(HFOV, EYE + 0.18, low_slope=0.02)
    for i in range(6):
        c = mem.update(i / 13, render_drop(10, 2.2), 10.0, 0.0, (0.0, 0.0))
    assert c.cells == 0


def test_memory_still_stores_a_box_on_a_plain_floor_seen_from_the_same_view():
    mem = HazardMemory(HFOV, EYE, low_slope=0.02)
    for i in range(6):
        c = mem.update(i / 13, render(10, box=(1.8, 0.3, 0.3)), 10.0, 0.0, (0.0, 0.0))
    assert c.cells > 0


def _obj(oid, cls, c, size, yaw=0.0, clr=0.0, hazard=True):
    return {
        "id": oid,
        "class": cls,
        "hazard": hazard,
        "floor_clearance_m": clr,
        "obb": {"center": c, "size": size, "yaw_rad": yaw},
    }


def test_objects_json_builds_geoms():
    from metasight import fovrooms

    data = {
        "floor_y": 0.1,
        "ceiling_height_m": 2.8,
        "objects": [
            _obj("desk_a", "desk", [1.0, 0.5, -1.0], [1.0, 0.8, 0.6], clr=0.6),
            _obj("box_b", "floor clutter", [0.0, 0.2, 0.0], [0.4, 0.2, 0.1], yaw=0.5),
            _obj("laptop_c", "laptop", [1.0, 0.7, -1.0], [0.3, 0.02, 0.2], hazard=False),
            _obj("door_d", "door", [0.5, 1.2, -2.5], [0.8, 2.1, 0.05]),
        ],
    }
    boxes, by_id = fovrooms.scene_boxes(data, open_doors=("door_d",))
    assert set(by_id) == {"desk_a", "box_b"}  # the laptop is no hazard, the door a wall gap
    desk = by_id["desk_a"]
    assert len(desk) == 1 + 4  # slab plus four legs
    slab = desk[0]
    assert math.isclose(slab[0], 1.0) and math.isclose(slab[1], 1.0)  # Y = -z
    assert math.isclose(slab[2] + slab[5], 0.8, abs_tol=1e-6)  # top 0.9 minus floor 0.1
    assert slab[2] - slab[5] >= 0.55  # underside at the clearance
    assert len(by_id["box_b"]) > 1  # a turned footprint is cut into strips
    rest = boxes[len(desk) + len(by_id["box_b"]) :]
    assert len([b for b in rest if b[1] > 2.0]) == 2  # back wall split by the open door


def test_loose_floor_backs_a_pitch_that_agrees_and_not_one_that_does_not():
    z = render(20.0)
    assert loose_floor(z, HFOV, 20.5, EYE, 20.2)
    assert not loose_floor(z, HFOV, 20.5, EYE, 23.0)  # the pitch in use is 3 degrees off
    assert not loose_floor(z, HFOV, 20.5, EYE * 1.3, 20.2)  # camera height 30% off
    assert not loose_floor(np.zeros_like(z), HFOV, 20.0, EYE, 20.0)  # no depth, no floor


def _ring(**kw):
    """A step result with a cell in every sector: 0.8 m away, `close` metres closed on."""
    hidden = [0.8] * 8
    band = ["low"] * 8
    close = [kw.get("close", 0.5)] * 8
    return hidden, band, close


def test_gate_still_mute_silences_the_sectors_the_camera_sees_only_after_a_second():
    h, b, c = _ring()
    kw = dict(approach=False, mute_rear=False)
    gh, gb, _ = gate_memory_cues(h, b, c, 1.5, **kw)
    assert [gh[s] for s in (7, 0, 1)] == [None] * 3 and gb[0] is None
    assert gh[2] == 0.8 and gh[4] == 0.8  # sectors out of the camera's view keep their cue
    assert gate_memory_cues(h, b, c, 0.9, **kw)[0] == h  # a step within the second: no mute
    assert gate_memory_cues(h, b, c, None, **kw)[0] == h  # unknown stillness never mutes
    _, _, p = gate_memory_cues(h, b, c, 2.0, path=0.7, path_close=0.5, **kw)
    assert p is None  # the path cell is ahead, in view


def test_gate_approach_needs_the_wearer_to_have_closed_in_on_the_cell():
    h, b, c = _ring()
    c[0], c[1], c[2], c[3] = 0.05, -0.2, None, math.nan
    kw = dict(mute_still=False, mute_rear=False)
    gh, _, p = gate_memory_cues(h, b, c, 0.0, path=0.6, path_close=0.05, **kw)
    assert gh[0] is None and gh[1] is None and gh[2] is None and gh[3] is None
    assert gh[4] == 0.8 and gh[7] == 0.8  # 0.5 m closed
    assert p is None
    assert gate_memory_cues(h, b, c, 0.0, path=0.6, path_close=0.4, **kw)[2] == 0.6


def test_gate_rear_mute_keeps_front_and_side_only():
    h, b, c = _ring()
    gh, gb, _ = gate_memory_cues(h, b, c, 0.0, mute_still=False, approach=False)
    assert [s for s in range(8) if gh[s] is not None] == [0, 1, 2, 6, 7]
    assert gb[4] is None and gb[0] == "low"
    # the gate never adds a cue: no cell stays no cell
    none = [None] * 8
    assert gate_memory_cues(none, none, none, 0.0, mute_still=False)[0] == none
    assert gate_memory_cues([math.inf] * 8, none, none, 0.0, approach=False)[0] == none


def _prior_box(cz, x=0.0, sx=0.4, sz=0.4, h=0.9, movable=False, clearance=0.0, **extra):
    """A prior-map hazard box in map coordinates (x right, z ahead, heights above the floor)."""
    return {
        "id": "o",
        "class": "box",
        "hazard": True,
        "movable": movable,
        "center": np.array([x, h / 2, cz]),
        "size": [sx, h, sz],
        "yaw_rad": 0.0,
        "floor_clearance_m": clearance,
        **extra,
    }


def test_prior_cells_leave_the_space_under_an_overhang_and_skip_non_hazards():
    desk = _prior_box(1.0, sx=1.2, sz=0.7, h=0.85, clearance=0.72)
    pts, need, grp = prior_cells([desk], band=(0.08, 1.75))
    assert pts[:, 2].min() >= 0.72 and pts[:, 2].max() <= 0.85
    assert len(pts) == len(need) == len(grp)
    assert abs(pts[:, 1].min() - 0.65) < 0.06 and abs(pts[:, 1].max() - 1.35) < 0.06
    skipped = [
        _prior_box(1.0, hazard=False),
        {**_prior_box(1.0), "class": "door"},
        _prior_box(1.0, sx=0.3, sz=0.2),
    ]
    assert len(prior_cells(skipped, min_dim=0.4)[0]) == 0
    # the map-to-memory matrix: map (x, z) -> memory (-z, x)
    turned = prior_cells([_prior_box(1.0, x=0.5)], xz=np.array([[0.0, -1, 0], [1, 0, 0]]))[0]
    assert abs(turned[:, 0].mean() + 1.0) < 0.05 and abs(turned[:, 1].mean() - 0.5) < 0.05


def test_seeded_prior_reports_a_hidden_hazard_from_the_first_frame():
    mem = HazardMemory(HFOV, EYE)
    n = mem.seed([_prior_box(-1.3), _prior_box(2.0, hazard=False)])
    assert n > 0 and mem.prior.all() and len(set(mem.key)) == len(mem.key)
    out = mem.update(0.0, render(15), 15.0, 0.0)  # facing +z, the box is behind
    assert 1.0 < out.hidden[4] < 1.4 and math.isinf(out.hidden[0])
    # a prior cell does not expire, even after the memory's keep time and a walk away
    out = mem.update(60.0, np.zeros((H, W), np.float32), 15.0, 0.0, (0.0, -0.5))
    assert 0.5 < out.hidden[4] < 0.9 and mem.prior.all()


def test_prior_cells_clear_after_one_look_when_movable_and_several_when_fixed():
    free = render(15)  # the room is empty: the camera sees the floor through the prior box
    for movable, frames in ((True, 1), (False, 3)):
        mem = HazardMemory(HFOV, EYE)
        mem.seed([_prior_box(1.6, movable=movable)], need_fixed=3, need_movable=1)
        n = len(mem.key)
        cleared = [mem.update(i * 0.1, free, 15.0, 0.0).cleared for i in range(frames)]
        assert cleared[-1] > 0 and all(c == 0 for c in cleared[:-1]) and len(mem.key) < n


def test_prior_cell_seen_where_the_map_has_it_is_kept_and_its_count_resets():
    mem = HazardMemory(HFOV, EYE)
    mem.seed([_prior_box(1.6)], need_fixed=3)  # front face at 1.4 m
    mem.update(0.0, render(15), 15.0, 0.0)  # seen through once: counts 1 on the cells in view
    assert mem.free.max() == 1
    for i in range(1, 6):
        mem.update(i * 0.1, render(15, box=(1.4, 0.2, 0.9)), 15.0, 0.0)
    front = mem.prior & (mem.pts[:, 1] < 1.46)
    assert front.sum() > 0 and mem.free[front].max() == 0


def test_live_cells_still_clear_at_once_beside_prior_cells():
    mem = HazardMemory(HFOV, EYE, low_slope=0.0)
    mem.seed([_prior_box(-1.3)])
    for i in range(3):
        mem.update(i * 0.1, render(15, box=(1.5, 0.15, 0.3)), 15.0, 0.0)
    live = int((~mem.prior).sum())
    assert live > 0 and mem.need[~mem.prior].max() == 1
    out = mem.update(0.4, render(15), 15.0, 0.0)  # the box was carried away
    assert out.cleared > 0 and (~mem.prior).sum() < live and mem.prior.sum() > 0


def test_seed_margin_confirm_and_retire():
    far = _prior_box(-1.3)
    plain, padded = HazardMemory(HFOV, EYE), HazardMemory(HFOV, EYE)
    plain.seed([far])
    padded.seed([far], margin=0.25)
    a = plain.update(0.0, render(15), 15.0, 0.0).hidden[4]
    b = padded.update(0.0, render(15), 15.0, 0.0).hidden[4]
    assert abs((b - a) - 0.25) < 1e-6
    # unconfirmed: nothing reports until the camera has seen the object where the map has it
    sure = HazardMemory(HFOV, EYE)
    sure.seed([_prior_box(1.6), _prior_box(-1.3)], confirm_n=10)
    seen = render(15, box=(1.4, 0.2, 0.9))
    assert math.isinf(sure.update(0.0, seen, 15.0, 0.0).hidden[4])
    sure.update(0.05, seen, 15.0, 0.0)
    out = sure.update(0.1, render(15), 15.0, math.pi)  # turned round: the seen box is behind
    assert math.isfinite(out.hidden[4])
    # retire: a map object the camera keeps seeing through goes whole, cells out of view too
    gone = HazardMemory(HFOV, EYE)
    gone.seed([_prior_box(1.6, sx=1.4)], retire_frac=0.3, retire_min=5, need_fixed=99)
    n = len(gone.key)
    gone.update(0.0, render(15), 15.0, 0.0)
    assert len(gone.key) == 0 < n


def test_fov_loop_takes_a_prior():
    from metasight.fovmem import FovLoop

    loop = FovLoop(HFOV, EYE, prior={"floor_y": 0.0, "objects": [_prior_box(1.5)]})
    assert loop.mem.prior.sum() > 0
