import math

import numpy as np
import pytest

from metasight import spatial_sound as ss

HEAD = np.array([0.0, 1.6, 0.0])
FWD = np.array([0.0, 0.0, -1.0])  # ARKit: y up, looking down -z, so right is +x
UP = np.array([0.0, 1.0, 0.0])


def at(az_deg, dist=2.0, dy=0.0):
    a = math.radians(az_deg)
    return HEAD + np.array([math.sin(a) * dist, dy, -math.cos(a) * dist])


@pytest.mark.parametrize("az", [0, 45, 90, 135, 180, -135, -90, -45])
def test_relative_reads_back_the_azimuth(az):
    d = ss.relative(HEAD, FWD, UP, at(az))
    assert ((d.az_deg - az + 180) % 360) - 180 == pytest.approx(0, abs=1e-9)
    assert d.el_deg == pytest.approx(0) and d.dist_m == pytest.approx(2.0)


def test_relative_in_a_z_up_world_and_with_a_tilted_head():
    # z up, looking along +y: right is +x
    d = ss.relative(np.zeros(3), [0, 1, 0], [0, 0, 1], [1, 1, -1])
    assert d.az_deg == pytest.approx(45) and d.el_deg < 0
    # looking down at the floor does not change which side is right
    d = ss.relative(HEAD, [0, -0.8, -0.6], UP, at(90))
    assert d.az_deg == pytest.approx(90)


def test_head_vectors_from_an_arkit_matrix():
    pose = np.eye(4)
    pose[:3, 3] = HEAD
    yaw = math.radians(90)  # turned left: -x is ahead
    pose[:3, :3] = [
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ]
    p, f, u = ss.head_vectors(pose)
    assert f == pytest.approx([-1, 0, 0]) and u == pytest.approx([0, 1, 0])
    assert ss.relative(p, f, u, HEAD + [0, 0, -2]).az_deg == pytest.approx(90)  # old front: right


def energy(st):
    return (st.astype(float) ** 2).sum(axis=0)


def highs(x):
    spec = np.abs(np.fft.rfft(x, 4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / ss.SAMPLE_RATE)
    return spec[f > 5000].sum() / spec.sum()


@pytest.mark.parametrize("method", ["hrtf", "pan"])
def test_left_right_front_back(method):
    noise = np.random.default_rng(0).standard_normal(8192) * 0.1
    out = {az: ss.place(noise, ss.Direction(az, 0, 1), method) for az in (0, 90, 180, -90)}
    e = {az: energy(st) for az, st in out.items()}
    db = {az: 10 * math.log10(v[1] / v[0]) for az, v in e.items()}  # right over left
    assert db[90] > 6 and db[-90] < -6  # clearly on its side
    assert abs(db[0]) < 1.5 and abs(db[180]) < 1.5  # front and back sit in the middle
    # the right ear hears a right source first
    lag = np.argmax(np.correlate(out[90][:2048, 1], out[90][:2048, 0], "full")) - 2047
    assert lag < 0
    # back is duller than front, the cue that tells them apart
    assert highs(out[180].sum(1)) < highs(out[0].sum(1))


def test_hrir_mirrors_the_left_half():
    r, left = ss.hrir(60, 0), ss.hrir(-60, 0)
    assert np.allclose(r[:, ::-1], left)
    assert ss.hrir(0, -25).shape[1] == 2  # between measured elevations: the nearest


def test_rules_match_the_phone():
    assert ss.band(2.5) is None and ss.band(1.5) == "slow"
    assert ss.band(0.7) == "fast" and ss.band(0.3) == "stop"
    assert ss.tick_period(1.999) == pytest.approx(2.0, abs=0.01)
    assert ss.tick_period(1.0) == pytest.approx(1.2) and ss.tick_period(0.999) == pytest.approx(
        0.7, abs=0.01
    )
    assert ss.tick_period(0.3) == pytest.approx(0.2) and math.isinf(ss.tick_period(3))
    assert ss.beacon_period(6) == pytest.approx(1.4)
    assert ss.beacon_period(ss.ARRIVE_M) == pytest.approx(0.45)
    assert ss.beacon_period(2) < ss.beacon_period(3)
    assert ss.beacon_gain(10) == pytest.approx(ss.BEACON_GAIN / 4)
    assert ss.aligned(9) and not ss.aligned(-11)


def walk(n=200, dt=0.05, start=(0.0, 1.6, 0.0), step=(0.0, 0.0, -0.025)):
    t = np.arange(n) * dt
    pos = np.array(start) + np.outer(np.arange(n), step)
    return t, pos, np.tile(FWD, (n, 1)), np.tile(UP, (n, 1))


def test_beacon_speeds_up_turns_aligned_and_ends_with_the_chime():
    t, pos, f, u = walk()  # 5 m straight at a chair 4 m ahead
    chair = np.array([0.0, 0.5, -4.0])
    ev = ss.schedule(t, pos, f, u, target=chair)
    kinds = [e.kind for e in ev]
    assert kinds[-1] == "arrive" and kinds.count("arrive") == 1
    beacons = [e.t for e in ev if e.kind.startswith("beacon")]
    gaps = np.diff(beacons)
    assert gaps[0] > gaps[-1] + 0.5  # faster as it closes in
    assert all(k == "beacon_aligned" for k in kinds[:-1])  # straight ahead the whole way
    assert all(e.t <= ev[-1].t for e in ev)  # nothing after arrival


def test_beacon_off_axis_plays_the_single_note_from_that_side():
    t, pos, f, u = walk(n=40, step=(0, 0, 0))
    ev = ss.schedule(t, pos, f, u, target=at(-70, 3))
    assert ev and all(e.kind == "beacon" and e.where.az_deg < -60 for e in ev)
    st = ss.render(t, pos, f, u, target=at(-70, 3))
    left, right = energy(st)
    assert left > 2 * right


def test_hazards_tick_only_when_near_and_faster_when_closer():
    t, pos, f, u = walk(n=110)  # 2.75 m toward a box whose face is 3 m ahead
    box = ss.Hazard((-0.5, 0.0, -3.5), (0.5, 0.9, -3.0))
    ev = ss.schedule(t, pos, f, u, hazards=[box])
    assert ev and all(e.where.ground_m < 2.0 for e in ev)
    first = ev[0].t
    assert first > 1.0 / 0.5 * 0.9  # silent for the first metre or so
    gaps = np.diff([e.t for e in ev])
    assert gaps[-1] < gaps[0]
    assert {"slow", "fast", "stop"} <= {e.kind for e in ev}


def test_sound_off_is_silence_and_render_shape():
    t, pos, f, u = walk(n=40)
    st = ss.render(t, pos, f, u, target=at(30), beacon=False, ticks=False)
    assert st.shape == (int((t[-1] - t[0]) * ss.SAMPLE_RATE), 2) and not st.any()
    st = ss.render(t, pos, f, u, target=at(30), method="pan")
    assert st.dtype == np.float32 and 0 < np.abs(st).max() < 1


def test_write_wav(tmp_path):
    t, pos, f, u = walk(n=20)
    p = tmp_path / "a.wav"
    ss.write_wav(p, ss.render(t, pos, f, u, target=at(0)))
    assert p.stat().st_size > 44
