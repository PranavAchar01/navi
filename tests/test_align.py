import numpy as np

from metasight.align import (
    VoxelSet,
    apply_sim3,
    map_resolution,
    pose_through_sim3,
    robust_similarity,
    umeyama,
)


def random_sim3(seed=0, scale=1.07):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return scale, q, rng.normal(size=3) * 3


def test_umeyama_recovers_a_similarity():
    s, r, t = random_sim3()
    src = np.random.default_rng(1).normal(size=(500, 3))
    s2, r2, t2 = umeyama(src, apply_sim3(src, s, r, t))
    assert np.isclose(s, s2) and np.allclose(r, r2, atol=1e-8) and np.allclose(t, t2, atol=1e-8)


def test_robust_fit_survives_outliers_and_noise():
    s, r, t = random_sim3(3, 0.93)
    rng = np.random.default_rng(4)
    src = rng.normal(size=(2000, 3))
    dst = apply_sim3(src, s, r, t) + rng.normal(0, 0.01, (2000, 3))
    dst[:300] += rng.normal(0, 2.0, (300, 3))  # 15% gross outliers
    s2, r2, t2, resid = robust_similarity(src, dst)
    assert abs(s2 - s) < 0.01 and np.allclose(t2, t, atol=0.03) and resid < 0.05


def test_robust_fit_rejects_an_implausible_scale_by_going_rigid():
    s, r, t = random_sim3(5, 2.0)
    src = np.random.default_rng(6).normal(size=(500, 3))
    out = robust_similarity(src, apply_sim3(src, s, r, t))
    assert out[0] == 1.0


def test_too_few_points_gives_none():
    assert robust_similarity(np.zeros((10, 3)), np.zeros((10, 3))) is None


def test_pose_through_sim3_matches_points():
    s, r, t = random_sim3(7)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    moved = pose_through_sim3(pose, s, r, t)
    assert np.allclose(moved[:3, 3], apply_sim3(np.array([[1.0, 2, 3]]), s, r, t)[0])


def test_voxel_set_only_returns_new_voxels():
    v = VoxelSet(0.1)
    pts = np.array([[0, 0, 0], [0.01, 0.01, 0.01], [1, 1, 1]], float)
    assert v.add_new(pts).sum() == 2
    assert v.add_new(pts).sum() == 0
    assert v.add_new(np.array([[2.0, 2, 2]])).sum() == 1 and len(v) == 3


def test_map_resolution_for_phone_and_glasses_aspects():
    assert map_resolution(1280, 720) == (518, 294)
    assert map_resolution(720, 1280) == (294, 518)
    assert map_resolution(504, 896) == (294, 518)
    assert map_resolution(640, 480) == (518, 392)
