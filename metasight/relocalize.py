"""Place one frame in a fixed home: match it to the saved views, solve PnP on their points.

Each saved view keeps its room points per pixel (HomeView.pts, every STRIDE-th pixel), so a SIFT
match between a live frame and a view is a 2D to 3D pair, and PnP RANSAC over those pairs gives the
camera in room metres. On the bedroom this lands a frame 6 cm from the dense run's camera on the
walk the views came from, 13 cm on a walk two days earlier (median of the frames it places), where
a MapAnything try on eight frames landed 0.2 to 2.7 m off.
"""

from __future__ import annotations

import cv2
import numpy as np

from .loops import sift_features

STRIDE = 2  # memory.STRIDE: the pixel step of HomeView.pts
MIN_INLIERS = 40  # with 40, 2 of 46 and 1 of 35 placed frames landed over 0.5 m off (2 walks)
RATIO = 0.75  # Lowe's ratio test
REPROJ_PX = 4.0
TOP_VIEWS = 4  # views a frame is matched against, best looking first


def view_k(pts: np.ndarray, mask: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """The pinhole K that best maps a view's own points to their pixels."""
    v, u = np.nonzero(mask)
    c = (pts[v, u].astype(np.float64) - pose[:3, 3]) @ pose[:3, :3]
    x, y = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
    fx, cx = np.linalg.lstsq(np.c_[x, np.ones_like(x)], u * STRIDE, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y, np.ones_like(y)], v * STRIDE, rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class RoomMatcher:
    """SIFT features of the saved views, kept where the view has a room point."""

    def __init__(self, bgrs: list[np.ndarray], views: list) -> None:
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.shape = bgrs[0].shape[:2] if bgrs else (0, 0)  # the views' frame size
        self.des: list[np.ndarray | None] = []
        self.xyz: list[np.ndarray] = []
        for bgr, v in zip(bgrs, views, strict=True):
            xy, des = sift_features(bgr)
            if des is None:
                self.des.append(None)
                self.xyz.append(np.empty((0, 3), np.float32))
                continue
            h, w = v.mask.shape
            gv = np.clip(np.round(xy[:, 1] / STRIDE).astype(int), 0, h - 1)
            gu = np.clip(np.round(xy[:, 0] / STRIDE).astype(int), 0, w - 1)
            ok = v.mask[gv, gu]
            self.des.append(des[ok])
            self.xyz.append(v.pts[gv[ok], gu[ok]].astype(np.float32))
        self.k = np.median([view_k(v.pts, v.mask, v.pose) for v in views], axis=0)
        # Every view's features in one index, to rank the views for a frame that comes with no
        # order: the views holding most of its nearest features look most like it.
        self.indexed = [i for i, d in enumerate(self.des) if d is not None and len(d)]
        self.flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
        if self.indexed:
            self.flann.add([self.des[i] for i in self.indexed])
            self.flann.train()
        # Horizontal field of view, degrees. On the bedroom it is 52.8, and a sweep of the focal
        # on another walk's PnP inliers peaks at 52 too (the glasses profile's 44.8 is the
        # emulator's crop, not these frames).
        self.hfov = float(np.degrees(2 * np.arctan(self.k[0, 2] / self.k[0, 0])))

    def locate(
        self, bgr: np.ndarray, order: list[int] | None = None
    ) -> tuple[np.ndarray, int] | None:
        """The frame's camera pose in the room (4x4, camera to room) and its PnP inlier count,
        matched against the first TOP_VIEWS of `order` (view indices, best looking first; by
        default the views that share most of its features)."""
        q, des = sift_features(bgr)
        if des is None or len(q) < 2:
            return None
        if order is None:
            if not self.indexed:
                return None
            near = [self.indexed[m.imgIdx] for m in self.flann.match(des)]
            order = list(np.argsort(-np.bincount(near, minlength=len(self.des))))
        p2, p3 = [], []
        for i in order[:TOP_VIEWS]:
            vd = self.des[i]
            if vd is None or len(vd) < 2:
                continue
            for pair in self.bf.knnMatch(des, vd, k=2):
                if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                    a = pair[0]
                    p2.append(q[a.queryIdx])
                    p3.append(self.xyz[i][a.trainIdx])
        if len(p2) < MIN_INLIERS:
            return None
        ok, rv, tv, inl = cv2.solvePnPRansac(
            np.array(p3, np.float64),
            np.array(p2, np.float64),
            self.k,
            None,
            reprojectionError=REPROJ_PX,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < MIN_INLIERS:
            return None
        r = cv2.Rodrigues(rv)[0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r.T, (-r.T @ tv).ravel()
        return pose, len(inl)
