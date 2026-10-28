"""Build the static project page (site/index.html) from the saved benchmark results.

Numbers come from bench/results/map.jsonl (TUM walks, tags BEFORE and NOW) and the LVIS finder
detections in data/lvis; path plots from the saved trajectories. Only public datasets appear.
Prose lives in site/copy.json so it can go through the text checks on its own.
"""

import json
import sys
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_finder import LV, score  # noqa: E402

BEFORE, NOW = "nl7", "k1"
FIND_BEFORE, FIND_NOW = "s640", "tiled"
OUT = ROOT / "site"
WALKS = {
    "freiburg1_desk": "Desk, close up",
    "freiburg1_360": "Turning on the spot",
    "freiburg1_room": "Round a room",
    "freiburg2_desk": "Round a desk, slowly",
    "freiburg3_long_office_household": "Office loop",
    "freiburg3_nostructure_texture_near_withloop": "Flat floor posters",
    "freiburg2_pioneer_slam": "Robot across a hall",
}
PATHS = ["freiburg3_long_office_household", "freiburg2_pioneer_slam"]


def rows():
    out = {}
    for line in open(ROOT / "bench/results/map.jsonl"):
        d = json.loads(line)
        for k in WALKS:
            if d.get("source", "").endswith(k) and d["tag"] in (BEFORE, NOW):
                out[(k, d["tag"])] = d  # later rows win
    return out


def bars(title_id, cats, a, b, unit, fmt):
    """Grouped horizontal bars, BEFORE (slot 2) over NOW (slot 1), labelled at the bar end."""
    row_h, bar_h, gap, left, width = 46, 14, 2, 190, 640
    top = 8
    peak = max(max(a), max(b))
    step = next(m * 10**e for e in range(-2, 4) for m in (1, 2, 5) if m * 10**e * 4 >= peak)
    vmax = step * 4.6  # room for the value label past the longest bar
    h = top + row_h * len(cats) + 24
    sx = lambda v: left + (width - left) * v / vmax  # noqa: E731
    g = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-labelledby="{title_id}" class="chart">']
    for i in range(5):
        v = step * i
        x = sx(v)
        g.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - 22}" class="grid"/>')
        g.append(f'<text x="{x:.1f}" y="{h - 6}" class="tick" text-anchor="middle">{fmt(v)}</text>')
    for i, c in enumerate(cats):
        y = top + i * row_h + 6
        at = f'x="{left - 12}" y="{y + bar_h + 4}" class="cat" text-anchor="end"'
        g.append(f"<text {at}>{escape(c)}</text>")
        for j, (v, cls, name) in enumerate(((a[i], "s2", "Before"), (b[i], "s1", "Now"))):
            yy = y + j * (bar_h + gap)
            w = max(sx(v) - left, 2)
            tip = f"{c}: {name} {fmt(v)}{unit}"
            g.append(
                f'<g class="mark"><title>{escape(tip)}</title>'
                f'<rect x="{left}" y="{yy - 3}" width="{width - left}" '
                f'height="{bar_h + 4}" class="hit"/>'
                f'<path d="M{left},{yy} h{w - 4} a4,4 0 0 1 4,4 v{bar_h - 8} '
                f'a4,4 0 0 1 -4,4 h-{w - 4} z" class="{cls}"/>'
                f'<text x="{left + w + 6:.1f}" y="{yy + bar_h - 3}" class="val">'
                f"{fmt(v)}{unit}</text></g>"
            )
    g.append("</svg>")
    return "".join(g)


def path_plot(npz_before, npz_now, label_id):
    """Top-down: the true path in ink, each estimate in its series colour, two panels."""
    panels = []
    zs = [np.load(npz_before), np.load(npz_now)]
    gt = zs[1]["gt"]
    c = gt.mean(0)
    _, _, vt = np.linalg.svd(gt - c)
    ax = vt[:2]  # the two directions the walk spreads along
    pts = [z["gt"] for z in zs] + [z["est"] for z in zs]
    proj = [(p - c) @ ax.T for p in pts]
    allp = np.concatenate(proj)
    lo, hi = allp.min(0), allp.max(0)
    span = max(hi - lo) * 1.08
    mid = (lo + hi) / 2
    size = 300

    def poly(p):
        q = (p - mid) / span * (size - 20) + size / 2
        return " ".join(f"{x:.1f},{size - y:.1f}" for x, y in q)

    for k, (cls, name) in enumerate((("s2", "Before"), ("s1", "Now"))):
        z = zs[k]
        err = float(np.sqrt(np.mean(np.sum((z["est"] - z["gt"]) ** 2, 1)))) * 100
        panels.append(
            f'<figure class="panel"><figcaption><span class="key {cls}"></span>{name}: '
            f"{err:.0f} cm off</figcaption>"
            f'<svg viewBox="0 0 {size} {size}" role="img" aria-labelledby="{label_id}">'
            f'<polyline points="{poly(proj[k])}" class="truth"/>'
            f'<polyline points="{poly(proj[2 + k])}" class="est {cls}"/></svg></figure>'
        )
    return "".join(panels)


def main():
    copy = json.load(open(OUT / "copy.json"))
    r = rows()
    walks = [k for k in WALKS if (k, BEFORE) in r and (k, NOW) in r]
    a = [100 * r[(k, BEFORE)]["ate_m"] for k in walks]
    b = [100 * r[(k, NOW)]["ate_m"] for k in walks]
    st = json.load(open(LV / "set.json"))
    fb = score(st, json.load(open(LV / f"dets-{FIND_BEFORE}.json")))
    fn = score(st, json.load(open(LV / f"dets-{FIND_NOW}.json")))
    items = list(st["classes"])
    pa = [fb[i]["ap50"] for i in items]
    pb = [fn[i]["ap50"] for i in items]
    res = ROOT / "bench/results"
    plots = {
        k: path_plot(
            res / f"traj-rgbd_dataset_{k}-{BEFORE}.npz",
            res / f"traj-rgbd_dataset_{k}-{NOW}.npz",
            f"p-{k}",
        )
        for k in PATHS
    }
    table_map = "".join(
        f"<tr><td>{escape(WALKS[k])}</td><td>{r[(k, 'nl7')]['path_m']:.1f} m</td>"
        f"<td>{x:.1f} cm</td><td>{y:.1f} cm</td></tr>"
        for k, x, y in zip(walks, a, b, strict=True)
    )
    table_find = "".join(
        f"<tr><td>{escape(i)}</td><td>{fb[i]['n_gt']}</td><td>{x:.2f}</td><td>{y:.2f}</td></tr>"
        for i, x, y in zip(items, pa, pb, strict=True)
    )
    fill = {
        "MAP_BARS": bars("t-map", [WALKS[k] for k in walks], a, b, " cm", lambda v: f"{v:.0f}"),
        "FIND_BARS": bars("t-find", items, pa, pb, "", lambda v: f"{v:.2f}"),
        "MAP_MEAN_BEFORE": f"{np.mean(a):.0f}",
        "MAP_MEAN_NOW": f"{np.mean(b):.0f}",
        "FIND_MEAN_BEFORE": f"{np.mean(pa):.2f}",
        "FIND_MEAN_NOW": f"{np.mean(pb):.2f}",
        "N_PHOTOS": str(len(st["images"])),
        "N_WALKS": str(len(walks)),
        "PATH_OFFICE": plots[PATHS[0]],
        "PATH_HALL": plots[PATHS[1]],
        "TABLE_MAP": table_map,
        "TABLE_FIND": table_find,
    }
    html = (OUT / "template.html").read_text()
    for k, v in {**copy, **fill}.items():
        html = html.replace("{{" + k + "}}", v if k in fill else escape(v))
    assert "{{" not in html, html[html.index("{{") : html.index("{{") + 40]
    (OUT / "public").mkdir(exist_ok=True)
    (OUT / "public/index.html").write_text(html)
    print(OUT / "public/index.html")


if __name__ == "__main__":
    main()
