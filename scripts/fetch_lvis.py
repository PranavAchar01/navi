"""Fetch the finder test set: LVIS v1 val images for the watch-list items (CC BY 4.0 labels,
COCO images). Needs data/lvis/lvis_v1_val.json from
https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip ("glasses case" has no LVIS class)."""

import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path.home() / "helloworld/meta-sight/data/lvis"
d = json.load(open(root / "lvis_v1_val.json"))
CLS = {
    "keys": 605,
    "wallet": 1156,
    "phone": 230,
    "remote control": 881,
    "backpack": 34,
    "water bottle": 1162,
}
imgs = {i["id"]: i for i in d["images"]}
pos = {c: set() for c in CLS}
anns = {}
for a in d["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)
    for lab, cid in CLS.items():
        if a["category_id"] == cid:
            pos[lab].add(a["image_id"])
random.seed(0)
pick = set()
for lab, cid in CLS.items():
    p = sorted(pos[lab])
    random.shuffle(p)
    pick |= set(p[:60])
    neg = sorted(i for i, im in imgs.items() if cid in im["neg_category_ids"])
    random.shuffle(neg)
    pick |= set(neg[:60])
out = []
for i in sorted(pick):
    im = imgs[i]
    out.append(
        dict(
            id=i,
            url=im["coco_url"],
            w=im["width"],
            h=im["height"],
            neg=im["neg_category_ids"],
            nex=im["not_exhaustive_category_ids"],
            boxes=[
                (a["category_id"], a["bbox"])
                for a in anns.get(i, [])
                if a["category_id"] in CLS.values()
            ],
        )
    )
(root / "img").mkdir(exist_ok=True)
json.dump(dict(classes=CLS, images=out), open(root / "set.json", "w"))
print(len(out), "images", {k: len(v) for k, v in pos.items()})


def get(o):
    p = root / "img" / f"{o['id']}.jpg"
    if not p.exists():
        urllib.request.urlretrieve(o["url"], p)


with ThreadPoolExecutor(16) as ex:
    list(ex.map(get, out))
