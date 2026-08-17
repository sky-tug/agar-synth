#!/usr/bin/env python3
"""
01 -- estimate the plate circle of every demo image and cache the colony table.

Writes rows.npy (one record per colony) and plates.npy (one record per plate);
the other exploration scripts read those two caches instead of re-detecting the
circle every time.

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import json, math
from pathlib import Path

import numpy as np, cv2

DATA = Path(__file__).resolve().parents[3] / "data" / "AGAR_representative" / "lower-resolution"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def plate_circle(path):
    """Estimate the plate disc from the image: Hough, with thresholding as fallback."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    small = cv2.resize(img, (w//4, h//4))
    c = cv2.HoughCircles(cv2.medianBlur(small,5), cv2.HOUGH_GRADIENT, dp=1,
                         minDist=small.shape[0], param1=100, param2=30,
                         minRadius=int(small.shape[0]*0.30),
                         maxRadius=int(small.shape[0]*0.52))
    if c is not None:
        x,y,r = c[0][0]
        return float(x*4), float(y*4), float(r*4), (w,h), "hough"
    # fallback: centre of mass of the bright region
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ys, xs = np.nonzero(th)
    x, y = xs.mean()*4, ys.mean()*4
    r = math.sqrt(th.sum()/255/math.pi)*4
    return float(x), float(y), float(r), (w,h), "otsu"

rows=[]
plates=[]
for jf in sorted(DATA.glob("*.json")):
    sid = jf.stem
    j = json.load(open(jf))
    im = str(DATA / f"{sid}.jpg")
    cx, cy, R, (W,H), method = plate_circle(im)
    plates.append(dict(sid=sid, cx=cx, cy=cy, R=R, W=W, H=H, method=method,
                       n=j["colonies_number"], classes="+".join(j["classes"])))
    for L in j["labels"]:
        bx = L["x"] + L["width"]/2.0
        by = L["y"] + L["height"]/2.0
        rows.append(dict(sid=sid, cls=L["class"], x=bx, y=by,
                         w=L["width"], h=L["height"],
                         r_norm=math.hypot(bx-cx, by-cy)/R,
                         theta=math.atan2(by-cy, bx-cx)))
np.save(OUT / "rows.npy", np.array(rows, dtype=object))
np.save(OUT / "plates.npy", np.array(plates, dtype=object))

print("=== PLATE CIRCLE ===")
for p in plates:
    print(f"{p['sid']}  {p['W']}x{p['H']}  center=({p['cx']:.0f},{p['cy']:.0f})  R={p['R']:.0f}  "
          f"R/W={p['R']/p['W']:.3f}  [{p['method']}]  n={p['n']}  {p['classes']}")
