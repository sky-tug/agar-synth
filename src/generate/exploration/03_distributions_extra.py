#!/usr/bin/env python3
"""
03 -- follow-up questions left open by 02: box squareness, per-colony touching
rate, crowding vs size, per-class radial difference, plate composition and the
distribution family of the colony count.

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "out"

rows = list(np.load(OUT / "rows.npy", allow_pickle=True))
plates = {p['sid']: p for p in np.load(OUT / "plates.npy", allow_pickle=True)}

print("=== A. ARE THE BOXES SQUARE? ===")
square = sum(1 for r in rows if r['w']==r['h'])
print(f"  boxes with w == h: {square}/{len(rows)}  ({100*square/len(rows):.1f}%)")
if square < len(rows):
    ar = [(r['sid'],r['cls'],r['w'],r['h']) for r in rows if r['w']!=r['h']][:10]
    print("  examples:", ar)

print("\n=== B. TOUCHING COLONY RATE (per colony, not per pair) ===")
touching=0; total=0; ov=[]
for sid in plates:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs]); S=np.array([(r['w']+r['h'])/4 for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0],P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    RS=S[:,None]+S[None,:]
    t=(D<RS)
    touching += t.any(1).sum(); total += len(rs)
    # interpenetration depth (over the overlapping pairs)
    ii,jj=np.where(np.triu(t,1))
    for a,b in zip(ii,jj):
        ov.append(1 - D[a,b]/RS[a,b])
print(f"  colonies touching at least one neighbour: {touching}/{total}  ({100*touching/total:.1f}%)")
if ov:
    ov=np.array(ov)
    print(f"  overlap depth (0=tangential, 1=fully nested): median={np.median(ov):.2f} p90={np.percentile(ov,90):.2f} max={ov.max():.2f}")

print("\n=== C. DOES A CROWDED PLATE MEAN SMALL COLONIES? ===")
for cls in ["S.aureus","E.coli"]:
    xs=[];ys=[]
    for sid,p in plates.items():
        v=[(r['w']+r['h'])/2 for r in rows if r['sid']==sid and r['cls']==cls]
        if len(v)>=5: xs.append(len([r for r in rows if r['sid']==sid])); ys.append(np.median(v))
    if len(xs)>=4:
        rho,pv=stats.spearmanr(xs,ys)
        print(f"  {cls:<14} n_plates={len(xs)}  Spearman(n_colonies, median_size)={rho:+.2f} p={pv:.2f}")
        print(f"                 n={xs}  median size={[round(y,1) for y in ys]}")

print("\n=== D. PER-CLASS RADIAL DIFFERENCE ===")
byc=defaultdict(list)
for r in rows: byc[r['cls']].append(r['r_norm'])
for c,v in sorted(byc.items()):
    v=np.array(v); print(f"  {c:<14} n={len(v):4d}  r_norm median={np.median(v):.3f}  r^2 mean={np.mean(v**2):.3f}")

print("\n=== E. CLASS COMBINATION PER PLATE ===")
for sid,p in sorted(plates.items()):
    cnt=Counter(r['cls'] for r in rows if r['sid']==sid)
    print(f"  {sid}  n={p['n']:3d}  {dict(cnt)}")

print("\n=== F. DISTRIBUTION FAMILY FOR THE COUNT ===")
ns=np.array([p['n'] for p in plates.values()])
print(f"  ln(n): mean={np.log(ns).mean():.2f} std={np.log(ns).std(ddof=1):.2f}")
print(f"  lognormal back-transform: median={math.exp(np.log(ns).mean()):.0f}, "
      f"p10={math.exp(np.log(ns).mean()-1.28*np.log(ns).std(ddof=1)):.0f}, "
      f"p90={math.exp(np.log(ns).mean()+1.28*np.log(ns).std(ddof=1)):.0f}")
print(f"  WARNING: n=10 plates. This picks a FAMILY only, not the parameters.")
