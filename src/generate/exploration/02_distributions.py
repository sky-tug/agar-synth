#!/usr/bin/env python3
"""
02 -- the core layout distributions: radial, angular, count, size, touching,
clustering. Reads the caches written by 01_plate_circle.py.

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

rows = list(np.load(OUT / "rows.npy", allow_pickle=True))
plates = {p['sid']: p for p in np.load(OUT / "plates.npy", allow_pickle=True)}

print("=== 1. RADIUS CHECK (if r_norm > 1 the circle is wrong) ===")
for sid in plates:
    rr = [r['r_norm'] for r in rows if r['sid']==sid]
    print(f"{sid}  n={len(rr):3d}  r_norm max={max(rr):.3f}  p95={np.percentile(rr,95):.3f}")

print("\n=== 2. RADIAL DISTRIBUTION (all colonies) ===")
rn = np.array([r['r_norm'] for r in rows])
print(f"total colonies: {len(rn)}")
for q in [0,10,25,50,75,90,95,99,100]:
    print(f"  p{q:<3d} = {np.percentile(rn,q):.3f}")
# Uniform-on-disk expectation: P(r_norm<t) = t^2  -> r_norm^2 ~ Uniform(0,1)
u = rn**2
print(f"\n  r_norm^2 mean   = {u.mean():.3f}  (0.500 expected under a uniform distribution)")
print(f"  r_norm^2 median = {np.median(u):.3f}  (0.500 under a uniform distribution)")
from scipy import stats
ks = stats.kstest(u, 'uniform')
print(f"  KS test (uniform disc H0): D={ks.statistic:.3f}  p={ks.pvalue:.2e}")

print("\n  density per ring (5 rings of EQUAL AREA, ~20% each if uniform):")
edges = [math.sqrt(i/5) for i in range(6)]
for i in range(5):
    m = ((rn>=edges[i]) & (rn<edges[i+1])).sum()
    print(f"   r {edges[i]:.2f}-{edges[i+1]:.2f}: {m:4d}  {100*m/len(rn):.1f}%")

print("\n=== 3. ANGULAR DISTRIBUTION (uniformity) ===")
th = np.array([r['theta'] for r in rows])
ks2 = stats.kstest((th+math.pi)/(2*math.pi), 'uniform')
print(f"  KS (uniform angle H0): D={ks2.statistic:.3f}  p={ks2.pvalue:.2f}")

print("\n=== 4. COLONY COUNT ===")
ns = np.array([p['n'] for p in plates.values()])
print(f"  n: {sorted(ns)}   median={np.median(ns):.0f} mean={ns.mean():.1f}")

print("\n=== 5. SIZE (per class, px) ===")
byc = defaultdict(list)
for r in rows: byc[r['cls']].append((r['w']+r['h'])/2)
for c, v in sorted(byc.items()):
    v=np.array(v)
    print(f"  {c:<14} n={len(v):4d}  median={np.median(v):6.1f}  "
          f"p10={np.percentile(v,10):5.1f} p90={np.percentile(v,90):6.1f}  "
          f"std/median={v.std()/np.median(v):.2f}")
print("\n  aspect ratio (w/h):")
for c in sorted(byc):
    ar=np.array([r['w']/r['h'] for r in rows if r['cls']==c])
    print(f"  {c:<14} median={np.median(ar):.2f}  p10={np.percentile(ar,10):.2f} p90={np.percentile(ar,90):.2f}")

print("\n=== 6. DO THE COLONIES TOUCH EACH OTHER ===")
all_nn=[]; overlapping=0; total_pairs=0; touching=0
for sid in plates:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs])
    S=np.array([(r['w']+r['h'])/4 for r in rs])  # radius ~ half edge
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1])
    np.fill_diagonal(D, 1e9)
    nn=D.min(1); all_nn+= list(nn)
    # centre distance < sum of radii -> touching/overlapping
    RS=S[:,None]+S[None,:]
    m=np.triu(D<RS,1)
    touching+=m.sum(); total_pairs+=len(rs)*(len(rs)-1)//2
    # box IoU>0
all_nn=np.array(all_nn)
print(f"  nearest neighbour distance (px): median={np.median(all_nn):.0f} "
      f"p5={np.percentile(all_nn,5):.0f} p25={np.percentile(all_nn,25):.0f}")
print(f"  touching/overlapping colony pairs: {touching} / {total_pairs} pairs  ({100*touching/total_pairs:.2f}%)")
# normalized: nn / own diameter
nn_norm=[]
for sid in plates:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs]); S=np.array([(r['w']+r['h'])/2 for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    nn_norm += list(D.min(1)/S)
nn_norm=np.array(nn_norm)
print(f"  nearest neighbour / colony diameter: median={np.median(nn_norm):.2f} "
      f"p5={np.percentile(nn_norm,5):.2f} p10={np.percentile(nn_norm,10):.2f}  "
      f"(<1.0 = touching: {100*(nn_norm<1).mean():.1f}%)")

print("\n=== 7. CLUSTERING (is the layout random?) ===")
# Clark-Evans: observed mean NN / random expectation (0.5/sqrt(density))
for sid in sorted(plates):
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<5: continue
    p=plates[sid]; A=math.pi*p['R']**2
    P=np.array([[r['x'],r['y']] for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    obs=D.min(1).mean(); exp=0.5/math.sqrt(len(rs)/A)
    print(f"  {sid}  n={len(rs):3d}  CE={obs/exp:.2f}   "
          f"{'clustered' if obs/exp<0.9 else ('regular' if obs/exp>1.1 else 'random')}")
