import numpy as np, math, json, glob, os
from collections import defaultdict, Counter
from scipy import stats
rows = list(np.load("/home/claude/faz3/rows.npy", allow_pickle=True))
plak = {p['sid']: p for p in np.load("/home/claude/faz3/plak.npy", allow_pickle=True)}

print("=== A. KUTULAR KARE MI? ===")
kare = sum(1 for r in rows if r['w']==r['h'])
print(f"  w == h olan kutu: {kare}/{len(rows)}  (%{100*kare/len(rows):.1f})")
if kare < len(rows):
    ar = [(r['sid'],r['sinif'],r['w'],r['h']) for r in rows if r['w']!=r['h']][:10]
    print("  ornekler:", ar)

print("\n=== B. DEGEN KOLONI ORANI (cift degil, koloni bazli) ===")
degen=0; tum=0; ov=[]
for sid in plak:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs]); S=np.array([(r['w']+r['h'])/4 for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0],P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    RS=S[:,None]+S[None,:]
    t=(D<RS)
    degen += t.any(1).sum(); tum += len(rs)
    # ic ice gecme derinligi (ortusen ciftlerde)
    ii,jj=np.where(np.triu(t,1))
    for a,b in zip(ii,jj):
        ov.append(1 - D[a,b]/RS[a,b])
print(f"  en az bir komsusuna degen koloni: {degen}/{tum}  (%{100*degen/tum:.1f})")
if ov:
    ov=np.array(ov)
    print(f"  ortusme derinligi (0=tegetsel, 1=tam ic ice): medyan={np.median(ov):.2f} p90={np.percentile(ov,90):.2f} maks={ov.max():.2f}")

print("\n=== C. KALABALIK PLAK -> KUCUK KOLONI MU? ===")
for sinif in ["S.aureus","E.coli"]:
    xs=[];ys=[]
    for sid,p in plak.items():
        v=[(r['w']+r['h'])/2 for r in rows if r['sid']==sid and r['sinif']==sinif]
        if len(v)>=5: xs.append(len([r for r in rows if r['sid']==sid])); ys.append(np.median(v))
    if len(xs)>=4:
        rho,pv=stats.spearmanr(xs,ys)
        print(f"  {sinif:<14} plak sayisi={len(xs)}  Spearman(n_koloni, medyan_boyut)={rho:+.2f} p={pv:.2f}")
        print(f"                 n={xs}  medyan boyut={[round(y,1) for y in ys]}")

print("\n=== D. SINIF BAZLI RADYAL FARK ===")
byc=defaultdict(list)
for r in rows: byc[r['sinif']].append(r['r_norm'])
for c,v in sorted(byc.items()):
    v=np.array(v); print(f"  {c:<14} n={len(v):4d}  r_norm medyan={np.median(v):.3f}  r^2 ort={np.mean(v**2):.3f}")

print("\n=== E. PLAK BASINA SINIF KOMBINASYONU ===")
for sid,p in sorted(plak.items()):
    say=Counter(r['sinif'] for r in rows if r['sid']==sid)
    print(f"  {sid}  n={p['n']:3d}  {dict(say)}")

print("\n=== F. SAYIM DAGILIMI ICIN DAGILIM AILESI ===")
ns=np.array([p['n'] for p in plak.values()])
print(f"  ln(n): ort={np.log(ns).mean():.2f} std={np.log(ns).std(ddof=1):.2f}")
print(f"  lognormal geri-donusum: medyan={math.exp(np.log(ns).mean()):.0f}, "
      f"p10={math.exp(np.log(ns).mean()-1.28*np.log(ns).std(ddof=1)):.0f}, "
      f"p90={math.exp(np.log(ns).mean()+1.28*np.log(ns).std(ddof=1)):.0f}")
print(f"  UYARI: n=10 plak. Bu sadece bir aile secimi, parametre degil.")
