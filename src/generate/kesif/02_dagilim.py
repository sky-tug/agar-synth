import numpy as np, math
from collections import defaultdict
rows = list(np.load("/home/claude/faz3/rows.npy", allow_pickle=True))
plak = {p['sid']: p for p in np.load("/home/claude/faz3/plak.npy", allow_pickle=True)}

print("=== 1. YARICAP DOGRULAMASI (r_norm > 1 varsa cember yanlis) ===")
for sid in plak:
    rr = [r['r_norm'] for r in rows if r['sid']==sid]
    print(f"{sid}  n={len(rr):3d}  r_norm maks={max(rr):.3f}  p95={np.percentile(rr,95):.3f}")

print("\n=== 2. RADYAL DAGILIM (tum koloniler) ===")
rn = np.array([r['r_norm'] for r in rows])
print(f"toplam koloni: {len(rn)}")
for q in [0,10,25,50,75,90,95,99,100]:
    print(f"  p{q:<3d} = {np.percentile(rn,q):.3f}")
# Uniform-on-disk beklentisi: P(r_norm<t) = t^2  -> r_norm^2 ~ Uniform(0,1)
u = rn**2
print(f"\n  r_norm^2 ortalama = {u.mean():.3f}  (duzgun dagilimda 0.500 beklenir)")
print(f"  r_norm^2 medyan   = {np.median(u):.3f}  (duzgun dagilimda 0.500)")
from scipy import stats
ks = stats.kstest(u, 'uniform')
print(f"  KS testi (duzgun disk H0): D={ks.statistic:.3f}  p={ks.pvalue:.2e}")

print("\n  halka bazli yogunluk (esit ALANLI 5 halka, duzgunse hepsi ~%20):")
kenar = [math.sqrt(i/5) for i in range(6)]
for i in range(5):
    m = ((rn>=kenar[i]) & (rn<kenar[i+1])).sum()
    print(f"   r {kenar[i]:.2f}-{kenar[i+1]:.2f}: {m:4d}  %{100*m/len(rn):.1f}")

print("\n=== 3. ACISAL DAGILIM (duzgunluk) ===")
th = np.array([r['theta'] for r in rows])
ks2 = stats.kstest((th+math.pi)/(2*math.pi), 'uniform')
print(f"  KS (duzgun aci H0): D={ks2.statistic:.3f}  p={ks2.pvalue:.2f}")

print("\n=== 4. KOLONI SAYISI ===")
ns = np.array([p['n'] for p in plak.values()])
print(f"  n: {sorted(ns)}   medyan={np.median(ns):.0f} ort={ns.mean():.1f}")

print("\n=== 5. BOYUT (sinif bazli, px) ===")
byc = defaultdict(list)
for r in rows: byc[r['sinif']].append((r['w']+r['h'])/2)
for c, v in sorted(byc.items()):
    v=np.array(v)
    print(f"  {c:<14} n={len(v):4d}  medyan={np.median(v):6.1f}  "
          f"p10={np.percentile(v,10):5.1f} p90={np.percentile(v,90):6.1f}  "
          f"std/medyan={v.std()/np.median(v):.2f}")
print("\n  en/boy orani (w/h):")
for c in sorted(byc):
    ar=np.array([r['w']/r['h'] for r in rows if r['sinif']==c])
    print(f"  {c:<14} medyan={np.median(ar):.2f}  p10={np.percentile(ar,10):.2f} p90={np.percentile(ar,90):.2f}")

print("\n=== 6. KOLONILER BIRBIRINE DEGIYOR MU ===")
tum_nn=[]; cakisan=0; toplam_cift=0; degiyor=0
for sid in plak:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs])
    S=np.array([(r['w']+r['h'])/4 for r in rs])  # yaricap ~ yari-kenar
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1])
    np.fill_diagonal(D, 1e9)
    nn=D.min(1); tum_nn+= list(nn)
    # merkez mesafesi < yaricaplar toplami -> degiyor/cakisiyor
    RS=S[:,None]+S[None,:]
    m=np.triu(D<RS,1)
    degiyor+=m.sum(); toplam_cift+=len(rs)*(len(rs)-1)//2
    # kutu IoU>0
tum_nn=np.array(tum_nn)
print(f"  en yakin komsu mesafesi (px): medyan={np.median(tum_nn):.0f} "
      f"p5={np.percentile(tum_nn,5):.0f} p25={np.percentile(tum_nn,25):.0f}")
print(f"  degen/cakisan koloni cifti: {degiyor} / {toplam_cift} cift  (%{100*degiyor/toplam_cift:.2f})")
# normalize: nn / kendi capina
nn_norm=[]
for sid in plak:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs]); S=np.array([(r['w']+r['h'])/2 for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    nn_norm += list(D.min(1)/S)
nn_norm=np.array(nn_norm)
print(f"  en yakin komsu / koloni capi: medyan={np.median(nn_norm):.2f} "
      f"p5={np.percentile(nn_norm,5):.2f} p10={np.percentile(nn_norm,10):.2f}  "
      f"(<1.0 = degiyor: %{100*(nn_norm<1).mean():.1f})")

print("\n=== 7. KUMELENME (yerlesim rastgele mi?) ===")
# Clark-Evans: gozlenen ort. NN / rastgele beklenen (0.5/sqrt(yogunluk))
for sid in sorted(plak):
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<5: continue
    p=plak[sid]; A=math.pi*p['R']**2
    P=np.array([[r['x'],r['y']] for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    obs=D.min(1).mean(); exp=0.5/math.sqrt(len(rs)/A)
    print(f"  {sid}  n={len(rs):3d}  CE={obs/exp:.2f}   "
          f"{'kumelenmis' if obs/exp<0.9 else ('duzenli' if obs/exp>1.1 else 'rastgele')}")
