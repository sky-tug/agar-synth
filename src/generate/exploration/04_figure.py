#!/usr/bin/env python3
"""
04 -- the summary figure of the layout exploration: 10 plate maps plus three
panels (radial density, touching, Clark-Evans).

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import math
from pathlib import Path

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "out"

rows=list(np.load(OUT / "rows.npy",allow_pickle=True))
plates={p['sid']:p for p in np.load(OUT / "plates.npy",allow_pickle=True)}

SPECIES=["S.aureus","B.subtilis","P.aeruginosa","E.coli","C.albicans"]
C={s:c for s,c in zip(SPECIES,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"])}
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#8a8a85"

fig=plt.figure(figsize=(15,9.2), facecolor=SURF)
gs=fig.add_gridspec(3,5,height_ratios=[1,1,0.95],hspace=0.34,wspace=0.14,
                    left=0.045,right=0.985,top=0.885,bottom=0.075)

sids=sorted(plates,key=lambda s:plates[s]['n'])
for i,sid in enumerate(sids):
    ax=fig.add_subplot(gs[i//5,i%5]); p=plates[sid]
    ax.add_patch(Circle((p['cx'],p['cy']),p['R'],fill=False,ec=MUT,lw=1.2))
    ax.add_patch(Circle((p['cx'],p['cy']),p['R']*0.90,fill=False,ec=MUT,lw=1,ls=(0,(3,3))))
    for r in [x for x in rows if x['sid']==sid]:
        rr=max((r['w']+r['h'])/4, 16)
        ax.add_patch(Circle((r['x'],r['y']),rr,fc=C[r['cls']],
                            ec=SURF,lw=1.2,alpha=0.9))
    ax.set_xlim(0,2048); ax.set_ylim(2048,0); ax.set_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"{sid}  -  n={p['n']}",fontsize=9,color=INK2,pad=4)
    ax.set_facecolor(SURF)

def style(ax,t,xl):
    ax.set_facecolor(SURF); ax.set_title(t,fontsize=10,color=INK,pad=8,loc="left")
    ax.set_xlabel(xl,fontsize=8.5,color=INK2)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color(MUT); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2,labelsize=8,length=3)
    ax.grid(axis="y",color="#e6e5e0",lw=0.8); ax.set_axisbelow(True)

# panel 1: density over rings of equal area
ax=fig.add_subplot(gs[2,0:2]); style(ax,"Radial density -- 5 rings of equal area","fraction of the plate radius  (r/R)")
rn=np.array([r['r_norm'] for r in rows]); edges=[math.sqrt(i/5) for i in range(6)]
share=[100*((rn>=edges[i])&(rn<edges[i+1])).sum()/len(rn) for i in range(5)]
ax.bar(range(5),share,color="#2a78d6",width=0.72)
ax.axhline(20,color=INK2,lw=1.6,ls=(0,(4,3)))
ax.text(4.42,26.5,"uniform distribution = 20%",fontsize=8,color=INK2,ha="right")
ax.set_xticks(range(5)); ax.set_xticklabels([f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(5)],fontsize=7.5)
ax.set_ylabel("% of colonies",fontsize=8.5,color=INK2)
for i,v in enumerate(share): ax.text(i,v+0.6,f"{v:.1f}%",ha="center",fontsize=8,color=INK)
ax.set_ylim(0,29)

# panel 2: nearest neighbour / diameter
ax=fig.add_subplot(gs[2,2:4]); style(ax,"Do the colonies touch each other?","nearest neighbour distance / colony diameter")
nn=[]
for sid in plates:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<2: continue
    P=np.array([[r['x'],r['y']] for r in rs]); S=np.array([(r['w']+r['h'])/2 for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0],P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    nn+=list(D.min(1)/S)
nn=np.array(nn)
ax.hist(np.clip(nn,0,4),bins=np.arange(0,4.05,0.2),color="#1baf7a",ec=SURF,lw=1.4)
ax.text(3.9,2,">= 4",fontsize=7.5,color=INK2,ha="center",va="bottom",rotation=90)
ax.axvline(1.0,color="#eb6834",lw=2)
ax.text(1.08,ax.get_ylim()[1]*0.88,f"left of this = touching\n{100*(nn<1).mean():.0f}% of colonies",
        fontsize=8.5,color="#eb6834")
ax.set_ylabel("colony count",fontsize=8.5,color=INK2)

# panel 3: Clark-Evans
ax=fig.add_subplot(gs[2,4]); style(ax,"Is the layout random?","Clark-Evans ratio")
ce=[]
for sid in sids:
    rs=[r for r in rows if r['sid']==sid]
    if len(rs)<5: continue
    p=plates[sid]; P=np.array([[r['x'],r['y']] for r in rs])
    D=np.hypot(P[:,None,0]-P[None,:,0],P[:,None,1]-P[None,:,1]); np.fill_diagonal(D,1e9)
    ce.append(D.min(1).mean()/(0.5/math.sqrt(len(rs)/(math.pi*p['R']**2))))
ax.axvspan(0.9,1.1,color="#e6e5e0")
ax.scatter(ce,range(len(ce)),s=60,color="#eda100",ec=SURF,lw=1.5,zorder=3)
ax.axvline(1.0,color=INK2,lw=1.4)
ax.set_yticks([]); ax.set_xlim(0.75,1.25)
ax.set_ylim(-1.2,len(ce)+0.6)
ax.text(1.0,-0.95,"random",fontsize=8,color=INK2,ha="center")
ax.text(0.775,-0.95,"clustered",fontsize=8,color=INK2)
ax.text(1.245,-0.95,"regular",fontsize=8,color=INK2,ha="right")

fig.legend(handles=[Line2D([],[],marker='o',ls='',ms=8,mfc=C[s],mec=SURF,label=s) for s in SPECIES],
           loc="upper right",bbox_to_anchor=(0.985,0.975),ncol=5,frameon=False,
           fontsize=9,labelcolor=INK2,handletextpad=0.3,columnspacing=1.4)
fig.text(0.045,0.955,"AGAR layout exploration -- 10 demo plates, 387 colonies",fontsize=15,color=INK,weight="bold")
fig.text(0.045,0.925,"Dashed circle = 90% of the plate radius. Disc size is proportional to the real colony size (a lower bound is applied for visibility).",
         fontsize=9.5,color=INK2)
fig.savefig(OUT / "yerlesim_kesfi.png",dpi=135,facecolor=SURF)
print("ok")
