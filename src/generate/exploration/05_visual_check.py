#!/usr/bin/env python3
"""
05 -- side-by-side visual check of the layout model: real plates vs the main
line vs the A3 naive ablation.

Expects `layout.py sample` to have already written its plans under OUT
(main line -> s100_0/plans, ablation -> a3/plans).

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import json
from pathlib import Path

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "data" / "processed" / "labels"
OUT = Path(__file__).resolve().parent / "out"

SPECIES=["S.aureus","B.subtilis","P.aeruginosa","E.coli","C.albicans"]
C={s:c for s,c in zip(SPECIES,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"])}
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#8a8a85"
R=0.465; CX=CY=0.5

def real(stem):
    lab = LABELS / f"{stem}.txt"
    return [dict(cls=SPECIES[int(l.split()[0])],xc=float(l.split()[1]),yc=float(l.split()[2]),
                 diameter=(float(l.split()[3])+float(l.split()[4]))/2)
            for l in open(lab).read().split("\n") if l.strip()]
def read_plan(f):
    return json.load(open(f))["colonies"]

panel_rows=[
 ("REAL  (AGAR demo)",[real(s) for s in ["14627","14380","14684","14581"]]),
 ("GENERATED  (main line)",[read_plan(f) for f in sorted((OUT/"s100_0"/"plans").glob("*.json"))[:4]]),
 ("A3 NAIVE  (ablation)",[read_plan(f) for f in sorted((OUT/"a3"/"plans").glob("*.json"))[:4]]),
]
fig,axs=plt.subplots(3,4,figsize=(11.5,9.2),facecolor=SURF)
plt.subplots_adjust(left=0.115,right=0.99,top=0.885,bottom=0.015,hspace=0.06,wspace=0.05)
for i,(title,plates) in enumerate(panel_rows):
    for j,colonies in enumerate(plates):
        ax=axs[i,j]; ax.set_facecolor(SURF)
        ax.add_patch(Circle((CX,CY),R,fill=False,ec=MUT,lw=1.2))
        for k in colonies:
            ax.add_patch(Circle((k["xc"],k["yc"]),max(k["diameter"]/2,0.008),
                                fc=C[k["cls"]],ec=SURF,lw=1.0,alpha=0.9))
        ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect(1)
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#e0dfd9"); sp.set_lw(0.8)
        ax.text(0.03,0.05,f"n={len(colonies)}",fontsize=8,color=INK2)
    axs[i,0].set_ylabel(title,fontsize=11,color=INK,labelpad=12,weight="bold" if i==0 else "normal")
fig.legend(handles=[Line2D([],[],marker='o',ls='',ms=8,mfc=C[s],mec=SURF,label=s) for s in SPECIES],
           loc="upper right",bbox_to_anchor=(0.99,0.975),ncol=5,frameon=False,
           fontsize=9,labelcolor=INK2,handletextpad=0.3,columnspacing=1.3)
fig.text(0.02,0.955,"Visual check -- layout model",fontsize=15,color=INK,weight="bold")
fig.text(0.02,0.925,"In A3 the colonies spill outside the plate, pile up at the edge and their size does not match their class. The ablation is no longer empty.",
         fontsize=9.5,color=INK2)
fig.savefig(OUT / "yerlesim_gozle.png",dpi=140,facecolor=SURF)
print("ok")
