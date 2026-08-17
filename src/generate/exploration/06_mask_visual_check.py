#!/usr/bin/env python3
"""
06 -- visual check of the mask builder: background, mask and the two overlaid,
for three generated plates.

Expects `mask.py build --split-masks` to have already written its output under
OUT/mask100 (provenance/, backgrounds/, masks/).

One-shot exploration script; results are recorded in ../LAYOUT_FINDINGS.md.
"""
import json
from pathlib import Path

import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "out"

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"
SYNTH_C="#2a78d6"; ERASE_C="#eb6834"
C = OUT / "mask100"
names=[json.load(open(f))["name"] for f in sorted((C/"provenance").glob("*.json"))]
chosen=[names[0], names[3], names[7]]

fig,axs=plt.subplots(3,3,figsize=(11.6,11.9),facecolor=SURF)
plt.subplots_adjust(left=0.005,right=0.995,top=0.885,bottom=0.005,hspace=0.045,wspace=0.02)
titles=["1 - background (real plate)","2 - mask","3 - overlaid"]
for i,name in enumerate(chosen):
    prov=json.load(open(C/"provenance"/f"{name}.json"))
    bg=cv2.cvtColor(cv2.imread(str(C/"backgrounds"/f"{name}.jpg")),cv2.COLOR_BGR2RGB)
    synth=cv2.imread(str(C/"masks"/f"{name}_synthetic.png"),0)
    erase=cv2.imread(str(C/"masks"/f"{name}_erase.png"),0)
    layer=np.zeros_like(bg); layer[erase>0]=[235,104,52]; layer[synth>0]=[42,120,214]
    blended=cv2.addWeighted(bg,0.55,layer,0.45,0)
    for j,im in enumerate([bg,layer,blended]):
        ax=axs[i,j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#e0dfd9"); sp.set_lw(0.8)
        if i==0: ax.set_title(titles[j],fontsize=11,color=INK,pad=7)
    axs[i,0].text(40,150,f"{prov['background']}",fontsize=11,color="w",
                  bbox=dict(fc="#00000088",ec="none",pad=3))
    axs[i,2].text(40,150,f"{prov['n_synthetic']} synthetic  -  "
                         f"{prov['n_erased_real']} to erase",
                  fontsize=10,color="w",bbox=dict(fc="#00000088",ec="none",pad=3))
fig.legend(handles=[Line2D([],[],marker='s',ls='',ms=10,mfc=SYNTH_C,mec=SURF,
                           label="synthetic -- diffusion will draw colonies here"),
                    Line2D([],[],marker='s',ls='',ms=10,mfc=ERASE_C,mec=SURF,
                           label="erase -- the real colony will be painted over")],
           loc="upper left",bbox_to_anchor=(0.012,0.945),ncol=2,frameon=False,
           fontsize=10,labelcolor=INK2,handletextpad=0.4,columnspacing=2.0)
fig.text(0.012,0.965,"Visual check -- mask builder",fontsize=15,color=INK,weight="bold")
fig.text(0.012,0.905,"The blue disc is exactly the label box: no margin, no dilation. Diffusion cannot spill outside it -> zero label error.",
         fontsize=9.5,color=INK2)
fig.savefig(OUT / "maske_gozle.png",dpi=118,facecolor=SURF)
print("ok")
