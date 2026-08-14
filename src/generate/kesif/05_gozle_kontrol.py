import json, glob, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

SIN=["S.aureus","B.subtilis","P.aeruginosa","E.coli","C.albicans"]
C={s:c for s,c in zip(SIN,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"])}
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#8a8a85"
R=0.465; CX=CY=0.5

def gercek(stem):
    et=f"/mnt/user-data/uploads/agar-synth/data/processed/labels/{stem}.txt"
    return [dict(sinif=SIN[int(l.split()[0])],xc=float(l.split()[1]),yc=float(l.split()[2]),
                 cap=(float(l.split()[3])+float(l.split()[4]))/2)
            for l in open(et).read().split("\n") if l.strip()]
def plan(f):
    return json.load(open(f))["koloniler"]

satirlar=[
 ("GERÇEK  (AGAR demo)",[gercek(s) for s in ["14627","14380","14684","14581"]]),
 ("ÜRETİLEN  (ana hat)",[plan(f) for f in sorted(glob.glob("/home/claude/faz3/cikti/s100_0/planlar/*.json"))[:4]]),
 ("A3 NAİF  (ablasyon)",[plan(f) for f in sorted(glob.glob("/home/claude/faz3/cikti/a3/planlar/*.json"))[:4]]),
]
fig,axs=plt.subplots(3,4,figsize=(11.5,9.2),facecolor=SURF)
plt.subplots_adjust(left=0.115,right=0.99,top=0.885,bottom=0.015,hspace=0.06,wspace=0.05)
for i,(ad,plaklar) in enumerate(satirlar):
    for j,kol in enumerate(plaklar):
        ax=axs[i,j]; ax.set_facecolor(SURF)
        ax.add_patch(Circle((CX,CY),R,fill=False,ec=MUT,lw=1.2))
        for k in kol:
            ax.add_patch(Circle((k["xc"],k["yc"]),max(k["cap"]/2,0.008),
                                fc=C[k["sinif"]],ec=SURF,lw=1.0,alpha=0.9))
        ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect(1)
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#e0dfd9"); sp.set_lw(0.8)
        ax.text(0.03,0.05,f"n={len(kol)}",fontsize=8,color=INK2)
    axs[i,0].set_ylabel(ad,fontsize=11,color=INK,labelpad=12,weight="bold" if i==0 else "normal")
fig.legend(handles=[Line2D([],[],marker='o',ls='',ms=8,mfc=C[s],mec=SURF,label=s) for s in SIN],
           loc="upper right",bbox_to_anchor=(0.99,0.975),ncol=5,frameon=False,
           fontsize=9,labelcolor=INK2,handletextpad=0.3,columnspacing=1.3)
fig.text(0.02,0.955,"Gözle kontrol — yerleşim modeli",fontsize=15,color=INK,weight="bold")
fig.text(0.02,0.925,"A3'te koloniler plak dışına taşıyor, kenara yığılıyor ve boyutları sınıfıyla uyumsuz. Ablasyon artık boş değil.",
         fontsize=9.5,color=INK2)
fig.savefig("/home/claude/faz3/yerlesim_gozle.png",dpi=140,facecolor=SURF)
print("ok")
