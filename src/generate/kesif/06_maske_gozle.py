import json, glob, numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"
SEN="#2a78d6"; SIL="#eb6834"
C="/home/claude/faz3/cikti/maske100"
adlar=[json.load(open(f))["ad"] for f in sorted(glob.glob(C+"/kayit/*.json"))]
sec=[adlar[0], adlar[3], adlar[7]]

fig,axs=plt.subplots(3,3,figsize=(11.6,11.9),facecolor=SURF)
plt.subplots_adjust(left=0.005,right=0.995,top=0.885,bottom=0.005,hspace=0.045,wspace=0.02)
basliklar=["1 · arka plan (gerçek plak)","2 · maske","3 · üst üste"]
for i,ad in enumerate(sec):
    kay=json.load(open(f"{C}/kayit/{ad}.json"))
    bg=cv2.cvtColor(cv2.imread(f"{C}/arkaplan/{ad}.jpg"),cv2.COLOR_BGR2RGB)
    ms=cv2.imread(f"{C}/maske/{ad}_sentetik.png",0)
    ml=cv2.imread(f"{C}/maske/{ad}_silme.png",0)
    kap=np.zeros_like(bg); kap[ml>0]=[235,104,52]; kap[ms>0]=[42,120,214]
    ust=cv2.addWeighted(bg,0.55,kap,0.45,0)
    for j,im in enumerate([bg,kap,ust]):
        ax=axs[i,j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#e0dfd9"); sp.set_lw(0.8)
        if i==0: ax.set_title(basliklar[j],fontsize=11,color=INK,pad=7)
    axs[i,0].text(40,150,f"{kay['arka_plan']}",fontsize=11,color="w",
                  bbox=dict(fc="#00000088",ec="none",pad=3))
    axs[i,2].text(40,150,f"{kay['sentetik_koloni']} sentetik  ·  "
                         f"{kay['silinen_gercek_koloni']} silinecek",
                  fontsize=10,color="w",bbox=dict(fc="#00000088",ec="none",pad=3))
fig.legend(handles=[Line2D([],[],marker='s',ls='',ms=10,mfc=SEN,mec=SURF,
                           label="sentetik — difüzyon buraya koloni çizecek"),
                    Line2D([],[],marker='s',ls='',ms=10,mfc=SIL,mec=SURF,
                           label="silme — gerçek koloninin üzeri boyanacak")],
           loc="upper left",bbox_to_anchor=(0.012,0.945),ncol=2,frameon=False,
           fontsize=10,labelcolor=INK2,handletextpad=0.4,columnspacing=2.0)
fig.text(0.012,0.965,"Gözle kontrol — maske üretici",fontsize=15,color=INK,weight="bold")
fig.text(0.012,0.905,"Mavi disk etiket kutusunun tam kendisi: pay yok, büyütme yok. Difüzyon dışına taşamaz → etiket hatası sıfır.",
         fontsize=9.5,color=INK2)
fig.savefig("/home/claude/faz3/maske_gozle.png",dpi=118,facecolor=SURF)
print("ok")
