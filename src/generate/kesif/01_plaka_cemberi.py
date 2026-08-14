import json, glob, os, math
import numpy as np, cv2

D = "/mnt/user-data/uploads/agar-synth/data/AGAR_representative/lower-resolution"

def plaka_cemberi(path):
    """Plak dairesini goruntuden kestir: Hough + yedek olarak esikleme."""
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
    # yedek: parlak bolgenin kutlesi
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ys, xs = np.nonzero(th)
    x, y = xs.mean()*4, ys.mean()*4
    r = math.sqrt(th.sum()/255/math.pi)*4
    return float(x), float(y), float(r), (w,h), "otsu"

rows=[]
plak=[]
for jf in sorted(glob.glob(D+"/*.json")):
    sid = os.path.splitext(os.path.basename(jf))[0]
    j = json.load(open(jf))
    im = D+f"/{sid}.jpg"
    cx, cy, R, (W,H), yontem = plaka_cemberi(im)
    plak.append(dict(sid=sid, cx=cx, cy=cy, R=R, W=W, H=H, yontem=yontem,
                     n=j["colonies_number"], sinif="+".join(j["classes"])))
    for L in j["labels"]:
        bx = L["x"] + L["width"]/2.0
        by = L["y"] + L["height"]/2.0
        rows.append(dict(sid=sid, sinif=L["class"], x=bx, y=by,
                         w=L["width"], h=L["height"],
                         r_norm=math.hypot(bx-cx, by-cy)/R,
                         theta=math.atan2(by-cy, bx-cx)))
np.save("/home/claude/faz3/rows.npy", np.array(rows, dtype=object))
np.save("/home/claude/faz3/plak.npy", np.array(plak, dtype=object))

print("=== PLAK CEMBERI ===")
for p in plak:
    print(f"{p['sid']}  {p['W']}x{p['H']}  merkez=({p['cx']:.0f},{p['cy']:.0f})  R={p['R']:.0f}  "
          f"R/W={p['R']/p['W']:.3f}  [{p['yontem']}]  n={p['n']}  {p['sinif']}")
