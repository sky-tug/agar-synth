#!/usr/bin/env python3
"""
Egitim sarmalayicisi -- her kosuda AYNI protokol, ve SURE/VRAM loglanir.

Neden dogrudan `yolo train` degil:
  1. Outline bolum 6 (hesaplama maliyeti) icin sure, GPU-saat ve tepe VRAM
     ILK kosudan itibaren kayit altinda olmali. Sonradan geri donup olculemez.
  2. Augmentation politikasi konfigurasyon dosyasindan gelmeli, komut
     satirindan degil. Aksi halde "hangi kosuda ne aciktI" sorusu cevapsiz kalir.
  3. Ultralytics'in varsayilanlari surumden surume degisiyor. Buradaki config
     TUM augmentation alanlarini ACIKCA yaziyor -- ortuk varsayilan kalmiyor.
  4. Faz 5'te protokol donduruldugunda, dondurulan sey bu dosya + config olacak.

Kullanim:
    # duman testi (demo veriyle, GPU var mi / hat calisiyor mu)
    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\
        --epochs 10 --name duman_testi --duman

    # G100 tam egitim, tek seed -- Faz 2'nin ana isi
    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\
        --name G100_s0

    # klasik kol
    python scripts/train.py --config configs/base.yaml \\
        --overlay configs/aug_c_kopyala.yaml --level 25 --seed 0 --name BC_G25_s0

Cikti: runs/<name>/olcum.json  ->  scripts/butce.py bunu okur
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------

def derin_birlestir(a: dict, b: dict) -> dict:
    """b'yi a'nin uzerine bindir (ic ice sozlukler dahil)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = derin_birlestir(out[k], v) if (
            isinstance(v, dict) and isinstance(out.get(k), dict)) else v
    return out


def gpu_bilgisi() -> dict:
    bilgi = {"cuda": False}
    try:
        import torch
        bilgi["torch"] = torch.__version__
        bilgi["cuda"] = bool(torch.cuda.is_available())
        if bilgi["cuda"]:
            bilgi["cuda_surum"] = torch.version.cuda
            bilgi["kart"] = torch.cuda.get_device_name(0)
            bilgi["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            bilgi["kart_sayisi"] = torch.cuda.device_count()
    except Exception as e:
        bilgi["hata"] = str(e)
    try:
        bilgi["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception:
        pass
    return bilgi


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(KOK), "rev-parse", "--short", "HEAD"],
            text=True, timeout=10).strip()
    except Exception:
        return "bilinmiyor"


def liste_yolu(data_root: Path, seviye: int) -> Path:
    """
    Ultralytics'in okuyacagi mutlak yollu liste.
    seviye 100 -> train.txt, digerleri -> train_<seviye>.txt
    """
    ad = "train" if seviye == 100 else f"train_{seviye}"
    p = data_root / "lists" / f"{ad}.txt"
    if not p.exists():
        sys.exit(f"HATA: {p} yok.\n  once: python src/make_splits.py --data {data_root}")
    return p


def liste_dogrula(p: Path, data_root: Path) -> int:
    """
    Listedeki yollarin YANINDA etiket bulunabiliyor mu?
    Ultralytics etiketi, yoldaki son '/images/' parcasini '/labels/' ile
    degistirerek arar. Liste ham AGAR klasorunu gosteriyorsa etiket BULUNAMAZ
    ve model sessizce 'nesne yok' ogrenir -- en sinsi hata bu.
    """
    satir = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not satir:
        sys.exit(f"HATA: {p} bos. (Demo pakette val/test bos cikar -- tam veri gerekli.)")
    eksik = 0
    for s in satir[:50]:
        ip = Path(s)
        if f"{'/'}images{'/'}" not in str(ip):
            sys.exit(
                f"HATA: liste ham veri yolunu gosteriyor:\n    {ip}\n"
                f"  Ultralytics etiketi '/images/' -> '/labels/' ile bulur.\n"
                f"  Listeler {data_root/'images'} altini gostermeli.\n"
                f"  Duzeltme: src/make_splits.py icindeki write_paths()")
        lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
        eksik += (not lp.exists())
    if eksik:
        sys.exit(f"HATA: ilk 50 goruntunun {eksik} tanesinde etiket dosyasi yok.")
    return len(satir)


def dataset_yaml_yaz(hedef: Path, data_root: Path, train_liste: Path,
                     names: list[str]) -> Path:
    hedef.parent.mkdir(parents=True, exist_ok=True)
    icerik = {
        "path": str(data_root),
        "train": str(train_liste),
        "val": str(data_root / "lists" / "val.txt"),
        "test": str(data_root / "lists" / "test.txt"),
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    hedef.write_text(yaml.safe_dump(icerik, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
    return hedef


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overlay", help="augmentation kolu (aug_b_klasik.yaml gibi)")
    ap.add_argument("--level", type=int, default=100,
                    choices=[100, 50, 25, 10], help="gercek veri seviyesi (%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True, help="kosu adi -> runs/<name>")
    ap.add_argument("--epochs", type=int, help="config'i ez (duman testi icin)")
    ap.add_argument("--imgsz", type=int, help="config'i ez")
    ap.add_argument("--batch", type=int, help="config'i ez (VRAM'e gore)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--duman", action="store_true",
                    help="duman testi: W&B kapali, sonuc olcum.json'a 'duman' diye isaretlenir")
    ap.add_argument("--kuru", action="store_true",
                    help="egitimi baslatma, sadece hazirligi ve dogrulamalari yap")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.overlay:
        cfg = derin_birlestir(cfg, yaml.safe_load(
            Path(args.overlay).read_text(encoding="utf-8")))

    t = cfg["train"]
    if args.epochs:
        t["epochs"] = args.epochs
    if args.imgsz:
        t["imgsz"] = args.imgsz
    if args.batch:
        t["batch"] = args.batch

    data_root = (KOK / cfg["data"]["root"]).resolve()
    names = cfg["data"]["names"]

    train_liste = liste_yolu(data_root, args.level)
    n_train = liste_dogrula(train_liste, data_root)
    n_val = liste_dogrula(data_root / "lists" / "val.txt", data_root)

    cikti = KOK / "runs" / args.name
    cikti.mkdir(parents=True, exist_ok=True)
    ds_yaml = dataset_yaml_yaz(cikti / "dataset.yaml", data_root, train_liste, names)

    gpu = gpu_bilgisi()
    print("=" * 62)
    print(f"KOSU        : {args.name}")
    print(f"seviye      : G{args.level}   seed {args.seed}")
    print(f"train / val : {n_train} / {n_val} goruntu")
    print(f"imgsz       : {t['imgsz']}   batch {t['batch']}   epochs {t['epochs']}")
    print(f"augment     : {args.overlay or 'base (minimum: flip + HSV)'}")
    print(f"GPU         : {gpu.get('kart', 'YOK -- CPU')}"
          f"{'  ' + str(gpu.get('vram_gb')) + ' GB' if gpu.get('cuda') else ''}")
    print("=" * 62)

    if not gpu.get("cuda"):
        print("! UYARI: CUDA gorunmuyor. CPU'da egitim Faz 2 icin anlamsiz.")
    if args.kuru:
        print("\n--kuru: hazirlik tamam, egitim baslatilmadi.")
        print(f"dataset yaml -> {ds_yaml}")
        return

    # ---------------- Egitim ----------------
    import torch
    from ultralytics import YOLO

    if gpu.get("cuda"):
        torch.cuda.reset_peak_memory_stats()

    epoch_sureleri = []
    son = {"t": time.perf_counter()}

    def on_epoch_end(trainer):
        simdi = time.perf_counter()
        epoch_sureleri.append(round(simdi - son["t"], 3))
        son["t"] = simdi

    model = YOLO(cfg["model"]["weights"])
    model.add_callback("on_fit_epoch_end", on_epoch_end)

    kwargs = dict(
        data=str(ds_yaml),
        epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
        patience=t["patience"], optimizer=t["optimizer"],
        lr0=t["lr0"], lrf=t["lrf"], momentum=t["momentum"],
        weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"],
        cos_lr=t["cos_lr"], deterministic=t["deterministic"],
        workers=t["workers"], amp=t["amp"], val=t["val"], plots=t["plots"],
        seed=args.seed, device=args.device,
        project=str(KOK / "runs"), name=args.name, exist_ok=True,
        **{k: v for k, v in cfg["augment"].items() if v is not None},
    )

    baslangic = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    sonuc = model.train(**kwargs)
    toplam = time.perf_counter() - t0

    tepe_vram = (round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
                 if gpu.get("cuda") else None)

    # ---------------- Olcum kaydi ----------------
    gerceklesen = len(epoch_sureleri)
    olcum = {
        "kosu": args.name,
        "duman_testi": bool(args.duman),
        "tarih_utc": baslangic.isoformat(),
        "git_commit": git_commit(),
        "seviye": args.level,
        "seed": args.seed,
        "overlay": args.overlay,
        "n_train": n_train,
        "n_val": n_val,
        "imgsz": t["imgsz"],
        "batch": t["batch"],
        "epochs_planlanan": t["epochs"],
        "epochs_gerceklesen": gerceklesen,
        # --- bolum 6 (hesaplama maliyeti) buradan yazilacak ---
        "toplam_sure_s": round(toplam, 2),
        "toplam_sure_dk": round(toplam / 60, 2),
        "epoch_basina_s": round(toplam / max(gerceklesen, 1), 2),
        "epoch_sureleri_s": epoch_sureleri,
        "gpu_saat": round(toplam / 3600, 4),
        "tepe_vram_gb": tepe_vram,
        "goruntu_saniye": round(n_train * max(gerceklesen, 1) / toplam, 2),
        "ortam": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **gpu,
        },
        "augment": cfg["augment"],
    }
    try:
        olcum["metrikler_ultralytics"] = {
            k: float(v) for k, v in sonuc.results_dict.items()
            if isinstance(v, (int, float))
        }
    except Exception:
        pass

    (cikti / "olcum.json").write_text(
        json.dumps(olcum, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"toplam sure    : {olcum['toplam_sure_dk']:.1f} dk "
          f"({olcum['gpu_saat']:.3f} GPU-saat)")
    print(f"epoch basina   : {olcum['epoch_basina_s']:.1f} s "
          f"({gerceklesen} epoch kostu)")
    print(f"tepe VRAM      : {tepe_vram} GB")
    print(f"olcum -> {cikti / 'olcum.json'}")
    print("=" * 62)
    print("\nSIRADAKI:")
    print(f"  python src/eval/evaluate.py --weights runs/{args.name}/weights/best.pt "
          f"--split val  --out runs/{args.name}/eval --imgsz {t['imgsz']}")
    print(f"  python scripts/butce.py --olcum runs/{args.name}/olcum.json")


if __name__ == "__main__":
    main()
