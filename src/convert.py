#!/usr/bin/env python3
"""
AGAR JSON -> YOLO conversion.

What it does:
  - Scans the *.json files under the AGAR root
  - Applies the Outline 3.2 filters (lower-resolution + countable + 5 microorganisms)
  - Discards images containing defects / contamination COMPLETELY
  - Reads the image size from the file itself (it is not in the JSON, 2048x2048 is not assumed)
  - Produces out/labels/<id>.txt  and  out/images/<id>.<ext> (symlink)
  - Produces out/manifest.csv -> the source for the splits, the subsamples and the paper's data table

Usage:
    python convert.py --src /path/AGAR_representative --out data/processed
    python convert.py --src ... --out ... --copy     # copy instead of symlinking
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# --- FIXED CLASS ORDER -- will not change for the whole project -----------
CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}

# Artifact classes left out of scope
ARTIFACT_CLASSES = {"defects", "contamination"}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def normalize_class(name: str) -> str:
    """'S. aureus' -> 'S.aureus'. Both spellings occur in AGAR."""
    return name.replace(" ", "").strip()


def find_image(json_path: Path):
    """Finds the image file that has the same name as the JSON."""
    for ext in IMG_EXTS:
        cand = json_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AGAR root folder")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--background", default="lower-resolution",
                    help="subset to keep (default: lower-resolution). "
                         "if you pass 'all' no filter is applied")
    ap.add_argument("--copy", action="store_true",
                    help="copy the images instead of symlinking them")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"ERROR: source folder does not exist: {src}")

    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    json_files = sorted(src.rglob("*.json"))
    if not json_files:
        sys.exit(f"ERROR: no JSON found under {src}")

    # Counters -- the reason every image was dropped is on the record
    n = Counter()
    class_boxes = Counter()          # box count per class
    unknown_classes = Counter()      # unexpected class names
    rows = []

    for jp in json_files:
        n["json_total"] += 1

        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            n["dropped_bad_json"] += 1
            print(f"  ! skipped broken JSON: {jp.name} ({e})")
            continue

        # --- Filter 1: subset ---
        bg = meta.get("background")
        if args.background != "all" and bg != args.background:
            n["dropped_subset"] += 1
            continue

        labels = meta.get("labels") or []

        # --- Filter 2: countable ---
        # In AGAR, colony-level labels exist ONLY for countable images.
        # For uncountable and empty images the labels list comes back empty.
        # Decision 3.43: these two used to be added up in a SINGLE counter. The
        # data section of the paper will report "how many images were dropped and
        # why" separately, so they are split apart via colonies_number:
        # empty -> 0, uncountable -> >0.
        if len(labels) == 0:
            cn = meta.get("colonies_number")
            try:
                cn = int(cn)
            except (TypeError, ValueError):
                cn = None
            if cn == 0:
                n["dropped_empty"] += 1
            elif cn is not None and cn > 0:
                n["dropped_uncountable"] += 1
            else:
                n["dropped_unlabelled_unknown"] += 1
            n["dropped_not_countable"] += 1
            continue

        # --- Filter 3: images containing an artifact class are dropped entirely ---
        # (deleting the box but keeping the image teaches the model "there is no object here")
        present = {normalize_class(l.get("class", "")) for l in labels}
        present |= {normalize_class(c) for c in (meta.get("classes") or [])}
        if present & ARTIFACT_CLASSES:
            n["dropped_artifact"] += 1
            continue

        unknown = present - set(CLASS_ORDER) - ARTIFACT_CLASSES
        if unknown:
            for b in unknown:
                unknown_classes[b] += 1
            n["dropped_unknown_class"] += 1
            continue

        # --- Image file and its SIZE (not in the JSON, read from the file) ---
        img_path = find_image(jp)
        if img_path is None:
            n["dropped_no_image"] += 1
            print(f"  ! image not found: {jp.name}")
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            n["dropped_image_unreadable"] += 1
            print(f"  ! could not open image: {img_path.name} ({e})")
            continue

        # --- Box conversion ---
        lines = []
        per_class = defaultdict(int)
        dropped_boxes = 0
        clipped_boxes = 0

        for l in labels:
            cname = normalize_class(l.get("class", ""))
            cid = CLASS_TO_ID.get(cname)
            if cid is None:
                dropped_boxes += 1
                continue

            x, y = float(l["x"]), float(l["y"])          # top-left corner
            w, h = float(l["width"]), float(l["height"])

            # clip to the image bounds
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(W), x + w), min(float(H), y + h)
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                clipped_boxes += 1

            bw, bh = x1 - x0, y1 - y0
            if bw <= 1 or bh <= 1:      # broken / zero-area box
                dropped_boxes += 1
                continue

            xc = (x0 + bw / 2) / W
            yc = (y0 + bh / 2) / H
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw / W:.6f} {bh / H:.6f}")
            per_class[cname] += 1
            class_boxes[cname] += 1

        if not lines:
            n["dropped_no_valid_box"] += 1
            continue

        n["boxes_dropped"] += dropped_boxes
        n["boxes_clipped"] += clipped_boxes

        # --- Writing ---
        stem = img_path.stem
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")

        dst_img = out / "images" / img_path.name
        # Decision 3.44: exists() FOLLOWS the symlink. If the source AGAR folder
        # has been moved the link breaks, exists() returns False, the code tries
        # to create the symlink again and blows up with FileExistsError. The
        # is_symlink() check catches the broken link too.
        if dst_img.is_symlink() and not dst_img.exists():
            print(f"  ! refreshing broken symlink: {dst_img.name}")
            dst_img.unlink()
        if not dst_img.is_symlink() and not dst_img.exists():
            if args.copy:
                shutil.copy2(img_path, dst_img)
            else:
                dst_img.symlink_to(img_path)

        rows.append({
            "sample_id": meta.get("sample_id", stem),
            "stem": stem,
            "image_path": str(img_path),
            "label_path": str(out / "labels" / f"{stem}.txt"),
            "width": W,
            "height": H,
            "background": bg,
            "n_boxes": len(lines),
            "colonies_number": meta.get("colonies_number", ""),
            "classes": "|".join(sorted(per_class)),
            "n_classes": len(per_class),
            **{f"n_{c}": per_class.get(c, 0) for c in CLASS_ORDER},
        })
        n["kept"] += 1

    if not rows:
        sys.exit("ERROR: no image made it through the filters. "
                 "Check the --background value.")

    # --- manifest.csv ---
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # --- class names file (for YOLO) ---
    (out / "classes.txt").write_text("\n".join(CLASS_ORDER) + "\n", encoding="utf-8")

    # --- SUMMARY ---
    print("\n" + "=" * 58)
    print("CONVERSION SUMMARY")
    print("=" * 58)
    print(f"  scanned JSON             : {n['json_total']}")
    print(f"  dropped / subset         : {n['dropped_subset']}")
    print(f"  dropped / not countable  : {n['dropped_not_countable']}"
          f"   (empty {n['dropped_empty']} + uncountable {n['dropped_uncountable']}"
          f" + unknown {n['dropped_unlabelled_unknown']})   <- to be reported SEPARATELY in the paper")
    print(f"  dropped / artifact class : {n['dropped_artifact']}   <- to be reported in the paper")
    print(f"  dropped / unknown class  : {n['dropped_unknown_class']}")
    print(f"  dropped / image missing  : {n['dropped_no_image']}")
    print(f"  dropped / no valid box   : {n['dropped_no_valid_box']}")
    print(f"  KEPT IMAGES              : {n['kept']}")
    print(f"  total boxes              : {sum(class_boxes.values())}")
    print(f"  clipped boxes            : {n['boxes_clipped']}")
    print(f"  dropped boxes (broken)   : {n['boxes_dropped']}")
    print("\n  boxes per class:")
    for c in CLASS_ORDER:
        print(f"    {CLASS_TO_ID[c]}  {c:<16} {class_boxes.get(c, 0)}")
    if unknown_classes:
        print("\n  ! unexpected class names (check them):")
        for k, v in unknown_classes.most_common():
            print(f"    {k}: {v}")
    print(f"\n  manifest -> {manifest}")
    print("=" * 58)
    print("\nNEXT STEP: python check_labels.py --data",
          out, "--n 30\n")


if __name__ == "__main__":
    main()
