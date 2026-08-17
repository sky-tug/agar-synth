#!/usr/bin/env python3
"""
Produces the splits (train/val/test) and the nested subsample lists.

Design decisions:
  - The split is at IMAGE level (pieces of the same plate must not fall into different sets)
  - STRATIFIED by species combination: if C.albicans disappears in the 10% subset,
    the difference between levels comes from class imbalance, not from the amount of data
  - NESTED: the 10% subset is a SUBSET of the 25%, the 25% of the 50%, the 50% of the 100%
  - The lists are written under splits/ as STEMS -> they get committed to git, they are portable
  - The absolute-path lists that Ultralytics will read are produced under data/.../lists/
    (those are in gitignore; they are regenerated on every machine)

Usage:
    python src/make_splits.py --data data/processed --seed 42
    python src/make_splits.py --data data/processed --val 0.15 --test 0.15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_ORDER = ["S.aureus", "B.subtilis", "P.aeruginosa", "E.coli", "C.albicans"]
LEVELS = [50, 25, 10]          # nested subsample levels (% within train)


def stratified_split(df, val_frac, test_frac, rng):
    """Stratum = species combination. Each stratum is split within itself."""
    train, val, test = [], [], []
    for _, grp in df.groupby("classes", sort=True):
        idx = grp["stem"].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # guarantee that train does not end up empty in very small strata
        while n - n_test - n_val < 1 and (n_test + n_val) > 0:
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1
        test += idx[:n_test]
        val += idx[n_test:n_test + n_val]
        train += idx[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def nested_subsamples(train_stems, df, rng):
    """
    Stepwise narrowing so that 50% ⊂ 100%, 25% ⊂ 50%, 10% ⊂ 25%.
    Stratified sampling at every step.
    """
    lookup = df.set_index("stem")["classes"].to_dict()
    subsets = {}
    pool = list(train_stems)
    prev_level = 100
    for lvl in LEVELS:
        ratio = lvl / prev_level          # share relative to the previous level
        selected = []
        by_stratum = {}
        for s in pool:
            by_stratum.setdefault(lookup[s], []).append(s)
        for _, items in sorted(by_stratum.items()):
            items = sorted(items)
            rng.shuffle(items)
            k = max(1, int(round(len(items) * ratio)))
            selected += items[:k]
        selected = sorted(selected)
        subsets[lvl] = selected
        pool = selected                    # the next level will be a SUBSET of THIS one
        prev_level = lvl
    return subsets


def class_table(stems, df):
    """Box count per class + image count for one set."""
    sub = df[df["stem"].isin(stems)]
    row = {"images": len(sub)}
    total = 0
    for c in CLASS_ORDER:
        v = int(sub[f"n_{c}"].sum())
        row[c] = v
        total += v
    row["boxes"] = total
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    man_path = data / "manifest.csv"
    if not man_path.exists():
        sys.exit(f"ERROR: {man_path} does not exist. run convert.py first.")

    df = pd.read_csv(man_path)
    df["stem"] = df["stem"].astype(str)
    rng = np.random.default_rng(args.seed)

    class _R:                       # thin wrapper to shuffle with an np Generator
        @staticmethod
        def shuffle(x):
            perm = rng.permutation(len(x))
            x[:] = [x[i] for i in perm]

    train, val, test = stratified_split(df, args.val, args.test, _R)
    subsets = nested_subsamples(train, df, _R)

    # ---------------- Writing: stem lists (these go into git) ----------------
    sd = Path(args.splits_dir)
    sd.mkdir(parents=True, exist_ok=True)

    def write(name, stems):
        (sd / f"{name}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    write("train", train)
    write("val", val)
    write("test", test)
    for lvl, stems in subsets.items():
        write(f"train_{lvl}", stems)

    # ---------------- absolute-path lists for Ultralytics ---------------
    #
    # CAUTION -- Ultralytics looks for the label file by replacing the last
    # '/images/' segment of the image path with '/labels/'.
    # For that reason the lists MUST point not at the RAW AGAR folder but at
    # the data/processed/images/ tree that convert.py produces.
    # If the raw path is written the label cannot be found, Ultralytics warns
    # but does not stop, and the model learns "there is no object in these
    # images". A silent, fatal error.
    lists = data / "lists"
    lists.mkdir(exist_ok=True)
    img_dir = data / "images"
    path_of = {}
    for s, raw in df.set_index("stem")["image_path"].to_dict().items():
        p = img_dir / Path(raw).name          # convert.py symlinked it here
        if not p.exists():                    # search if the extension differs
            matches = sorted(img_dir.glob(f"{s}.*"))
            if not matches:
                sys.exit(f"ERROR: no image for {s} under {img_dir}. "
                         f"run convert.py first.")
            p = matches[0]
        path_of[s] = p

    def write_paths(name, stems):
        # CAUTION: DO NOT USE .resolve(). The files under data/processed/images/
        # are SYMLINKS to the raw AGAR folder; resolve() follows them and turns
        # the path into the raw folder -> the '/images/' segment disappears ->
        # Ultralytics cannot find the label. path_of is already an absolute path
        # (data is already .resolve()d).
        (lists / f"{name}.txt").write_text(
            "\n".join(str(path_of[s]) for s in stems) + "\n",
            encoding="utf-8")

    for name, stems in [("train", train), ("val", val), ("test", test)]:
        write_paths(name, stems)
    for lvl, stems in subsets.items():
        write_paths(f"train_{lvl}", stems)

    # ---------------- Verification ----------------
    all_ok = True

    print("\n=== ULTRALYTICS LABEL RESOLUTION CHECK ===")
    # The check must be done over EXACTLY THE SAME lines that were WRITTEN TO THE FILE.
    # (The previous version checked path_of, but what was written to the file was
    #  the resolve()d form; the check passed and training blew up.)
    written = [l.strip() for l in
               (lists / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    raw_paths = [y for y in written if "/images/" not in y]
    # Decision 3.39: this used to be written[:200]. On the full data train is
    # ~8000 images -> 2.5% of it was being checked. The whole purpose of this
    # file is to catch silent label loss; a check done by sampling does not
    # serve that purpose. Path.exists() takes microseconds, 8000 of them will
    # not take a second.
    missing = [y for y in written
               if not Path(y.replace("/images/", "/labels/")).with_suffix(".txt").exists()]
    print(f"  lines in the list with no '/images/' segment: {len(raw_paths)}"
          f"{'  <-- PROBLEM (the symlink may have been resolved)' if raw_paths else ''}")
    print(f"  label not found via '/images/' -> '/labels/': {len(missing)}"
          f" / {len(written)}"
          f"{'  <-- PROBLEM' if missing else ''}")
    if written:
        print(f"  example line: {written[0]}")
    all_ok &= not missing and not raw_paths

    print("\n=== NESTEDNESS CHECK ===")
    chain = [("train_10", subsets[10], "train_25", subsets[25]),
             ("train_25", subsets[25], "train_50", subsets[50]),
             ("train_50", subsets[50], "train", train)]
    for an, a, bn, b in chain:
        ok = set(a).issubset(set(b))
        all_ok &= ok
        print(f"  {an} ⊂ {bn} : {'YES' if ok else 'NO  <-- PROBLEM'}")

    print("\n=== LEAKAGE CHECK (do the sets stay disjoint) ===")
    for an, a, bn, b in [("train", train, "val", val), ("train", train, "test", test),
                         ("val", val, "test", test)]:
        intersection = set(a) & set(b)
        all_ok &= not intersection
        print(f"  {an} ∩ {bn} : {len(intersection)} images"
              f"{'  <-- PROBLEM' if intersection else ''}")

    # ---------------- Distribution table ----------------
    rows = {}
    for name, stems in [("train", train), ("val", val), ("test", test)]:
        rows[name] = class_table(stems, df)
    for lvl in LEVELS:
        rows[f"train_{lvl}"] = class_table(subsets[lvl], df)
    tab = pd.DataFrame(rows).T
    tab.to_csv(sd / "distribution.csv")

    print("\n=== DISTRIBUTION PER SET (box counts) ===")
    print(tab.to_string())

    # whether the class shares are preserved across the levels
    shares = tab[CLASS_ORDER].div(tab["boxes"], axis=0).mul(100).round(1)
    print("\n=== CLASS SHARES (%) -- should be similar across the levels ===")
    print(shares.to_string())

    print(f"\nstem lists -> {sd}/        (commit these to git)")
    print(f"path lists  -> {lists}/   (gitignore, can be regenerated)")

    # ---------------- Small-data warnings ----------------
    warn_msgs = []
    if len(val) == 0 or len(test) == 0:
        warn_msgs.append(
            f"val={len(val)}, test={len(test)} -- the set is EMPTY. The strata are "
            "too small (there are only a few images per species combination). It "
            "resolves itself on the full data; on the demo package splitting is "
            "not meaningful.")
    levels = [("train", train)] + [(f"train_{l}", subsets[l]) for l in LEVELS]
    for (an, a), (bn, b) in zip(levels, levels[1:]):
        if len(a) == len(b):
            warn_msgs.append(f"{an} and {bn} are the same size ({len(a)}) -- the "
                             "narrowing stops because at least 1 image is kept from "
                             "every stratum. Not a problem on the full data.")
    if warn_msgs:
        print("\n=== WARNINGS ===")
        for u in warn_msgs:
            print(f"  ! {u}")

    print(f"\nStatus: {'OK' if all_ok else 'THERE IS A PROBLEM -- look above'}\n")


if __name__ == "__main__":
    main()