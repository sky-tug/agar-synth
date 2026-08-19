#!/usr/bin/env python3
"""
GPU budget calculation -- the gate of Phase 2: "do the ~85 runs fit the budget?"

How it works:
  Starting from a single measured run (preferably G100, single seed, full
  training) it scales up the whole grid. The cost model:

      time  ~  n_train_images  x  n_epochs  x  (imgsz / measured_imgsz)^2

  The first factor is linear (one epoch = one pass), the third is quadratic
  (pixel count). This is an estimate; update it with --metrics as real
  measurements come in.

Usage:
    # after the G100 run
    python scripts/budget.py --metrics runs/G100_s0/run_metrics.json

    # by hand: 500 images, 90 seconds per epoch, 150 epochs
    python scripts/budget.py --n-train 500 --epoch-s 90 --epochs 150 --imgsz 1280

    # against the budget limit and with alternative scenarios
    python scripts/budget.py --metrics runs/G100_s0/run_metrics.json --budget 200 --scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GRID DEFINITION -- roadmap Phases 5/6/7
# (name, training set size / full-data ratio, which phase)
# In the "+S" arms the total training set is topped up to the full-data size.
#
# 12 August 2026 -- the target journal is settled: Journal of Engineering
# Sciences and Design (JESD, TR Dizin). The scope was cut accordingly:
#   - classic arm: 3 levels -> only G25 (the midpoint of the substitution curve)
#   - amount sweep: 0.5x/1x/2x/4x -> 0.5x and 2x  (4x was on its own the most
#     expensive item: 3.25 full-run equivalents x 3 seeds)
#   - second detector control: out of scope (the journal does not expect it)
#   - XAI arm ADDED (advisor approved) -- no training, inference only
# The old wide plan can still be selected below with --arms.
# ---------------------------------------------------------------------------

# (name, total training set share, SYNTHETIC share)
# The synthetic share is NOT DERIVED FROM THE NAME -- that is exactly why on
# 17 Aug 2026 the generation cost of the ablation arms was not being counted at
# all (their names contain no "+S", yet all three require a brand new synthetic
# set). Now it is written out explicitly.
MAIN_GRID = [
    ("G100",  1.00, 0.00), ("G50",   0.50, 0.00),
    ("G50+S", 1.00, 0.50), ("G25",   0.25, 0.00),
    ("G25+S", 1.00, 0.75), ("G10",   0.10, 0.00),
    ("G10+S", 1.00, 0.90), ("S100",  1.00, 1.00),
]

# Cut down: only at the G25 level. For answering the diffusion vs classic
# question the midpoint of the substitution curve is enough.
CLASSIC = [("B_G25", 0.25, 0.00), ("C_G25", 0.25, 0.00)]
CLASSIC_WIDE = [
    ("B_G50", 0.50, 0.00), ("B_G25", 0.25, 0.00), ("B_G10", 0.10, 0.00),
    ("C_G50", 0.50, 0.00), ("C_G25", 0.25, 0.00), ("C_G10", 0.10, 0.00),
]

# Synthetic amount sweep: G25 base (0.25) + synthetic.
# 1x = the amount that tops the real data up to the full data set (0.75); that
# one is already in the main grid as G25+S.
AMOUNT = [("G25+S_0.5x", 0.25 + 0.375, 0.375), ("G25+S_2x", 0.25 + 1.50, 1.50)]
AMOUNT_WIDE = [
    ("G25+S_0.5x", 0.25 + 0.375, 0.375),
    ("G25+S_2x",   0.25 + 1.50,  1.50),
    ("G25+S_4x",   0.25 + 3.00,  3.00),
]

# Advisor: "You may pick and use three different scenarios. The mask can be
# held fixed."
# [!] All three sit on the G25 base and all three require a SEPARATE synthetic
# set (A1 with the non-adapted model, A2 without real backgrounds, A3 with
# naive layout). The same images cannot be reused -- the whole point of the
# ablation is that the generation differs. Decision 3.18 (A3 redefined) also
# requires this.
ABLATION = [("A1_noLoRA", 1.00, 0.75), ("A2_background", 1.00, 0.75),
            ("A3_naive_layout", 1.00, 0.75)]

SECOND_DETECTOR = [("YOLO11_G25", 0.25, 0.00), ("YOLO11_G25+S", 1.00, 0.75)]

ARMS = {
    "main_grid":       (MAIN_GRID, 5, "Phase 6"),
    "classic":         (CLASSIC, 3, "Phase 6"),
    "amount_sweep":    (AMOUNT, 3, "Phase 6"),
    "ablation":        (ABLATION, 3, "Phase 7"),
    # --- non-default, enabled with --arms ---
    "classic_wide":    (CLASSIC_WIDE, 3, "Phase 6"),
    "amount_wide":     (AMOUNT_WIDE, 3, "Phase 6"),
    "second_detector": (SECOND_DETECTOR, 2, "Phase 7"),
}

# The arms computed by default (JESD scope)
DEFAULT_ARMS = ["main_grid", "classic", "amount_sweep", "ablation"]

# XAI arm (Grad-CAM/++): NO training, inference with the existing weights.
# Its cost is negligible next to training but it is not zero -- show it.
XAI_CONFIGS = ["G100", "G25", "G25+S", "S100"]     # models to be compared
XAI_IMAGES = 200                                    # sample from the test set


def fmt_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours*60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours/24:.1f} days"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", help="runs/<name>/run_metrics.json")
    ap.add_argument("--n-train", type=int, help="training image count of the measured run")
    ap.add_argument("--epoch-s", type=float, help="measured seconds per epoch")
    ap.add_argument("--epochs", type=int, default=150, help="epochs to be run in the grid")
    ap.add_argument("--imgsz", type=int, help="imgsz to be used in the grid")
    ap.add_argument("--measured-imgsz", type=int, help="imgsz the measurement was made at")
    ap.add_argument("--full-size", type=int, default=None,
                    help="full AGAR countable+lower-res image count. "
                         "If the measurement was made on a subset, it is scaled from here.")
    ap.add_argument("--budget", type=float, help="GPU-hours you have")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma separated. default (JESD scope): "
                         + ",".join(DEFAULT_ARMS)
                         + "  |  all: " + ",".join(ARMS))
    ap.add_argument("--xai", action="store_true",
                    help="also account for the XAI arm (Grad-CAM inference)")
    ap.add_argument("--xai-sec-per-image", type=float, default=1.5,
                    help="Grad-CAM++ inference time for one image (s)")
    ap.add_argument("--scenarios", action="store_true",
                    help="if it does not fit, also show the cut-down scenarios")
    # generation (Phase 3) cost
    # Decision 3.53: generation cost is NOT seconds-per-image. Inpainting runs at
    # native resolution in tiles (decision 3.51), so the cost is
    #     tiles_per_image x seconds_per_tile
    # Measured on generated layouts: 16.3 tiles/plate at 512, 9.7 at 768.
    # The old --gen-sec-per-image 8 was implicitly assuming ~1 tile and
    # understated production by 2-4x.
    ap.add_argument("--tiles-per-image", type=float, default=16.3,
                    help="measured tiles per plate (512px tile -> 16.3, 768 -> 9.7). "
                         "src/generate/tiles.py tile_stats() reports this.")
    ap.add_argument("--sec-per-tile", type=float, default=0.0,
                    help="one inpainting pass for one tile. THIS IS THE BUDGET "
                         "LEVER: ~1.5 s at 20 steps, ~0.4 s at 4-8 distilled steps.")
    ap.add_argument("--gen-sec-per-image", type=float, default=0.0,
                    help="inpainting time of one synthetic image (s)")
    ap.add_argument("--lora-hours", type=float, default=0.0,
                    help="one LoRA training per level (hours)")
    ap.add_argument("--lora-count", type=int, default=4,
                    help="how many separate LoRAs will be trained. Decision 3.2: a "
                         "separate LoRA per level -> lora_10, lora_25, lora_50, lora_100 = 4")
    ap.add_argument("--synth-per-seed", action="store_true",
                    help="generate a SEPARATE synthetic set for each seed. "
                         "Statistically cleaner (the randomness of generation "
                         "enters the variance too) but it multiplies the "
                         "generation cost by the number of seeds.")
    args = ap.parse_args()

    # ---------------- measured baseline ----------------
    if args.metrics:
        m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        n_measured = m["n_train"]
        epoch_s = m["sec_per_epoch"]
        measured_imgsz = m["imgsz"]
        source = f"{m['run']} ({m.get('epochs_actual')} epochs, " \
                 f"{m.get('peak_vram_gb')} GB VRAM)"
        if m.get("smoke_test"):
            print("! WARNING: this is a SMOKE TEST measurement. On short runs the\n"
                  "  per-epoch time is misleading because of warm-up. Wait for the\n"
                  "  full G100 training.\n")
    elif args.n_train and args.epoch_s:
        n_measured, epoch_s = args.n_train, args.epoch_s
        measured_imgsz = args.measured_imgsz or args.imgsz or 1280
        source = "values entered by hand"
    else:
        sys.exit("ERROR: give --metrics OR (--n-train and --epoch-s).")

    imgsz = args.imgsz or measured_imgsz
    measured_imgsz = args.measured_imgsz or measured_imgsz
    n_full = args.full_size or n_measured

    # the seconds of one run at "full data size"
    s_per_img_epoch = epoch_s / max(n_measured, 1)
    imgsz_factor = (imgsz / measured_imgsz) ** 2
    full_run_s = s_per_img_epoch * n_full * args.epochs * imgsz_factor

    print("=" * 70)
    print("GPU BUDGET CALCULATION")
    print("=" * 70)
    print(f"measurement source   : {source}")
    print(f"measured             : {n_measured} images, {epoch_s:.1f} s/epoch, "
          f"imgsz {measured_imgsz}")
    print(f"grid assumption      : {n_full} images (full data), {args.epochs} epochs, "
          f"imgsz {imgsz}")
    if imgsz_factor != 1:
        print(f"imgsz scaling        : x{imgsz_factor:.2f}  "
              f"({measured_imgsz} -> {imgsz}, quadratic)")
    print(f"1 full-size run      : {fmt_hours(full_run_s/3600)}")
    print()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    total_hours, total_runs = 0.0, 0
    rows = []

    for arm in selected:
        if arm not in ARMS:
            sys.exit(f"ERROR: unknown arm '{arm}'. Options: {list(ARMS)}")
        configs, seed, phase = ARMS[arm]
        share = sum(f for _, f, _ in configs)
        runs = len(configs) * seed
        hours = share * seed * full_run_s / 3600
        total_hours += hours
        total_runs += runs
        rows.append((arm, phase, len(configs), seed, runs, share * seed, hours))

    w = max(len(r[0]) for r in rows)
    print(f"{'arm':<{w}}  {'phase':<8} {'cfg':>5} {'seed':>5} {'runs':>5} "
          f"{'full-run eq.':>13} {'GPU-hours':>10}")
    print("-" * 70)
    for name, phase, n_cfg, sd, runs, equiv, hours in rows:
        print(f"{name:<{w}}  {phase:<8} {n_cfg:>5} {sd:>5} {runs:>5} "
              f"{equiv:>13.2f} {hours:>10.1f}")
    print("-" * 70)
    print(f"{'TOTAL (training)':<{w}}  {'':<8} {'':>5} {'':>5} {total_runs:>5} "
          f"{'':>13} {total_hours:>10.1f}")

    # ---------------- generation cost (Phase 3) ----------------
    gen_hours = 0.0
    # sec-per-tile wins if given; otherwise fall back to the old per-image number
    per_image_sec = (args.tiles_per_image * args.sec_per_tile
                     if args.sec_per_tile else args.gen_sec_per_image)
    if per_image_sec or args.lora_hours:
        # The synthetic share is now read from the arm definition, not from the name.
        synth_share = 0.0
        for arm in selected:
            configs, _, _ = ARMS[arm]
            synth_share += sum(sp for _, _, sp in configs)

        # If the synthetic set is shared across seeds it is generated once.
        # Generating a separate one per seed is statistically cleaner (the
        # randomness of generation enters the variance too) but it multiplies
        # the cost by the number of seeds.
        multiplier = 1
        if args.synth_per_seed:
            multiplier = max(sd for _, sd, _ in (ARMS[a] for a in selected))

        n_synth = int(round(synth_share * n_full)) * multiplier
        gen_hours = (n_synth * per_image_sec / 3600
                     + args.lora_count * args.lora_hours)
        n_tiles = n_synth * args.tiles_per_image if args.sec_per_tile else None
        print()
        if args.sec_per_tile:
            # Decision 3.51/3.53: inpainting is tiled at native resolution.
            print(f"generation (Phase 3) : {n_synth} synthetic images "
                  f"x {args.tiles_per_image} tiles x {args.sec_per_tile} s/tile")
            print(f"                       = {n_tiles/1e6:.2f}M inpainting passes"
                  + (f"   [separate set per seed, x{multiplier}]" if multiplier > 1
                     else "   [single synthetic set, shared by the seeds]"))
            print(f"                       + {args.lora_count} LoRA x {args.lora_hours} h")
        else:
            print(f"generation (Phase 3) : {n_synth} synthetic images "
                  f"x {args.gen_sec_per_image} s  +  {args.lora_count} LoRA "
                  f"x {args.lora_hours} h"
                  + (f"   [separate set per seed, x{multiplier}]" if multiplier > 1
                     else "   [single synthetic set, shared by the seeds]"))
            print("  [!] --gen-sec-per-image is the OLD cost model and assumes ~1 tile "
                  "per image.\n      Inpainting is tiled (decision 3.51); use "
                  "--sec-per-tile instead.")
        print(f"                     = {gen_hours:.1f} GPU-hours")

    # ---------------- XAI arm (no training, inference) ----------------
    xai_hours = 0.0
    if args.xai:
        xai_hours = len(XAI_CONFIGS) * XAI_IMAGES * args.xai_sec_per_image / 3600
        print()
        print(f"XAI (Grad-CAM/++)    : {len(XAI_CONFIGS)} models "
              f"({', '.join(XAI_CONFIGS)}) x {XAI_IMAGES} images "
              f"x {args.xai_sec_per_image} s")
        print(f"                     = {xai_hours:.2f} GPU-hours   "
              f"(no training -- inference with the existing weights)")

    grand_total = total_hours + gen_hours + xai_hours
    print()
    print(f"GRAND TOTAL          : {fmt_hours(grand_total)}  ({total_runs} training runs)")

    # ---------------- budget comparison ----------------
    if args.budget:
        print()
        print("=" * 70)
        if grand_total <= args.budget:
            print(f"GATE: FITS.  {grand_total:.1f} / {args.budget:.0f} GPU-hours "
                  f"({100*grand_total/args.budget:.0f}% full)")
            print("You can move on to Phase 3.")
        else:
            print(f"GATE: DOES NOT FIT.  {grand_total:.1f} / {args.budget:.0f} GPU-hours "
                  f"-- {grand_total - args.budget:.1f} hours over")
            print("\nThe cut-down order suggested by the roadmap "
                  "(the main grid is left for LAST, it is the backbone of the paper):")
            for name, recipe, gain in [
                ("amount_sweep", "seed 3 -> 1", None),
                ("classic", "seed 3 -> 2", None),
                ("ablation", "single level, seed 3 -> 2", None),
                ("second_detector", "single configuration", None),
                ("main_grid", "seed 5 -> 3 (LAST RESORT -- with n=3 the std defence weakens)",
                 None),
            ]:
                if name in selected:
                    print(f"  - {name:<16} {recipe}")
        print("=" * 70)

    if args.scenarios:
        print("\n" + "=" * 70)
        print("SCENARIOS")
        print("=" * 70)
        # CAUTION: epochs and imgsz affect the TRAINING cost only.
        # Generation (inpainting) and XAI inference are independent of them --
        # they stay fixed.
        fixed = gen_hours + xai_hours
        side = 0.0
        if "classic" in selected:
            side += sum(f for _, f, _ in CLASSIC)
        if "amount_sweep" in selected:
            side += sum(f for _, f, _ in AMOUNT)
        scenarios = [
            ("full plan (the one above)", grand_total),
            ("main grid seed 5->3", grand_total - (
                sum(f for _, f, _ in MAIN_GRID) * 2 * full_run_s / 3600)
                if "main_grid" in selected else grand_total),
            ("side arms seed 3->2", grand_total - side * 1 * full_run_s / 3600),
            ("epoch 150->100", total_hours * 100 / args.epochs + fixed),
            ("imgsz 1280->1024 (! small colony risk)",
             total_hours * (1024 / imgsz) ** 2 + fixed),
            ("imgsz 1280->640  (!! C.albicans 8.6 px, not recommended)",
             total_hours * (640 / imgsz) ** 2 + fixed),
        ]
        for name, h in scenarios:
            marker = ""
            if args.budget:
                marker = "  <- fits" if h <= args.budget else ""
            print(f"  {name:<45} {fmt_hours(h):>10}{marker}")
        print("=" * 70)

    print("\nNote: this is an ESTIMATE. Update it with --metrics every time a new")
    print("measurement arrives; the table of section 6 (computational cost) will be")
    print("written from the real measurements.")


if __name__ == "__main__":
    main()
