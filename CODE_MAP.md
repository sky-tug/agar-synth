# Code map — agar-synth

What each file does, in the order the data moves through them. Line counts are
from `git ls-files "*.py" "*.sh" "*.yaml" | xargs wc -l`, 11 September 2026 —
8,985 lines of tracked code.

Read `ARCHITECTURE.md` for *why* the pipeline is shaped this way; this file is
*where things are*. Every decision number cited here is written up in
`DECISIONS.md` (English summary) and `decisions.tr.md` (full record).

---

## 1. The chain

```
AGAR JSON
    │  convert.py            filters, reads real image size, writes YOLO labels
    ▼
data/processed/             images (symlinks) · labels · manifest.csv
    │  check_labels.py       draws boxes on images — verification BY EYE
    │  analyze_manifest.py   descriptive stats · layout stats · quality control
    │  make_splits.py        stratified, nested, leak-checked splits
    ▼
splits/                     stem lists, committed to git
    │
    ├─── REAL ARMS ──────────────────────────────────────────────────┐
    │        train.py  →  runs/<name>/weights/best.pt                │
    │                                                                │
    └─── SYNTHETIC ARMS ─────────────────────────────────────────┐   │
             layout.py    coordinates, before any diffusion      │   │
             mask.py      background + mask (synthetic ∪ erase)  │   │
             adapt.py     one LoRA per data level                │   │
             tiles.py     native-resolution tiling               │   │
             inpaint.py   coordinates → pixels                   │   │
             species_check.py  is the painted colony the right   │   │
                               species? (the Phase 4 gate)       │   │
                  └──► synthetic images + labels ────────────────┘   │
                                                                     ▼
                                                              train.py
                                                                     │
                                                                     ▼
                                             evaluate.py   metrics.py
                                                     │
                                     ┌───────────────┴───────────────┐
                                     ▼                               ▼
                            threshold.py                     collect.py
                     seed spread → significance         runs/ → results/
                          (the grid's first output)       (goes into git)
```

The single idea the whole repository rests on: **the label is born in
`layout.py`, not in `inpaint.py`.** The box coordinates are decided before
diffusion runs, so the annotation is free and exact.

---

## 2. Data pipeline

### `src/convert.py` — 278 lines
AGAR JSON → YOLO. Applies the dataset filters (lower-resolution subset,
countable plates, five microorganisms), drops any image with a defect or
contamination entirely, and **reads the image size from the file** rather than
assuming 2048×2048. Writes `labels/<id>.txt`, symlinks `images/<id>.<ext>`, and
produces `manifest.csv` — the source for splits, subsamples and the paper's
data table.

```bash
python src/convert.py --src /path/AGAR_representative --out data/processed
```

### `src/check_labels.py` — 160 lines
Draws the YOLO boxes onto the images so a human can look at them. Not optional:
a coordinate convention error (top-left vs centre, an unnormalised value, the
wrong image size) travels silently through training and resurfaces much later as
"why is the model not learning".

### `src/analyze_manifest.py` — 288 lines
Three jobs at once: descriptive statistics for the paper, layout statistics for
Phase 3 (count, size, radial position), and quality control (duplicate boxes,
boxes outside the plate, degenerate boxes). Writes CSV tables and plots under
`<data>/reports/`.

### `src/make_splits.py` — 254 lines
Produces train/val/test and the nested subsample lists. Four properties, each
deliberate:

- split at **image** level — pieces of one plate cannot land in different sets
- **stratified** by species combination — if *C. albicans* vanished from the 10%
  subset, level-to-level differences would measure class imbalance, not data volume
- **nested** — `train_10 ⊂ train_25 ⊂ train_50 ⊂ train`
- stem lists go to `splits/` and into git; the absolute-path lists Ultralytics
  reads are regenerated per machine

---

## 3. Measurement

### `src/eval/metrics.py` — 379 lines
The measurement core, **independent of Ultralytics**. It exists for four reasons:
Ultralytics' `val()` reports no size-stratified AP (and the paper's central claim
is about small colonies); counting metrics (MAE / sMAPE) do not exist there at
all; a second-detector control needs the *same* measurement code to run against
both; and when the protocol freezes, the measurement has to freeze with it.

AP is algorithmically identical to COCO — IoU 0.50:0.05:0.95, 101-point
interpolation, area-range ignore semantics. Verified against `pycocotools` to
1e-4 where it is installed.

### `src/eval/test_metrics.py` — 396 lines, 50 checks
Sanity cases whose answers are known by hand. Since decision 3.94, `check()`
asserts under pytest — before that the checks ran, printed, and told pytest
nothing, so a failing check still reported green.

### `src/eval/evaluate.py` — 467 lines
The evaluation CLI, run identically on every run. Takes either `--weights` (loads
the model and predicts) or `--pred-dir` (reads ready-made predictions, for the
second-detector comparison). Produces `summary.json`, `class_ap.csv`,
`size_ap.csv`, `counting.csv`, and on val a `conf_curve.csv`.

Two rules it enforces at runtime rather than in prose:

- the counting confidence threshold **may not be searched on test** — run val
  first, pass the selected value to the test run, or the program stops
- the threshold locked in `base.yaml` is the ruler for the whole grid; a
  deliberate deviation must say so with `--override-conf-thr` (decision 3.88)

Since 3.94 it also reports `mAP*_minGT` next to the COCO figures: a (class, size)
cell with fewer ground-truth boxes than `eval.min_gt_for_cell` is left out of the
size-band average. *S. aureus* has exactly one large box in val, and COCO-style
averaging let that single box carry a quarter of `mAP_large`.

### `scripts/threshold.py` — 344 lines
Measures `significant_diff_threshold` from the seed spread of one arm:
`2 × sigma` of mAP50-95, with a chi-square interval on sigma itself. Stops —
rather than warning — when the runs it was given are not comparable: different
level, a repeated seed, a smoke test, a mismatched locked threshold, or a
**resumed run** (the measurement that judges resume deviation cannot contain
one). Refuses to write into the config if the field is not `null`.

### `scripts/collect.py` — 219 lines
`runs/` is gitignored because the weights are gigabytes — but every run's
`run_metrics.json` and `eval*/summary.json` are a few KB and are **not
reproducible** without hundreds of GPU-hours. This script gathers them into
`results/`, which does go into git (decision 3.33).

---

## 4. Training

### `scripts/train.py` — 489 lines
A wrapper, not a convenience. `yolo train` would lose the four things the paper
needs: time, GPU-hours and peak VRAM recorded from the first run; the
augmentation policy coming from the config rather than the command line;
protection against Ultralytics defaults drifting between releases (every
augmentation field is written out explicitly, and a `null` is rejected rather
than silently becoming a default); and a single file to freeze in Phase 5.

Interruption-safe since decision 3.87: `runs/<name>/segments.jsonl` is fsynced
every epoch, and `--resume` accumulates time, epochs and VRAM across segments.

```bash
python scripts/train.py --config configs/base.yaml --level 100 --seed 0 --name G100_s0
python scripts/train.py --config configs/base.yaml --level 100 --seed 0 --name G100_s0 --resume
```

### `configs/base.yaml` — 218 lines
The frozen protocol: `batch 4` · `workers 2` · `imgsz 1280` · `nbs 64` ·
`patience 50`, the locked counting threshold `conf_thr: 0.35`, the measured
`significant_diff_threshold: 0.007112`, and `min_gt_for_cell: 10`. Each carries
the decision number that put it there.

### `configs/aug_b_classic.yaml` — 47 lines · `aug_c_copypaste.yaml` — 44 lines
The two classical-augmentation comparison arms, applied as overlays on top of
`base.yaml` via `--overlay`.

---

## 5. Generation

### `src/generate/layout.py` — 601 lines
Answers "where, how big, which species" — **with no diffusion involved**. Three
subcommands: `fit` extracts distribution parameters from real labels, `sample`
draws synthetic layouts from them, `validate` compares the result against real
data. The count distribution comes from the empirical histogram; the lognormal
family was chosen on ten demo plates and was disproved on the full data
(decision 3.84). Colony spacing uses a Strauss process whose interaction
parameter is calibrated by bisection against the measured touching rate, not
hand-tuned.

`--level` is mandatory with no default — an anti-leakage rule enforced by
argparse (decision 3.5).

### `src/generate/mask.py` — 288 lines
Turns a layout plan into what diffusion actually receives: a real background
plate from that level's train subset, and a binary mask with **two** kinds of
region. Synthetic discs, where new colonies will be painted — and erase regions,
covering the real colonies already on the background plate.

The second one is not cosmetic. AGAR has no empty plates; using a background
as-is would leave unlabelled real colonies in a "synthetic" image, which is
exactly the same thing as injecting false negatives into the training data.

### `src/generate/tiles.py` — 260 lines
Native-resolution tiling. SD 1.5 inpaints on a 512 px canvas; downscaling a
2048 px plate to fit would shrink a median *C. albicans* colony from 27.5 px to
6.8 px, so the generator would be asked to synthesise a 7-pixel object and the
upscaler would invent its texture. The small-object arms of the substitution
curve — the paper's central claim — would then be measuring an artefact. Hence
tiles, and hence the cost model is tiles-per-image × seconds-per-tile rather
than seconds-per-image (decision 3.51).

### `src/generate/inpaint.py` — 743 lines
Coordinates → pixels. Reads what `mask.py` produced, runs masked diffusion tile
by tile, composites, and writes the synthetic plate next to a copy of the label
that was already final before any of this started.

It also **measures**: `gen_metrics.json` records tiles, seconds per tile, peak
VRAM, model and step count, and `budget.py` consumes those numbers. At ~58,000
synthetic images the difference between a 20-step sampler and a distilled one is
roughly 290 GPU-hours, so this is not a detail (decision 3.53).

### `src/generate/adapt.py` — 487 lines
LoRA adaptation of the inpainting model — **one per data level**, four in total.
This is a leakage lock, not a tuning choice: if the generator feeding the "only
10% of real data" arm had been adapted on 100% of the data, it would have seen
the 90% that was supposedly held out, and the arm's claim would be void
(decisions 3.2 / 3.3). Training crops come from the train split only.

Measured outcome: the crop pool changes tenfold across levels, the size of the
adaptation does not (eval-loss drop 3.4–4.1% at every level) — so level
differences on the substitution curve will not be coming from the LoRA.

### `src/generate/species_check.py` — 612 lines
The Phase 4 gate, and the one the project failed. Every other check asks whether
a synthetic plate looks *realistic*; this one asks whether it is *correct* —
does a colony labelled *P. aeruginosa* actually look like *P. aeruginosa*?

The distinction matters because a plate full of convincing colonies of the wrong
species passes every other gate, and then per-class AP measures nothing while the
paper concludes "synthetic data does not substitute" when the true finding is
"the generator ignored the class".

What it found: three-axis separability between synthetic species is 92%, but
removing the texture axis drops it to 15%, and the two intervals do not overlap.
The classes separate along an axis the generator gets wrong. Reported as a
finding (decisions 3.86, 3.90).

### `src/generate/test_generate.py` — 590 lines, 65 checks
Unit tests for the generation half: inverse-CDF sampling, Strauss bisection, mask
geometry, level-mismatch refusal, the constant that guarantees zero label error.

### `src/generate/exploration/` — 9 scripts, 1,147 lines
One-off measurements, kept because their numbers are cited in `DECISIONS.md`:
plate circle geometry, size and radial distributions, count distribution
(`08`, which disproved the lognormal assumption), ghost-colony scan (`09`),
mask and plate-edge visual checks.

---

## 6. Three patterns

The repository is built on three ideas. Follow them when adding code.

### Pattern 1 — A methodological rule is a runtime check, not a comment

A rule written in prose gets forgotten. Here they raise `SystemExit`:

| rule | enforced in |
|---|---|
| the counting threshold may not be searched on test | `evaluate.py` |
| the locked `conf_thr` must match, or `--override-conf-thr` must be given | `evaluate.py` |
| a real run cannot proceed with an empty validation list | `train.py` |
| `null` in the augmentation config is rejected — it would become an Ultralytics default | `train.py` |
| synthetic layout cannot be generated without an explicit `--level` | `layout.py` |
| the background pool level must match the plan level | `mask.py` |
| a LoRA that failed to load stops generation instead of warning | `inpaint.py` |
| split lists must be verified nested and leak-free before use | `make_splits.py` |
| the seed-spread measurement refuses resumed or non-comparable runs | `threshold.py` |
| a failing sanity check must fail the test suite | `test_metrics.py`, `test_generate.py` |

The last row is the newest and was the most embarrassing: 115 checks across 30
test functions reported nothing to pytest until decision 3.94. Everything was
green, and that greenness meant nothing.

### Pattern 2 — A silent failure is worse than a loud one

Every serious wound this project has taken was silent:

- split lists pointing at the wrong directory — Ultralytics *warned and
  continued*, so the model learned "there are no objects here"
- a validation check that inspected a variable instead of the file actually
  written — it passed, training blew up
- `max_det` set in the config and never passed through
- a colony silently dropped from a generated plate, skewing the count distribution
- a test suite that could not fail

So the code counts and prints: how many boxes were clipped, how many images were
dropped and why, how many colonies could not be placed, whether the background
pool is species-biased, which size cells were too thin to average. When you add
code, ask: *can this quietly produce a wrong answer? Then add a counter.*

### Pattern 3 — Measurement replaces guesswork, and the old guess stays next to it

`imgsz=1280` was a guess, then a memorisation test measured it. The colony
touching rate was a guess, then 387 colonies measured it. The Strauss parameter
could have been hand-tuned; it is calibrated by bisection. Generation cost was
estimated at 105 GPU-hours and measured at 156.

And where the measurement is weak, that is reported too. The significance
threshold was 0.00399 on three seeds and 0.00711 on five; the three-seed value
is not deleted, because how far a small-sample sigma can stray is itself a
finding (decision 3.93).

---

## 7. Where the numbers live

```
results/table.csv          one row per (run, split) — the paper's results table
results/threshold.json     seed spread, sigma, threshold, sub-metric spreads
results/raw/<run>.json     metrics + evaluation merged, per run
runs/<name>/               weights and per-run outputs — GITIGNORED
  run_metrics.json           time · epochs · GPU-hours · peak VRAM · git commit
  segments.jsonl             per-epoch record, survives kill -9
  eval_val/summary.json      the numbers collect.py picks up
DECISIONS.md               every decision with its reasoning (English)
decisions.tr.md            the same log unabridged, working language
```

Run `python scripts/collect.py` after a batch of runs. `runs/` is gitignored;
without it the results exist nowhere else.

---

## 8. Not covered here

A code audit carried out at the end of Phase 2 found 25 issues across the
repository. Several were fixed in Phases 3–5 (results now under version control,
a pinned `requirements.lock`, `--resume`, config values actually being read),
but the list has not been re-verified item by item against the current code, so
it is not reproduced here as if it were current. Re-verifying it is an open task.
