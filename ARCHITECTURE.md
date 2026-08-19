# Architecture

What every file does and — more importantly — **why it is written the way it is**.

`README.md` says what the project measures. `decisions.md` is the running log, in
Turkish, of every decision in the order it was taken. This document sits between
them: it is the map you read before changing anything.

Three ideas run through the whole repository. They are stated in the README and
enforced here:

1. **A methodological rule is a runtime check, not a comment.** There are
   **34 hard guards** across the CLIs. They call `sys.exit`. The list is in
   [§8](#8-every-runtime-guard).
2. **A silent failure is worse than a loud one.** Every stage counts what it
   dropped, clipped, skipped or could not place, and prints it.
3. **Measurement replaces guesswork.** Where a number was once a guess, the file
   that uses it now measures it, and the old guess is recorded next to the
   measurement so the size of the error stays visible.

---

## 0. The pipeline

```
   AGAR json + jpg
        |
        |  src/convert.py            filters, YOLO labels, manifest.csv
        v
   data/processed/  ──►  src/check_labels.py      draw the boxes, LOOK at them
        |             ──►  src/analyze_manifest.py  stats + quality control
        |
        |  src/make_splits.py        stratified, nested, leak-checked splits
        v
   splits/*.txt (git)  +  data/processed/lists/*.txt (absolute, per machine)
        |
        +──────────────────────────────┐
        |                              |
        |  REAL branch                 |  SYNTHETIC branch
        v                              v
   scripts/train.py              src/generate/layout.py    where / what / how big
        |                              |
        |                              v
        |                        src/generate/mask.py      mask + background + label
        |                              |
        |                              v
        |                        src/generate/adapt.py     LoRA for THIS level
        |                              |
        |                              v
        |                        src/generate/inpaint.py   tiles -> pixels
        |                              |    (uses tiles.py)
        |                              v
        |                        src/generate/species_check.py   is it the RIGHT species?
        |                              |
        +──────────────────────────────┘
        v
   src/eval/evaluate.py  (val -> pick conf_thr, then test)   uses src/eval/metrics.py
        |
        v
   scripts/collect.py    runs/ -> results/table.csv  (committed)
   scripts/budget.py     one measured run -> the whole grid in GPU-hours
```

Canonical order, as printed by each stage's own `NEXT:` block:

```
setup.sh → convert.py → make_splits.py → train.py
         → evaluate.py --split val → evaluate.py --split test --conf-thr <v>
         → budget.py → collect.py
```

---

## 1. Data pipeline

### `src/convert.py`

AGAR's JSON annotations → YOLO labels + an image tree + `manifest.csv`.

```
python src/convert.py --src data/AGAR_representative --out data/processed
                      [--background lower-resolution] [--copy]
```

`CLASS_ORDER = [S.aureus, B.subtilis, P.aeruginosa, E.coli, C.albicans]` is fixed
for the whole project — class ids 0–4. Everything downstream (labels, configs,
metrics, the paper's tables) is indexed by it, so changing the order silently
mislabels the entire dataset.

Four things here are less obvious than they look:

- **Image size is read from the image**, not assumed to be 2048×2048. It is not
  in the JSON, and a wrong W/H turns every normalised box into a plausible-looking
  wrong box.
- **`normalize_class()` strips spaces**: AGAR contains both `S. aureus` and
  `S.aureus`.
- **An image containing an artifact class is dropped whole**, not just the box.
  Deleting the box and keeping the image teaches the detector "there is no object
  here" — the same silent poisoning that `make_splits.py` guards against from the
  other end.
- **Countability is inferred, and the reasons are separated** (decision 3.43).
  AGAR gives colony-level labels only for countable images; empty plates and
  uncountable plates both arrive with an empty `labels` list. `colonies_number`
  splits them into `dropped_empty` / `dropped_uncountable` /
  `dropped_unlabelled_unknown`, because the paper's data section has to report
  *how many images were dropped and why*, not one lumped number.

The summary block prints fifteen counters, including `boxes_clipped` and
`boxes_dropped`. Boxes are clipped to the image; a post-clip box thinner than
1 px is dropped as broken.

Decision 3.44 hides in the symlink path: `Path.exists()` follows a symlink, so a
link into a moved AGAR folder reports `False`, `symlink_to` is retried, and the
program dies with `FileExistsError`. The code checks `is_symlink() and not exists()`
and unlinks first.

Writes `<out>/labels/*.txt`, `<out>/images/*` (symlinks by default),
`<out>/manifest.csv`, `<out>/classes.txt`.

### `src/check_labels.py`

Draws the converted boxes back onto the images so a human can look at them.

```
python src/check_labels.py --data data/processed [--n 30] [--seed 0] [--class-name S.aureus]
```

This step exists because a coordinate error — top-left instead of centre, an
un-normalised value, a swapped W/H — travels silently through training and comes
back weeks later as *"why is my model not learning"*. The script therefore ends
with an instruction block telling you what to look for, and the sentence
**"These 10 minutes save 3 days further down the road."**

Sampling is class-balanced (`n // 10` per class first, remainder from a shuffled
pool) so rare classes cannot be absent from what you inspect, and seeded so the
sample is reproducible.

Five automatic warnings run alongside the drawing: `missing_file`,
`size_mismatch`, `out_of_range`, `box_too_small` (< 4 px), and `outside_edge`.
The last one is decision 3.45: `out_of_range` checks centre and width
independently, so `xc=0.99, bw=0.10` passes while extending to 1.04 of the image.
`outside_edge` verifies the thing `convert.py`'s clipping is supposed to
guarantee — a check on a guarantee, not a duplicate.

### `src/analyze_manifest.py`

Descriptive statistics for the paper, layout statistics for Phase 3, and quality
control.

```
python src/analyze_manifest.py --data data/processed [--iou-dup 0.5] [--tiny-px 8]
```

Five printed sections (per-class summary, COCO size breakdown, colonies per
image, species combinations, quality control) and six files under
`<data>/reports/`.

Two decisions live here and both are about a measurement that could not fail:

- **Decision 3.49 — the radius unit.** `radius` used to be normalised by the
  image half-width, on the false claim that "the plate fills the square". The
  measured plate radius is `0.465 × width` (decision 3.10), so the true plate
  edge sat at 0.930 in that unit while the "outside the plate" checks were at
  0.95 and 1.0 — **the check could never fire**. Worse, `layout.py` normalised
  the same word by 0.465, so one term meant two things in two files. Both are now
  *relative to the plate radius*: 0 = centre, 1 = plate edge. `radius_outer`
  is reported separately, because a colony whose centre is inside the plate can
  still have its edge sticking out.
- **Decision 3.50 — the duplicate threshold.** `--iou-dup 0.5` was a guess.
  Real colonies genuinely touch (39 % of them, overlap depth up to 0.63,
  decision 3.12), so a high-IoU pair can be legitimate. In the demo package the
  real maximum IoU is **0.450** — a margin of 0.05. The script now prints the
  observed IoU percentiles (p50 … p99.9, max) and suggests a threshold from the
  tail instead of accepting the guess.

### `src/make_splits.py`

Stratified image-level train/val/test splits, plus **nested** training subsamples.

```
python src/make_splits.py --data data/processed [--val 0.15] [--test 0.15] [--seed 42]
```

`LEVELS = [50, 25, 10]`, applied in that order so that
`train_10 ⊂ train_25 ⊂ train_50 ⊂ train`. Nesting is what makes the substitution
curve a curve: if the 10 % subset were an independent draw, a difference between
levels could come from *which* images were drawn rather than from *how many*.

Stratification is by **species combination** (the manifest `classes` string). If
`C.albicans` vanished from the 10 % subset, the gap between levels would be class
imbalance wearing the costume of data volume.

Two `CAUTION` comments in this file are worth more than the code around them:

- **Ultralytics resolves labels by replacing the last `/images/` segment with
  `/labels/`.** The lists must therefore point at `data/processed/images/`, never
  at the raw AGAR tree. When they don't, Ultralytics *warns and continues*, and
  the model learns "there are no objects in these images." Described in the file,
  accurately, as "a silent, fatal error" (decision 2.0).
- **Do not call `.resolve()`** when writing those paths. The files under
  `data/processed/images/` are symlinks into the raw tree, and resolving them
  removes the `/images/` segment — reintroducing exactly the bug above.

The verification then reads `lists/train.txt` **back from disk** rather than
checking the in-memory variable. The earlier version checked the variable while
the file held the resolved form: the check passed and training blew up.

Decision 3.39: that missing-label check used to sample the first 200 lines. On
the full train set (~8000 images) that is 2.5 %. Since catching silent label loss
is the file's entire purpose, sampling defeated it; `Path.exists()` takes
microseconds and the check is now exhaustive.

Three verification blocks print explicitly: label resolution, nestedness, and
leakage (train∩val, train∩test, val∩test). Output goes to two places —
`splits/*.txt` (stems, committed to git, reproducibility) and
`data/processed/lists/*.txt` (absolute paths, gitignored, regenerated per machine).

---

## 2. Measurement

### `src/eval/metrics.py`

COCO-identical AP plus counting metrics. A library, no CLI, **no runtime guards** —
the measurement core stays pure and all enforcement lives at the CLI boundary in
`evaluate.py`.

It is written in-house for four reasons, all in the module docstring: Ultralytics
`val()` gives no size-stratified AP (and small colonies are the paper's central
claim); MAE/sMAPE do not exist there at all; the second-detector control must run
the *same* measurement code, because a detector-dependent metric invalidates any
cross-detector comparison; and the primary metric must be inside the repository so
it freezes with the protocol in Phase 5.

Protocol constants — changing any of them requires a `decisions.md` entry:

| constant | value |
|---|---|
| `IOU_THRS` | `linspace(0.5, 0.95, 10)` |
| `REC_THRS` | `linspace(0, 1, 101)` — 101-point interpolation |
| `AREA_RANGES_COCO` | small `< 32²`, medium `< 96²`, large — **in original image pixels, not `imgsz`** |

That last note is load-bearing. `evaluate.py` reads W/H from the image file, so
the size buckets stay in original pixels; if they ever followed `imgsz` the whole
size breakdown — the paper's main argument — would silently shift. Calibration:
*S.aureus* median 29 px → 841 px² → small; *E.coli* median 128 px → large.

Three subtleties:

- **The ignore mechanism.** When computing small-object AP the large GTs are not
  deleted, they are marked ignore: no reward, no penalty. Deleting them would turn
  a correct large detection into a false positive and unfairly depress the
  small-object score. Non-ignored GTs are sorted first so matching prefers them.
- **Detections are ranked globally.** Confidences are concatenated across all
  images and sorted once before the PR curve. Doing it per-image and averaging
  gives a different, wrong number — "the most common bug in hand-written AP code".
- **sMAPE, not MAPE**, and **ME alongside MAE**. Plain MAPE diverges on an empty
  plate; sMAPE is bounded 0–200 %. And MAE hides bias: telling a microbiologist
  "the model systematically undercounts" is more useful than "MAE is 4.2".

### `src/eval/evaluate.py`

The single evaluation CLI. Two modes: `--weights` (runs an Ultralytics
checkpoint) or `--pred-dir` (reads ready-made YOLO predictions, for the
second-detector comparison) — exactly one, enforced.

```
# VAL first: it selects the counting confidence threshold
python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \
    --split val --out runs/G100_s0/eval_val --tag G100_s0
# then TEST, with that threshold
python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \
    --split test --conf-thr <value> --out runs/G100_s0/eval_test --tag G100_s0
```

**The leakage guard is the point of this file.** `--split test` without
`--conf-thr` exits. Searching the counting threshold on the test set is leakage;
the rule is written in `metrics.py`'s docstring and enforced here. `--split train`
also requires it — not leakage, but meaningless, and it used to leave
`selected=None` and raise a `TypeError` deep inside `counting_metrics`.

**Predictions are matched by path, never by position.** The earlier
`zip(images, boxes_all)` had a failure mode with no error message: if Ultralytics
skipped one image, zip truncated silently and every image after that point was
scored against the wrong prediction. Now results are indexed into `by_stem`, and
a length mismatch is a hard stop.

Decision 3.48: `max_det` and `nms_iou` used to be hard-coded here while
`train.max_det` / `train.nms_iou` in `base.yaml` were read nowhere — changing the
config did nothing. In Phase 5 the thing that gets frozen has to be the thing
that actually runs, so they come from `--config` now and the constants are only a
fallback (`FALLBACK_MAX_DET = 1000` — AGAR plates reach 125 colonies).

`MIN_CONF = 0.001`: predictions below this are never recorded. Kept low because
COCO AP uses every detection; the counting threshold is applied separately.

Writes `summary.json`, `class_ap.csv`, `size_ap.csv`, `counting.csv`, and — only
on val, only when the threshold was searched — `conf_curve.csv`.

### `src/eval/test_metrics.py`

36 checks in 11 groups (42 when `pycocotools` is installed). Run it directly; it
exits non-zero on any failure and `setup.sh` gates on that.

The principle: do not trust a metric you wrote yourself until you have tested it
with cases whose answer you know by hand. So the checks are hand-computable —
half overlap must give IoU exactly 1/3; a 10 px shift gives 0.8181818 and
therefore AP50-95 = 7/10; MAE of (10,20,30) vs (12,18,30) is 4/3; sMAPE of a
0-vs-3 plate is exactly 200.

Group 11 cross-validates six statistics (`mAP50-95`, `mAP50`, `mAP75`, and the
three size buckets) against `COCOeval.stats` to **1e-4**, and skips cleanly when
pycocotools is absent.

Group 10 exists because of decision 3.38: `read_yolo_txt` was not being tested at
all. Every other check builds the annotation object directly with pixel boxes,
skipping the one place where the normalised→pixel conversion lives. The test
image is deliberately **800×400** — on a square image a W/H swap is invisible.

### `src/generate/test_generate.py`

61 checks in 19 groups, covering `layout.py`, `mask.py`, `tiles.py` and
`species_check.py`. It exists (decision 3.37) because `layout.py` contains at
least as much treacherous mathematics as `metrics.py` — empirical inverse-CDF
sampling, a Strauss gamma bisection, lognormal parameter estimation — and it
produces the data of the paper. `layout.py validate` is a *comparison report*,
not a unit test: a wrong quantile interpolation can leave the median correct
while breaking the tail.

Notable groups: 16 asserts the no-split tiling invariant on generated plates;
17 asserts that compositing leaves every pixel outside the mask byte-identical;
18 checks the class-fidelity measurement itself, including that separability
*collapses* when two classes are made identical — a verdict that cannot tell
"separated" from "collapsed" would report a false pass.

---

## 3. Training

### `configs/base.yaml`

The frozen protocol. Header rule: *every number here is a decision; if you change
one, write it into `decisions.md` and re-run the affected runs. A tweak made
after seeing the results reads as cherry-picking.*

**`imgsz: 1280` is the single most critical hyperparameter.** AGAR lower-res
images are 2048². Measured colony sizes in original pixels: *C.albicans* median
27.5 (min 14), *S.aureus* median 29.0, *E.coli* median 128. Projected to the
training resolution:

| imgsz | C.albicans median |
|---|---|
| 640 | 8.6 px — the P5 head's stride is 8; detection is near-impossible |
| 1024 | 13.8 px |
| **1280** | **17.2 px — chosen** |
| 1536 | 20.6 px |

Cost is quadratic in `imgsz`. **If VRAM is short, lower the batch, never the
`imgsz`.**

**The augmentation policy is a control, not a preference.** Ultralytics ships
`mosaic=1.0` by default. Do nothing, and the G50/G25/G10 control groups already
use classic augmentation — the "diffusion vs classic" comparison is blurred
before it starts, because the gain from diffusion cannot be separated from the
gain from mosaic. So the main grid runs **minimum augmentation**: only
label-preserving transforms that suit plate physics.

| on | why |
|---|---|
| `hsv_h/s/v` 0.015 / 0.7 / 0.4 | illumination and medium colour really do vary |
| `fliplr`, `flipud` 0.5 | an agar plate has no canonical orientation |

| off (explicitly `0.0`) | why |
|---|---|
| `degrees`, `translate`, `scale`, `shear`, `perspective` | colony size is a **class cue** (S.aureus small, E.coli large). Perturbing scale corrupts the class signal. |
| `mosaic`, `close_mosaic` | merges 4 plates into one frame — breaks scale, destroys the plate boundary, distorts colony density. Especially harmful for counting. |
| `mixup` | transparent superposition of two plates is microbiologically meaningless |
| `copy_paste` | **this is Baseline C's arm.** On in the main grid, "diffusion vs copy-paste" becomes undefined. |
| `cutmix` | added in ultralytics 8.4 — written out so a new field cannot default in silently |
| `erasing`, `bgr` | classification augmentations, unnecessary here |

Every field is written explicitly, including the zeros, because Ultralytics
defaults change between releases and an implicit default is irreproducible
(decision 2.13).

`eval.conf_thr: null` and `eval.significant_diff_threshold: null` are documented
placeholders to be filled from the first val run and the first seed spread.
`seeds.main_grid` is 5 values, `side_arms` 3 — approved by the advisor, who also
asked for min/max/mean in the plots and mean/std in the tables.

### `scripts/train.py`

Training wrapper. Four reasons it exists rather than `yolo train`: the time,
GPU-hours and peak VRAM for the paper's cost section cannot be measured
retroactively; the augmentation policy must come from a file, not a command line,
or "what was on in which run" stays unanswerable; Ultralytics defaults drift, so
all fields are written out; and Phase 5 freezes this file together with the
config.

```
python scripts/train.py --config configs/base.yaml --level 100 --seed 0 --name G100_s0
                        [--overlay configs/aug_b_classic.yaml] [--smoke] [--dry-run] [--resume]
```

**Six guards**, and the order of one of them is itself a lesson.
`AUGMENT_NULL_ALLOWED = {"auto_augment"}` (decision 3.40): null fields used to be
filtered out silently, whereupon Ultralytics used its own default — so
`mosaic: null` became `mosaic=1.0` and the augmentation policy collapsed without
a word. Any other null now stops the program. That check sits in **pre-flight**,
before the `--dry-run` early return, because a check that only fires on a real
run is a check that finds the problem 40 minutes late.

Decision 3.41: the `logging` block in `base.yaml` was read nowhere. Once you have
run `wandb login`, Ultralytics auto-detects it and logs into *its own* default
project, mixing smoke tests into real runs. `WANDB_MODE` / `WANDB_PROJECT` are now
set explicitly.

Decision 3.42: `--resume` must go through this wrapper. Calling
`YOLO(...).train(resume=True)` directly never writes `run_metrics.json`, and the
time, VRAM and commit are lost. The record carries `"resumed": true` so partial
measurements are visible as partial.

Empty-val fallback exists only under `--smoke`, and prints three yellow lines
saying the resulting mAP is meaningless. It is never reported.

Writes `runs/<name>/dataset.yaml` and `runs/<name>/run_metrics.json` — the latter
holds `gpu_hours`, `peak_vram_gb`, `sec_per_epoch`, `git_commit` and the full
augmentation dict, which is what `budget.py` and `collect.py` consume.

---

## 4. Generation (Phase 3)

The chain is: decide the layout → build the mask → adapt the model → paint only
inside the mask → check that what was painted is the right species.

Because the coordinates were ours to begin with, **the label is free and exact**.
That claim rests on one constant, `SYNTH_MARGIN = 1.00`, and one guarantee, the
mask gate in `tiles.composite`.

### `src/generate/layout.py`

Where the colonies go, how many, how big, of what species.

```
python src/generate/layout.py fit      --level 25 --list data/processed/lists/train_25.txt --out src/generate/layout_25.json
python src/generate/layout.py sample   --level 25 --params src/generate/layout_25.json --count 400 --seed 0 --out data/synthetic/s25
python src/generate/layout.py validate --params src/generate/layout_25.json --list ... --generated data/synthetic/s25
```

`--level` is **required with no default** (decision 3.5). A layout fitted on the
100 % subset and used to generate data for the 10 % arm would leak the full
data's statistics into the small-data arm — one of the three leakage channels
(decision 3.1–3.9: LoRA weights, background plate, **layout statistics**).

| constant | value | meaning |
|---|---|---|
| `PLATE_R_RATIO` | 0.465 | measured plate radius / image width (decision 3.10) |
| `MAX_OVERLAP` | 0.63 | deepest real overlap observed; the sampler may not exceed it |
| `ELLIPSE_JITTER` | 0.03 | colonies are not perfect circles |
| `PLACEMENT_TRIES` | 400 | attempts before a colony is declared unplaceable — and then it is **reported**, not dropped |
| `CALIB_PLATES` | 250 | plates simulated per bisection step when calibrating gamma |

**Colonies touch, and that had to be modelled** (decisions 3.12, 3.19). 39 % of
real colonies touch a neighbour. A naive "no overlap" sampler would make synthetic
plates *cleaner* than real ones; the detector would then fail on real occlusion,
and the failure would be read as "diffusion does not work". So placement is a
**Strauss point process**: a candidate touching another colony is accepted with
probability `gamma`, rejected and retried with `1 - gamma`. Gamma is not
hand-tuned — `fit` calibrates it by bisection against the measured touching rate,
using common random numbers so the bisection is not chasing Monte-Carlo noise. On
the demo package it lands at `gamma = 0.317`.

The touching rate ships with a **plate-level** bootstrap interval and a
`provisional: true` flag, because ten demo plates pick a distribution *family*,
not its parameters. `fit` prints that warning every time.

`--naive` implements ablation **A3** (decision 3.18). A3 was originally "random
placement", which would have been an empty ablation — real placement *is* nearly
random (Clark–Evans ≈ 1.0). Redefined as *naive* random: no plate circle, pooled
sizes across species, no touching constraint. A test proves the two differ (18 %
of naive colonies fall outside the plate, versus 0 %).

### `src/generate/tiles.py`

The tiling layer, and one invariant.

SD 1.5 works at 512 px. Downscaling a 2048 px plate to 512, inpainting and
upscaling would turn a 27.5 px *C.albicans* colony into 6.8 px — the `imgsz=1280`
argument again, but harsher: here the **generator** would be asked to synthesise a
7 px object and the upscaler would invent its texture. The small-class arms would
be measuring an artefact. So generation happens at **native resolution, in tiles**.

**The invariant (decision 3.52): no colony is ever split across two tiles.** Two
halves denoised in separate passes from different noise do not agree; a seam runs
through the colony and the detector learns a colony-shaped object with a line in
it. `place_tiles()` refuses to emit a layout that splits a colony, and
`test_generate.py` asserts it on generated plates.

The greedy placement recentres each window on what it absorbed, then **clamps** it
to the image. The first version's comment claimed the clamp "can only help". It
cannot: near the border the recentred window moves and a colony at the far edge of
the group falls outside. The assertion caught it on the first test run. Members
are now re-selected after clamping, with a seed-only fallback.

`composite(base, generated, mask, tile, feather=32)` is **the mask gate**: pixels
are written only where the mask says diffusion was allowed to paint. Models do
change pixels outside their mask; those changes never reach the plate. This is
what makes "the untouched background is byte-identical to the real plate" a fact
rather than a hope.

`DEFAULT_TILE = 512`, `CONTEXT_PAD = 24` (padding affects the **tile**, never the
mask, so decision 3.24 is untouched).

### `src/generate/mask.py`

Plan + real plate → mask, background, label.

```
python src/generate/mask.py pool  --level 25 --list ... --out src/generate/bg_pool_25.json
python src/generate/mask.py build --level 25 --plans data/synthetic/s25 --pool ... --split-masks --out data/synthetic/s25_masks
```

Two kinds of region:

| constant | value | why |
|---|---|---|
| `SYNTH_MARGIN` | **1.00** | the synthetic mask *is* the labelled disc. **No dilation.** Diffusion can only paint inside the mask, so a generated colony cannot outgrow its box. The zero-label-error claim rests on this constant, and a test asserts it. |
| `ERASE_MARGIN` | 1.35 | the erase mask is dilated generously — the shadow and halo of the real colony must go too |
| `ERASE_BLUR` | 9 | edge smoothing |

**Why an erase pass exists at all (decision 3.25): AGAR has no empty plates.** Use
a real plate as background unchanged and its own unlabelled colonies stay in the
image — the model learns "colony = background", which is injecting false negatives
into the training data by hand.

`pool` selects background plates with erase ratio ≤ 0.06 and then **checks itself
for species bias** (decision 3.27): the threshold looks at the *area* to be
erased, area is set by colony *size*, and size is set by *species*. So the
threshold silently prefers small-colony species. The check prints the pool's
species distribution against the full subset and flags every deviation. On the
demo package it flags four of five species — with two plates in the pool, it
should.

`--split-masks` writes `<name>_synthetic.png` and `<name>_erase.png` separately.
It is **mandatory** for the two-pass design (decision 3.60) — see `inpaint.py`.

### `src/generate/adapt.py`

LoRA adaptation of the inpainting UNet, one per data level.

```
python src/generate/adapt.py crops --level 100 --list ... --out src/generate/crops_100   # no GPU
python src/generate/adapt.py train --level 100 --crops src/generate/crops_100 --out src/generate/lora_100
```

Crops are extracted with `tiles.place_tiles()` on the **real** labels, so training
windows have the same geometry as inference windows. Decision 3.58: **every**
colony landing in a window is masked, not just the target — otherwise the model
learns to paint one colony while leaving its neighbours visible, which is not the
task it will be asked to do.

Decision 3.59: `--empty-ratio 0.20` adds crops whose mask covers plain medium, so
the model sees "a hole that should stay empty" during training. (This turned out
not to be enough on its own — see the erase pass in `inpaint.py`.)

Decision 3.61 — **the per-step loss is not reported and must not be read as a
training curve.** Every step samples a random timestep: a very noisy timestep
makes noise prediction easy, a nearly clean one makes it hard, and with batch = 1
that variance swamps the learning signal entirely. The first run printed
`0.0006, 0.0147, 0.0016, 0.0241` on consecutive steps. Read as a curve, that says
either "not learning" or "overfitting", and both would be wrong. Instead the file
prints a running mean and an **eval loss on fixed crops at fixed timesteps** — the
only number where the sole thing that changed is the model.

Decision 3.62: `--ckpt-every 500` writes intermediate adapters and measures the
eval loss at each, so a single run answers "how many steps?" instead of three.

Measured on an RTX 4060 Laptop: 168 crops, rank 16, 1500 steps, 1.05 s/step →
**0.44 h per LoRA**, against the 2 h that had been assumed.

### `src/generate/inpaint.py`

Mask + background → synthetic plate. Also the file that measures what generation
costs.

```
python src/generate/inpaint.py run   --level 100 --input data/synthetic/m100 --plans data/synthetic/s100 \
                                     --lora src/generate/lora_100 --out data/synthetic/g100
python src/generate/inpaint.py bench --input ... --plans ... --lora ... \
                                     --tiles 512,768 --steps-list 4,8,20 --plates 2 --out data/synthetic/bench
```

**`attach_lora()` proves the LoRA loaded** (decision 3.63). `adapt.py` saves a
peft adapter, whose keys are prefixed `base_model.model.`;
`pipe.load_lora_weights()` expects the diffusers/kohya naming, finds nothing,
prints *"No LoRA keys associated to UNet2DConditionModel found"* as a **warning**,
and generates happily from the completely unadapted base model. This happened on
the project's first real generation run. The whole 58,200-image grid could have
been produced by a model that had never seen an agar plate, and nothing in the
output would have said so — the images would just have been "a bit
disappointing", which is indistinguishable from "diffusion does not work for
this", which is the paper's headline claim. So the adapter is loaded through peft,
merged into the base weights, and a targeted weight is compared before and after.
Zero delta → `sys.exit`.

**Two passes** (decision 3.60). A single binary mask cannot distinguish "fill this"
from "leave this empty" — both are just holes, and the model fills them all, so
every filled erase region becomes an unlabelled colony. That is a representation
problem, not a prompting problem; no negative prompt fixes an input that does not
distinguish the two cases. Pass 1 erases, pass 2 synthesises, each with its own
prompt.

**Pass 1 is classical, not diffusion** (decision 3.65). Measured on the first real
plate, 35 erase regions:

| erase pass | regions that came back as a colony |
|---|---|
| diffusion + `lora_100` | 23 / 35 |
| diffusion, base model | 6 / 35 |
| `cv2.inpaint` (Telea, r = 7) | **0 / 35** |
| *(untouched real background, reference)* | 34 / 35 |

The LoRA makes the erase pass *worse*, necessarily: it was trained to turn a hole
on agar into a colony, and pass 1 hands it exactly that. Classical inpainting
cannot hallucinate a colony because it has nothing to hallucinate with — it
propagates the surrounding medium inward, which is what "plain agar" means here.
It is also free: the erase grid was 12 of that plate's 31 tiles, so the whole pass
left the GPU budget (−39 %). `--erase-method diffusion` is kept so the comparison
stays reproducible. `ERASE_WARN_PX = 120` flags a hole large enough that classical
smoothing might show.

**Cost, measured rather than assumed** (decisions 3.53, 3.64, 3.66):

| | assumed | measured |
|---|---|---|
| tiles per image | 16.3 | **19.0** (31.0 before the classical erase pass) |
| seconds per tile | 0.8 | **0.90 at 4 steps**, 3.21 at 20 |
| 58,200 images | 361 h | **232 GPU-hours** |

`bench` sweeps both levers and writes one plate per setting, because cost without
a picture is half an answer. It found that **768 is dominated**: area grows 2.25×
while tile count drops only 1.6×, so it is more expensive than 512 at every step
count — tile size is not a lever, and that could not be known without measuring.
And 4 steps is visually indistinguishable from 20 here (small masks, very strong
surrounding context), so `--steps` defaults to **4**.

**Per-colony canvas scaling** (decision 3.70) — the newest, and still under test.
Real colonies get *smoother* as they get larger: rank correlation between
diameter and texture is **−0.57**. Generated ones do not: **−0.18**. The
generator stamps texture at a fixed spatial frequency regardless of colony size,
which is exactly what a fixed latent grid does — SD's VAE downsamples by 8, so
the invented structure has a constant size in *canvas* pixels. Decision 3.51
solved this at the plate scale; this is the same problem one level down, at the
colony scale.

`canvas_for()` therefore rescales each tile so a colony occupies a constant
number of canvas pixels (`--target-diameter 128`), clamped to 256–768 because
cost is quadratic in the canvas. `--gen-scale` applies a flat factor instead, and
serves as the control: without it, a positive result could not distinguish
"scaling per colony helped" from "generating coarser helped". Both default to
off, so every number measured before this decision still compares.

The resampling cannot affect labels: `tiles.composite` gates on the
**full-resolution** mask, so no interpolation can move a colony outside its
labelled disc. Decision 3.24 holds regardless of what happens on the canvas.

### `src/generate/species_check.py`

Does a synthetic colony look like the species its label claims?

```
python src/generate/species_check.py --level 100 --generated data/synthetic/g100 \
    --list data/processed/lists/train.txt --out results/species_100.json
```

Every other gate in this repository asks whether a plate is **realistic**. None of
them asks whether it is **correct**. If the generator paints a convincing colony of
the wrong species, the plate passes zero-label-error, passes the ghost-colony
check, looks real — and per-class AP measures nothing, while the S100 arm scores
badly and the paper concludes "synthetic data does not substitute" when the true
finding is "the generator ignored the class". Those are different claims.

Three statistics on the **inner 35 % disc** of each box: `yellowness` (R − B),
`contrast` (colony grey minus the median grey of the surrounding medium), and
`texture` (Laplacian σ, `ksize=3`). The inner disc, not the whole box: the box
edge holds the colony rim and its gradient swamps everything. An earlier ad-hoc
version measured the whole mask, reported "synthetic texture 26.7 vs real 25.3",
and concluded texture was fine; on the inner disc the same data reads 5.8 vs 1.5.

Two verdicts:

1. **Ordering.** Rank the classes by each statistic in the real and generated
   data and compare the rankings (Spearman). Answers *is the class channel alive
   at all?* — robust to the generator being uniformly too pale or too grainy,
   which a per-class ratio is not.
2. **Separability.** The smallest between-class centroid distance in units of
   within-class spread, reported **with and without the texture axis**. Answers
   *could a classifier tell these apart?*

Reporting both is what makes the tool worth having. On the first run, ordering was
`+1.00` on all three statistics — the class channel works — and separability
retained **107 %** of the real value. With the texture axis removed it retained
**27 %**. The apparent success came entirely from the artefact: texture is wrong
in a class-dependent way, so it *fakes* class separation. One number would have
sent Phase 4 off on a false premise.

Guards and honesty rails: `--level` is required and the real list must belong to
that level (decisions 3.2 / 3.7); classes below `MIN_N = 15` are excluded **and
named**, never silently averaged; intervals are a **plate-level** bootstrap,
because colonies on one plate share a plate, a camera and a batch of medium.

### `src/generate/exploration/`

Six one-shot analyses that produced the numbers the layout model is built on —
the plate circle (0.465), the size and count distributions, the touching rate, the
figures, and two visual checks. They are kept because the constants above are only
defensible with the analysis that produced them. `LAYOUT_FINDINGS.md` is their
write-up.

---

## 5. Bookkeeping

### `scripts/budget.py`

Scales one measured run to the whole grid.

```
python scripts/budget.py --metrics runs/G100_s0/run_metrics.json --full-size 8000 \
    --tiles-per-image 19.0 --sec-per-tile 0.90 --lora-hours 0.44 --lora-count 4 --xai
```

Cost model: `time ≈ n_train × epochs × (imgsz / measured_imgsz)²` — linear in
images and epochs, quadratic in resolution.

The arm definitions are `(name, total training-set share, **synthetic share**)`
triples, and the third element is written out explicitly rather than derived from
the name. That is not style: on 17 August the generation cost of the three
ablation arms was not being counted at all, because their names contain no `+S` —
yet each needs a brand-new synthetic set (A1 a non-adapted model, A2 no real
backgrounds, A3 a naive layout), and the whole point of an ablation is that the
generation differs, so the images cannot be reused.

Decision 3.53: generation cost is **not** seconds per image. Inpainting is tiled,
so it is `tiles_per_image × seconds_per_tile`. The old `--gen-sec-per-image 8`
implicitly assumed ~1 tile and understated production by 2–4×; it still works and
prints a warning telling you to stop using it.

The budget gate prints a cut-down list in roadmap order when the total does not
fit — amount sweep first, main grid last, because it is the backbone of the paper
and dropping from 5 seeds to 3 weakens the standard-deviation defence.

### `scripts/collect.py`

`runs/` → `results/table.csv` + `results/raw/*.json` + `results/summary.txt`.

Decision 3.33, and it is the whole reason the file exists: `runs/` is gitignored
and should be — weights are gigabytes. But it also holds every run's
`run_metrics.json` and `eval*/summary.json`, so **all the results of the paper were
living outside version control**. Those files are a few KB and *not reproducible*:
regenerating them costs 500+ GPU-hours. If the laptop dies, or once the work moves
to the BİDB server, they exist nowhere. `results/` is committed.

The `eval*` glob deliberately catches `eval`, `eval_val` and `eval_test`. Runs
without `run_metrics.json` are listed loudly, with the likely cause (someone
called YOLO directly instead of `train.py`, so the duration and VRAM are gone).

### `scripts/setup.sh`

One command, idempotent: GPU check → conda env `agar` (Python 3.11) → PyTorch and
dependencies → GPU verification with a VRAM-derived batch suggestion → **both test
suites as a gate**. Non-zero exit if either fails, with "do not start training
before this passes".

Decision 3.47 pins `CUDA_CHANNEL=cu130`: all measurements were taken with
torch 2.13.0+cu130, and a different CUDA build makes the durations
non-comparable — the paper's cost table cannot be defended if it comes from two
machines.

The batch suggestion is measured, not guessed (decision 2.19): on an 8 GB card,
batch 8 at imgsz 1280 peaks at 6.05 GB, i.e. 80 % of usable VRAM. The previous
heuristic suggested 4 and left performance on the table.

It also checks for an **active venv before touching conda**: if one is active,
`conda activate` cannot override it, the packages land in the venv, and telling
the user "conda activate agar" at the end would point them at an empty
environment.

---

## 6. Where the leakage guards are

Three channels can leak information from a larger data level into a smaller one
(decisions 3.1–3.9). All three are closed by the same mechanism: the level is
encoded in the filename, and every consumer checks it.

| channel | carrier | enforced in |
|---|---|---|
| LoRA weights | `lora_10` … `lora_100` | `inpaint.py` (filename must carry the level), `adapt.py` |
| background plate | `bg_pool_<level>.json` | `mask.py` (pool level must equal `--level`) |
| layout statistics | `layout_<level>.json` | `layout.py` (params level must equal `--level`) |
| the comparison itself | the real list passed in | `species_check.py` |

And one more, on the measurement side rather than the generation side: the
counting confidence threshold may not be searched on the test set — `evaluate.py`.

## 7. Constants that are protocol

Changing any of these requires a `decisions.md` entry and a re-run of whatever it
touched.

| constant | value | file | decision |
|---|---|---|---|
| `CLASS_ORDER` | S.aureus, B.subtilis, P.aeruginosa, E.coli, C.albicans | everywhere | — |
| `imgsz` | 1280 | `base.yaml` | 2.8 |
| `IOU_THRS` / `REC_THRS` | 10 / 101 | `metrics.py` | — |
| `AREA_RANGES_COCO` | 32² / 96², original pixels | `metrics.py` | — |
| `max_det` | 1000 | `base.yaml` | 3.48 |
| `PLATE_R_RATIO` | 0.465 | `layout.py`, `analyze_manifest.py` | 3.10, 3.49 |
| `MAX_OVERLAP` | 0.63 | `layout.py` | 3.12 |
| `SYNTH_MARGIN` | **1.00** | `mask.py` | 3.24 |
| `ERASE_MARGIN` | 1.35 | `mask.py` | 3.25 |
| `DEFAULT_TILE` | 512 | `tiles.py` | 3.51, 3.66 |
| `--steps` | 4 | `inpaint.py` | 3.66 |
| erase method | classical | `inpaint.py` | 3.65 |
| `INNER` / `MIN_N` | 0.35 / 15 | `species_check.py` | 3.68 |

## 8. Every runtime guard

The count per file, since principle 1 is only real if it is countable:

| file | hard exits | what they protect |
|---|---|---|
| `convert.py` | 3 | source exists, JSON found, at least one image survives the filters |
| `check_labels.py` | 2 | manifest exists, the requested class exists |
| `analyze_manifest.py` | 1 | manifest exists |
| `make_splits.py` | 2 | manifest exists, every stem resolves to an image |
| `metrics.py` | 0 | *(pure core — enforcement lives at the CLI boundary)* |
| `evaluate.py` | 9 | **conf-threshold leakage**, weights-xor-preddir, prediction/image matching, duplicate stems |
| `train.py` | 6 | list exists / non-empty / points at `/images/` / labels resolve, **no `null` augmentation**, resume target exists |
| `layout.py` | ≥2 | `--level` required, params level must match |
| `mask.py` | ≥2 | pool level must match, plan level must match |
| `inpaint.py` | ≥4 | LoRA filename carries the level, **LoRA actually merged**, adapter directory is real, inputs readable |
| `adapt.py` | ≥2 | level check in pre-flight (before `import torch`, so it fires on a GPU-less box) |
| `species_check.py` | 5 | level/list agreement, class ids in range, both sides non-empty, some class comparable |
| `budget.py` | 2 | a measurement source, known arm names |
| `collect.py` | 2 | `runs/` exists, something was found |
| `setup.sh` | 5 + test gate | conda present, env created, deps installed, **both suites pass** |

Two of these were moved to pre-flight after being written in the wrong place:
`train.py`'s null-augmentation check sat after the `--dry-run` early return, and
`adapt.py`'s level check sat after `import torch`. A check that cannot run on the
cheap path is a check that finds the problem late.

## 9. Known gaps

Honest list, kept here so it is visible from the code rather than only from
`decisions.md`.

- **Species separability** retains 27 % of the real value on the demo subset
  (colour + contrast axes). This is the open problem of Phase 4.
- **Texture is at the wrong scale** (decision 3.69). Not a sampler artefact
  (4 steps 5.8, 20 steps 5.9, real 1.5), not a guidance artefact (3–6× at every
  value tested), and not the LoRA (20.2 without it, 18.9 with it, real 4.6). It
  is the fixed latent grid. `--target-diameter` (decision 3.70) is the first
  attempt at a fix and is not yet validated. Risk while it stands: the detector
  learns "grainy = colony" and the S100 arm measures the cue rather than the
  substitution. A control run for that is not designed yet.
- **Everything is fitted on 10 demo plates.** Gamma, the background pool, the
  layout quantiles and the LoRA all carry `provisional` in spirit if not in the
  file. Full AGAR access is pending.
- **`lora_10` sufficiency** is unmeasured. If the model cannot learn a usable
  colony appearance from ~800 images, the G10+S arm says "the generator was
  under-adapted", not "synthetic data does not help".
- **Clark–Evans** is 1.13 generated versus 1.03 real. A single-parameter Strauss
  process may not capture how *heterogeneous* real spacing is; a two-range model
  is the fallback.
- **`base.yaml`'s `eval.iou_thrs` and `eval.area_ranges` are documentation only** —
  the values that run are the constants in `metrics.py`. They agree today. Nothing
  checks that they still will.
