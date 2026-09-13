# agar-synth

**How many real annotated images is a synthetic image worth?**

A quantitative study of whether diffusion-generated, automatically-labelled
microbial colony images can substitute for real annotated data in object
detection — measured on the [AGAR dataset](https://agar.neurosys.com/).

> Undergraduate research project, Department of Computer Engineering,
> Gazi University. Advisor: Prof. Dr. Muhammet Ali Akçayol.
> Target venue: *Mühendislik Bilimleri ve Tasarım Dergisi* (JESD).
> **The paper is written in Turkish; the code and this README are in English.**

---

## The problem

Counting bacterial colonies on agar plates is done by eye. Automating it with an
object detector needs thousands of labelled images — and "labelled" means a human
drew a box around every single colony. AGAR contains **336,442** hand-drawn boxes.
Every new species, medium or camera setup restarts that cost.

The obvious fix is to generate images with a diffusion model. It does not work:
**you do not know where the colonies are in the image you just generated.** You
still have to annotate by hand. The cost does not disappear, it moves.

## The approach: mask-controlled inpainting

Reverse the order. Decide the layout *first* — "a 40 px *S. aureus* colony at
(942, 337), another at …" — then let the diffusion model fill in **only those
disks**, leaving everything else untouched.

The coordinates were yours to begin with, so **the label is free and exact**.

The synthetic mask is exactly the labelled disk, with no margin
([decision 3.24](DECISIONS.md)). Diffusion can only paint inside the mask, so a
generated colony cannot outgrow its box. That is what "zero label error" rests on,
and `src/generate/test_generate.py` asserts the constant that guarantees it.

## The actual research question

The technique above is a means. The measurement is the paper:

> Train the *same* detector on different data mixtures, test them all on the
> *same* real held-out set. If `25% real + synthetic` scores like `50% real`, the
> synthetic data replaced 25% of the annotation effort. That number is the result.

| arm | training data |
|---|---|
| `G100` `G50` `G25` `G10` | real only, at 100 / 50 / 25 / 10 % |
| `G50+S` `G25+S` `G10+S` | that fraction of real, topped up with synthetic |
| `S100` | synthetic only |

5 seeds each. Plus controls that answer *"could you have done this more
cheaply?"* — **Baseline B** (translate/scale on top of the flips and HSV jitter
every arm already uses) and **Baseline M** (mosaic + mixup). M was going to be a
copy-paste arm until Ultralytics' `copy_paste` turned out to be a no-op on
detect-format labels, so the arm is named for what it applies and copy-paste
will be measured separately ([decision 3.98](DECISIONS.md)). Ablations
**A1/A2/A3** (no LoRA, no real background, naive layout) answer *"which part
actually mattered?"*. 61 training runs total.

---

## Repository layout

```
data/processed/          convert.py output (gitignored)
splits/                  train/val/test stem lists  <- committed, reproducibility
configs/                 the protocol: every hyperparameter, with its rationale
runs/                    training outputs (gitignored)
results/                 collected metrics  <- committed, see scripts/collect.py
```

```
src/convert.py               AGAR JSON -> YOLO labels + manifest.csv
src/check_labels.py          draw boxes on images for visual verification
src/analyze_manifest.py      descriptive stats + quality control
src/make_splits.py           stratified, nested train/val/test splits

src/eval/metrics.py          COCO-identical AP, size breakdown, counting metrics
src/eval/evaluate.py         evaluation CLI
src/eval/test_metrics.py     50 sanity checks (cross-validated vs pycocotools)

src/generate/layout.py       where/what/how-big  -> synthetic coordinates
src/generate/mask.py         coordinates -> inpainting mask + background
src/generate/tiles.py        native-resolution tiling; no colony may be split
src/generate/adapt.py        LoRA adaptation of the inpainting model
src/generate/inpaint.py      mask + background -> synthetic plate (run / bench)
src/generate/species_check.py does a synthetic colony look like its label says?
src/generate/test_generate.py 65 sanity checks
src/generate/exploration/    one-shot analyses behind the layout model

scripts/train.py             training wrapper (logs time, VRAM, git commit)
scripts/evaluate → src/eval  see above
scripts/budget.py            GPU-hour estimate for the whole grid
scripts/collect.py           runs/ -> results/table.csv
scripts/setup.sh             one-command environment setup + test gate
```

`CODE_MAP.md` is the file-by-file map: what each script does, in the order the
data moves through them. `ARCHITECTURE.md` explains *why each line is written
the way it is*. `DECISIONS.md` is the decision log: every methodological choice
with its reason, and — where it happened — the measurement that later overturned
it. `decisions.tr.md` is the same log unabridged, in the working language of the
project.

---

## Quick start

```bash
git clone https://github.com/sky-tug/agar-synth
cd agar-synth
bash scripts/setup.sh          # env + deps + both test suites must pass
```

The full AGAR dataset requires registration at
[agar.neurosys.com](https://agar.neurosys.com/); a free representative sample is
downloadable without one. Point `--src` at whichever you have:

```bash
python src/convert.py      --src data/AGAR_representative --out data/processed
python src/make_splits.py  --data data/processed --seed 42
python src/check_labels.py --data data/processed --n 30      # then LOOK at them
python src/analyze_manifest.py --data data/processed
```

Train, evaluate, budget:

```bash
python scripts/train.py --config configs/base.yaml --level 100 --seed 0 --name G100_s0

# VAL first: it selects the counting confidence threshold
python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \
    --split val --out runs/G100_s0/eval_val --tag G100_s0
# then TEST, with that threshold. Searching it on test is leakage; the code refuses.
python src/eval/evaluate.py --weights runs/G100_s0/weights/best.pt \
    --split test --conf-thr <value> --out runs/G100_s0/eval_test --tag G100_s0

python scripts/collect.py       # runs/ -> results/table.csv  (commit this)
python scripts/budget.py --metrics runs/G100_s0/run_metrics.json --full-size 8000 \
    --gen-sec-per-image 8 --lora-hours 2 --lora-count 4 --xai
```

Generate synthetic plates. The three stages are chained per level by
`scripts/gen_queue.sh`; run by hand like this when you want one level only:

```bash
python src/generate/layout.py fit    --level 25 --list data/processed/lists/train_25.txt \
    --out src/generate/layout_25.json
python src/generate/layout.py sample --level 25 --params src/generate/layout_25.json \
    --count 400 --seed 0 --out data/synthetic/s25
python src/generate/layout.py validate --params src/generate/layout_25.json \
    --list data/processed/lists/train_25.txt --generated data/synthetic/s25

python src/generate/mask.py pool  --level 25 --list data/processed/lists/train_25.txt \
    --out src/generate/bg_pool_25.json
python src/generate/mask.py build --level 25 --plans data/synthetic/s25 \
    --pool src/generate/bg_pool_25.json --split-masks --out data/synthetic/s25_masks

# --split-masks is not optional: inpaint.py refuses a single combined mask,
# because it cannot express "fill this disk, empty that one" (decision 3.85).
python src/generate/inpaint.py run --level 25 --input data/synthetic/s25_masks \
    --plans data/synthetic/s25 --lora src/generate/lora_25 \
    --tile 512 --steps 4 --resume --out src/generate/synth_g25
```

Then build the arm's training list and train on it:

```bash
python scripts/make_arm_lists.py      # real subset + synthetic, with a sidecar
python scripts/train.py --config configs/base.yaml --level 25 --seed 0 \
    --train-list data/processed/lists/arm_G25plusS.txt --name G25plusS_s0
```

Or leave it to the queues and just watch:

```bash
python scripts/status.py    # card, current run, generation, arms, progress
```

---

## Design principles

Three ideas run through the whole repository. If you contribute, follow them.

**1 — A methodological rule is a runtime check, not a comment.**
A rule written in prose gets forgotten. Here they raise `SystemExit`:

| rule | enforced in |
|---|---|
| the counting threshold may not be searched on the test set | `evaluate.py` |
| a real run cannot proceed with an empty validation set | `train.py` |
| synthetic layout cannot be generated without an explicit data level | `layout.py` |
| the background pool level must match the plan level | `mask.py` |
| splits must be verified nested and leak-free before use | `make_splits.py` |
| a LoRA that did not actually load stops generation instead of warning | `inpaint.py` |
| class fidelity may not be compared against a different data level | `species_check.py` |
| `null` in the augmentation config is rejected — it would silently become an Ultralytics default | `train.py` |

**2 — A silent failure is worse than a loud one.**
Every serious wound this project has taken was silent: split lists pointing at the
wrong directory (Ultralytics *warned and continued*, so the model learned "there
are no objects here"); a validation check that inspected a variable instead of the
file actually written (it passed, training blew up); `max_det` set in the config
but never passed through; a colony silently dropped from a generated plate,
skewing the count distribution. So the code counts and prints everything: how many
boxes were clipped, how many images were dropped and why, how many colonies could
not be placed, whether the background pool is species-biased. When you add code,
ask: *can this quietly produce a wrong answer? Then add a counter.*

**3 — Measurement replaces guesswork.**
`imgsz=1280` was a guess, then a memorisation test measured it. The colony
touching rate was a guess, then 387 colonies measured it. The Strauss interaction
parameter could have been hand-tuned; it is calibrated by bisection instead. And
where the measurement is weak, that is reported too: the touching rate ships with
a plate-level bootstrap confidence interval and a `provisional: true` flag,
because ten demo plates pick a distribution *family*, not its parameters.

---

## Status

*Last updated: 13 September 2026.*

Phases 1–5 are closed. The full dataset (4,267 images, 83,208 boxes) is in use,
the protocol is frozen and tagged, and **the real-data axis of Phase 6 is
complete**: four data levels, five seeds each, twenty runs.

```
Phase 1  data preparation           done   4,267 images · 83,208 boxes
Phase 2  real-data baseline         done   mAP50-95 = 0.699
Phase 3  synthetic pipeline         done   4 levels: layout · bg pool · crops · LoRA
Phase 4  texture gate               FAILED measured, and the gate's own blind spot measured
Phase 5  freeze the protocol        done   tag phase5-done
Phase 6  52 training runs         running  23 done · 54.3 GPU-hours · real-data axis + B arm
Phase 7  ablations (9 runs)          ---
```

### Measured so far

How much real annotated data is actually worth, five seeds per arm, on val:

| arm | images | seeds | mAP50-95 | sigma | own threshold | mAP_small | GPU-h |
|---|---|---|---|---|---|---|---|
| G100 (100% real) | 2,987 | 5 | 0.69901 | 0.00356 | 0.00711 | 0.5742 | 21.74 |
| G50 (50% real) | 1,491 | 5 | 0.68091 | 0.00284 | 0.00569 | 0.5440 | 13.79 |
| G25 (25% real) | 746 | 5 | 0.66437 | 0.00381 | 0.00762 | 0.5076 | 7.16 |
| G10 (10% real) | 299 | 5 | 0.63705 | 0.00955 | 0.01910 | 0.4861 | 5.02 |

All three pairwise gaps clear both arms' thresholds (t = 8.9 / 7.8 / 5.9,
df = 8). The curve is log-linear in the number of real images:

```
mAP50-95 = 0.48573 + 0.01853 * log2(N)          R2 = 0.99768
```

Every halving of the real training set costs **0.0185 mAP50-95**. Read the size
of that: going from 2,987 annotated images down to 299 — a tenfold reduction in
labelling effort — costs 0.062, about 9% relative. The gap synthetic data has
to close is small, which is the demanding case for this project rather than the
flattering one.

**A prediction was written down before the last arm was run.** With three
points in hand (G100, G50, G25) the curve was fitted and G10 was predicted on
11 September, before any G10 run had started:

```
                      predicted    measured     difference   threshold
mAP50-95                0.64128     0.63705       -0.00423     0.01910  held
mAP_small               0.46480     0.48607       +0.02127     0.01344  missed
```

The primary metric landed inside its own measurement uncertainty — a curve
fitted to three points predicted a data level it had never seen. Fitting four
points after the fact and reporting R² = 0.998 would prove nothing; four points
always lie near a line. Saying the fourth one in advance is what tests the
curve, because the curve is meant to be used as a predictor.

The small-object metric missed, and on the high side: its curve **flattens** at
the low end instead of continuing to fall. That is reported as a finding, with
no mechanism claimed for it.

Slopes by object size, per halving of the real data:

| metric | loss per halving | R² | pairwise gaps resolved |
|---|---|---|---|
| `mAP_small` | 0.02714 | 0.97393 | 1 of 3 |
| `mAP50-95` (primary) | 0.01853 | 0.99768 | 3 of 3 |
| `mAP_medium` | 0.01715 | 0.99807 | 3 of 3 |
| `mAP_large` | 0.01702 | 0.99251 | 3 of 3 |

Small objects do suffer most from data scarcity — about 60% more than large
ones — and the primary metric's slope comes largely from them. The claim stands
on the fit, though: only one of `mAP_small`'s three single-step gaps is
separable from seed noise.

The size rows above use the `minGT` reading, not plain COCO. On this dataset
*S. aureus* has exactly **one** large ground-truth box in val, and COCO-style
averaging let that single box drive `mAP_large` — its seed spread was 32× the
primary metric's and none of its three gaps were separable. With cells below
ten ground-truth boxes excluded, the spread falls to 1.1–1.7× and all three
gaps separate. Both columns are reported side by side, and `metrics.py` itself
stays COCO-identical.

**The significance threshold is itself a measured quantity, and it was the first
output of the grid rather than the last.** `eval.significant_diff_threshold`
deliberately stayed `null` until five seeds of the same arm had been run, and
every arm carries its own:

```
G100   2 * sigma = 0.00711      sigma 90% CI  0.0023 - 0.0084
G10    2 * sigma = 0.01910      sigma 90% CI  0.0062 - 0.0227
```

Three seeds had put the G100 threshold at 0.00399 — an under-estimate by nearly
half. The three-seed value is kept next to the five-seed one in `DECISIONS.md`
rather than deleted, because how far a small-sample sigma can stray is itself a
finding.

At 299 training images the seed spread is 2.5–3.4× every other arm, so the
synthetic arms at that level will be judged against 0.01910, not the primary
arm's 0.00711. Using one shared threshold would have meant claiming 170% more
sensitivity than the data supports.

### What cheap augmentation is worth

Before spending a single synthetic run, the obvious objection was measured:
a classical augmentation pipeline is nearly free, so does it buy what diffusion
is supposed to buy? Three seeds at the G25 level, translation and scale on top
of the flips and HSV jitter the main arms already use:

| arm | images | seeds | mAP50-95 | sigma | mAP_small |
|---|---|---|---|---|---|
| G25 (25% real) | 746 | 5 | 0.66437 | 0.00381 | 0.5076 |
| B_G25 (+ translate/scale) | 746 | 3 | 0.68147 | 0.00044 | 0.5431 |

The gap is **+0.01710**, above both arms' thresholds (t = 8.64, df = 6). Put it
on the substitution curve and it reads:

```
746 real images + translate/scale  ~=  1,513 real images
```

Augmentation is worth **767 images' worth of annotation** — about the same as
one doubling of the real training set. For small objects it is stronger still
(+0.0354, 2.1x the pooled threshold), and notably this is the *first*
small-object difference in the project that separates from seed noise at all.

So the bar for synthetic data is now a measured number rather than an
assumption, and it is a demanding one. That is the honest way round: the
competitor was measured before the method it competes with.

One caveat, on the record: B's seed spread (0.00044) is eight times smaller
than G25's, and that is an artefact of the epoch ceiling rather than a property
of the arm — two of the three runs used the full 150 epochs and the third 142,
while the G25 runs stopped on patience at 89–109. Runs that all stop in the
same place lose the early-stopping variation that dominates the other arms. So
0.68147 is a lower bound, the arm's own 2σ is not used as a threshold, and the
judgement above rests on G25's threshold and the t test instead.

### Findings that cost a claim

Two statements were retracted after being measured properly:

- **"The class signal is real."** Three-axis separability between synthetic
  species is 92%, but with the texture axis removed it collapses to 15%, and
  the two intervals do not overlap. The classes separate along an axis the
  generator gets *wrong*. This is reported as a finding, not hidden.
- **"Small objects are found but poorly localised — mAP50 gap small, mAP75 gap
  a chasm."** Written from a single run. Across five seeds the large-vs-small
  gap is 11.9 sigma in mAP50 but only 1.6 sigma in mAP75: the direction holds
  (5/5 seeds), the magnitude cannot be claimed. The cause turned out to be
  concrete — *S. aureus* has exactly **one** large box in val, and COCO-style
  averaging let that single box carry a quarter of `mAP_large`.

### Cost, measured not estimated

| | early estimate | measured |
|---|---|---|
| tiles per image | 19.0 | **10.06** |
| seconds per tile | ~0.34 | **0.959** |
| peak VRAM (generation) | — | **2.889 GB** |
| training run (G100) | 3.5 GPU-h | **4.35 GPU-h** |
| training run (G50) | — | **2.76 GPU-h** |
| training run (G25) | — | **1.43 GPU-h** |
| training run (G10) | — | **1.00 GPU-h** |
| whole real-data axis (20 runs) | — | **47.72 GPU-h** |

The generation estimate was not 52% cheaper than assumed, as an earlier note
claimed; it is **49% more expensive**. See `scripts/budget.py`.

Training cost does not fall as fast as the data does: the small arms need
*more* epochs to converge (G10 averages 115 against G25's 97), so a tenth of
the data costs about a quarter of the GPU time rather than a tenth of it. The
estimate for G25 + G10 together was "roughly 25 GPU-hours"; they came in at
12.18.

### Open

Next, and not blocked on anything: the six classical-augmentation control runs
(`B_G25`, `C_G25`). They answer the obvious objection to this whole project —
that a strong augmentation pipeline would buy the same thing as synthetic data
for none of the cost.

Production-LoRA selection is deliberately unresolved and blocks the 26
synthetic-arm runs: longer LoRA training buys class separation and pays for it
in distribution fidelity, and the choice is tied to a measurement rather than
an opinion. See decision 3.89 in `DECISIONS.md`.

## Licence and citation

**Code: AGPL-3.0** (see `LICENSE`). The training and evaluation paths import
Ultralytics YOLO, which is AGPL-3.0; this repository is therefore released
under the same licence.

The AGAR dataset is CC BY-NC 2.0 and is **not** redistributed here — no images,
crops, or derived tiles are in this repository or its history.

If this is useful to you, please cite the AGAR dataset paper
(Majchrowska et al., 2021, [arXiv:2108.01234](https://arxiv.org/abs/2108.01234))
alongside this repository.
