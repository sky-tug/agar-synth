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
([decision 3.24](decisions.md)). Diffusion can only paint inside the mask, so a
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

5 seeds each. Plus controls: **Baseline B** (classic geometric/colour
augmentation) and **Baseline C** (copy-paste / mosaic / mixup) answer *"could you
have done this more cheaply?"*; ablations **A1/A2/A3** (no LoRA, no real
background, naive layout) answer *"which part actually mattered?"*. 61 training
runs total.

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
src/eval/test_metrics.py     36 sanity checks (cross-validated vs pycocotools)

src/generate/layout.py       where/what/how-big  -> synthetic coordinates
src/generate/mask.py         coordinates -> inpainting mask + background
src/generate/test_generate.py 36 sanity checks
src/generate/exploration/    one-shot analyses behind the layout model

scripts/train.py             training wrapper (logs time, VRAM, git commit)
scripts/evaluate → src/eval  see above
scripts/budget.py            GPU-hour estimate for the whole grid
scripts/collect.py           runs/ -> results/table.csv
scripts/setup.sh             one-command environment setup + test gate
```

`ARCHITECTURE.md` explains what each file does and, more importantly, *why each
line is written the way it is*. `decisions.md` is the decision log — **in Turkish**,
because the paper and the supervision are in Turkish.

---

## Quick start

```bash
git clone https://github.com/<user>/agar-synth
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

Generate synthetic layouts and masks (diffusion itself is not wired up yet):

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
    --pool src/generate/bg_pool_25.json --out data/synthetic/s25_masks
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

Data pipeline and measurement infrastructure are complete and tested.
The layout and mask stages of the generation pipeline are complete and tested.
LoRA adaptation and inpainting are pending full-dataset access and GPU allocation.

Estimated cost of the full grid: **512–761 GPU-hours** (61 training runs +
~58,000 synthetic images + 4 LoRA adaptations). See `scripts/budget.py`.

## Licence and citation

Code: MIT. The AGAR dataset is CC BY-NC 2.0 and is **not** redistributed here.
Ultralytics YOLO is AGPL-3.0 — see their terms if you reuse the training path.

If this is useful to you, please cite the AGAR dataset paper
(Majchrowska et al., 2021, [arXiv:2108.01234](https://arxiv.org/abs/2108.01234))
alongside this repository.
