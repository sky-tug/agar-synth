# Decision log

Every methodological choice in this project, with the reason it was made — and,
where it happened, the measurement that later overturned it.

This is a condensed English rendering of `decisions.tr.md`, which is the full
record kept in the working language (the paper, the supervision and the target
journal are all Turkish). Nothing here is a claim the Turkish file does not
make; where the two differ in detail, the Turkish file is authoritative because
it carries the full argument rather than its summary.

Numbering is continuous and never reused. A decision that was later retracted
keeps its number and its retraction is recorded next to it, because *how wrong
an earlier estimate was* is itself a result.

---

## Phase 1 — Data pipeline (11 August 2026)

| # | Decision | Reason |
|---|---|---|
| 1.1 | Class order is fixed: `S.aureus`=0, `B.subtilis`=1, `P.aeruginosa`=2, `E.coli`=3, `C.albicans`=4 | Never changes again. If it did, every label file would break silently. |
| 1.2 | Images containing `defects` / `contamination` are dropped **entirely** | Deleting the box but keeping the image teaches the model "there is no object here". The count of dropped images is reported in the paper. |
| 1.3 | Image size is read **from the file**, not from the JSON | AGAR's JSON has no size field. Assuming 2048×2048 produces a silent normalisation error. |
| 1.4 | Field is `classes` (the proposal said `microbes`); class name is `S.aureus`, no space | Verified against the demo package. |
| 1.5 | Split is **stratified** by species combination, at image level | If *C. albicans* vanished from the 10% subset, level-to-level differences would measure class imbalance rather than data volume. |
| 1.6 | Subsamples are **nested**: 10% ⊂ 25% ⊂ 50% ⊂ 100% | Otherwise a difference cannot be attributed to data volume rather than to *which* images were picked. |
| 1.7 | `splits/` stem lists go into git; absolute-path lists do not | Paths are machine-dependent, stems are not. |

## Phase 2 — Measurement infrastructure (12–13 August 2026)

| # | Decision | Reason |
|---|---|---|
| 2.0 | `make_splits.py` fixed: Ultralytics lists now point under `data/processed/images/`, not at the raw AGAR folder | Ultralytics finds labels by replacing the last `/images/` in the path with `/labels/`. With the raw path, **no label was found at all** — and Ultralytics warns and continues rather than stopping, so the model learned "these images contain no objects". |
| 2.1 | mAP / AP computed **independently of Ultralytics**, in `src/eval/metrics.py` | Ultralytics gives no size-stratified AP, and the paper's central claim is about small colonies. A second-detector control also needs the same measurement code to run on both. |
| 2.2 | AP is **algorithmically identical to COCO**: IoU 0.50:0.05:0.95, 101-point interpolation, out-of-range GTs ignored | A reviewer will ask "did you write your own metric?". The answer is yes, and it is cross-checked against pycocotools to 1e-4. |
| 2.3 | Size breakdown uses **COCO ranges** (32²/96²) on the original 2048 px image | Comparability with the AGAR paper. |
| 2.4 | The counting confidence threshold is **selected on val** and applied to test. `evaluate.py --split test` refuses to run without it | Searching the threshold on test means tuning on test. That is leakage, and the code enforces the rule rather than trusting memory. |
| 2.5 | sMAPE defined as `200·\|p−g\| / (\|p\|+\|g\|)`, zero when `p=g=0` | MAPE diverges on empty plates; sMAPE does not. |
| 2.6 | Signed mean error (`ME`) is reported alongside MAE | MAE hides bias. "The model systematically undercounts" is more useful to a microbiologist than "MAE 4.2". |
| 2.7 | Primary metric is `mAP50-95`. The "there is a difference" threshold is to be **measured**, not chosen | If the threshold were picked after seeing results, that is cherry-picking. It must exceed the seed-to-seed standard deviation. *(Measured in 3.92, revised in 3.93.)* |
| 2.8 | **`imgsz: 1280`** — if VRAM runs out, reduce batch, never imgsz | The single most critical hyperparameter. At 2048 px a median *C. albicans* colony is 27.5 px; at `imgsz=640` that becomes 8.6 px, at the detector head's stride. At 1280 it is 17.2 px. Cost scales with imgsz². |
| 2.9 | Augmentation in the main grid is **minimal**: `fliplr`, `flipud`, `hsv_*` only | Ultralytics defaults to `mosaic=1.0`. Left alone, the real-only control arms would already be using classical augmentation and the "diffusion vs classical" comparison would be muddied from the start. |
| 2.10 | `flipud` **on** (0.5) | An agar plate has no canonical orientation, so vertical flip is a legitimate transform here — unlike in natural images. |
| 2.11 | `scale` / `translate` / `degrees` **off** in the main grid | Colony size is a **class cue** (*S. aureus* small, *E. coli* large). Randomising scale destroys the class signal. Baseline B breaks this cue deliberately. |
| 2.12 | `copy_paste` off in the main grid, on in Baseline C | Otherwise "diffusion vs copy-paste" becomes undefined. |
| 2.13 | Every augmentation field is written **explicitly** in the config, including the zeros | Ultralytics defaults drift between releases. An implicit default is irreproducibility. |
| 2.14 | Training goes through `scripts/train.py`, never `yolo train` directly | Time, GPU-hours and peak VRAM must be recorded from the first run. They cannot be measured retroactively. |
| 2.15 | `max_det: 1000` | AGAR plates hold up to 125 colonies. The default 300 looks sufficient but clipping on crowded plates would corrupt the counting metric directly. |
| 2.16 | `make_splits.py` no longer calls `.resolve()`, **and the check was fixed too** | Files under `data/processed/images/` are symlinks into the raw AGAR tree; `resolve()` followed them, the `/images/` component disappeared, and labels were not found. Worse, the verification tested the value in a variable rather than the line written to disk — the check passed and training blew up. |
| 2.17 | `max_det` is now actually passed to `model.train()` | It was set in the config and silently ignored; Ultralytics used 300. Same class of bug: `cutmix` is now written explicitly as 0.0. |

## Phase 3 — Generation pipeline (14 August 2026)

The leakage analysis that shapes the whole synthetic half.

| # | Decision | Reason |
|---|---|---|
| 3.1 | **Leakage has three channels, not one**: (a) LoRA weights, (b) the real background plate inpainted onto, (c) the label statistics the layout and size distributions were fitted to | Only (a) had been considered. (b) and (c) are quieter but have the same effect: they void the claim that an arm used "only 10% of the real data". |
| 3.2 | **One LoRA per level** — four adaptations: `lora_10`, `lora_25`, `lora_50`, `lora_100`, each trained only on its own level's train subset | Since subsets are nested (1.6), `lora_10`'s data ⊂ `lora_25`'s. That is the point: every point on the substitution curve is generated within its own information budget. |
| 3.3 | LoRA trains on the **train split only**; val and test are never shown to the diffusion model at any level | Indirect contact through a generator is still contact. |
| 3.4 | **Background pool is per level** | Plate texture, lighting and edge shadow are all information belonging to that image. A shared pool would falsify the "only 25% real" sentence. |
| 3.5 | **Layout and size distributions are fitted per level.** `layout.py` refuses to run without an explicit `--level` | The subtlest channel. "Colonies are distributed like this, median size is that" extracted from 8,000 images would bury prior knowledge into a level built from 800. |
| 3.6 | **Wording for S100 is fixed.** The paper will not say "no real data was used". It will say the detector saw only synthetic images, and real data entered **indirectly** through LoRA adaptation, the background pool and layout statistics | The honesty anchor of the paper, written down early so it cannot be quietly softened later. |
| 3.7 | **Naming makes leakage structurally impossible**: every synthetic artefact carries its level in its name, and two paths with different levels in one command make the script **stop** | A check that leans on the filesystem rather than on human attention. Picking the wrong LoRA while tired three weeks later is the likeliest failure mode. |
| 3.8 | Ablation A1 ("no LoRA") is defined as generation at level 25 with the unadapted base model; background and layout still come from `train_25` | Only one variable may change, or A1 and A2 become indistinguishable. |
| 3.9 | Budget consequence: LoRA training is **four** line items, not one | The earlier total did not include this. |
| 3.10 | **Plate circle is fixed**: centre = image centre, `R = 0.465 · width`; never estimated at generation time | Found by Hough on 10/10 plates, R = 950 ± 6 px. One outlier had caught an inner ring — an estimation error must not leak into the pipeline. |
| 3.11 | Layout: θ ~ U(0,2π), radius from the **inverse CDF of the observed distribution**, not a parametric family | Angle is uniform (KS p=0.44). Radius is **not** (KS D=0.123, p=1.6e-5): the inner 89% disc is uniform, the outer ring takes 8.5% instead of 20%. |
| 3.12 | **Overlap is allowed**, clipped at the observed maximum depth 0.63; calibration target is a 35–45% touching rate | In real plates **39%** of colonies touch a neighbour. A naive "no overlap" rule would make synthetic plates systematically tidier than real ones. |
| 3.13 | Mask is circular, label box is square, with ~3% elliptical jitter | 96.9% of AGAR boxes are exactly square (375/387). |
| 3.14 | Size is sampled **per class**, not from a shared pool | Two regimes with nothing between them: *C. albicans* 26.5 px and *S. aureus* 29 px versus *E. coli* 128, *B. subtilis* 151, *P. aeruginosa* 155. Same reason as 2.11 — size is a class cue. |
| 3.15 | Radial distribution is **not** conditioned on class, for now | A weak tendency exists but cannot be established on ten plates. |
| 3.16 | Plate composition: one dominant species plus a small number of secondary ones, sampled from the observed combination distribution | Four of ten demo plates are mixed. A single-species generator would not match reality. |
| 3.17 | ~~Colony count is lognormal~~ | **Superseded by 3.84.** The family was chosen on ten plates; parameters cannot be estimated from ten plates, and on the full data the family itself does not hold. |
| 3.18 | A3 redefined as "naive placement": plate circle ignored, size drawn from one pool regardless of class, no touching constraint | **Finding: real layout is already random.** Clark–Evans ratio is 0.90–1.10 on 9 of 10 plates, so an ablation against "random placement" would have measured nothing. The ablation had to be redefined around the constraints that *do* matter. |
| 3.19 | Short-range repulsion via a **Strauss interaction parameter**, auto-calibrated by bisection against the measured touching rate | Not hand-tuned. Pure random placement produces too many touching pairs; the parameter is what makes the synthetic touching rate match the real one. |
| 3.20 | **Common random numbers during calibration**: recipes are generated once and reused for every candidate γ | In the first version γ rejection shifted the random stream, so the same γ produced different recipes and the bisection was fitting noise. |
| 3.21 | Count location uses the **median** of `ln(n)`, not the mean | One crowded plate (n=125) dominated the mean; generated medians were 15% below real. With the median the gap fell to 6%. |
| 3.22 | **A colony is never silently dropped.** When attempts run out, a γ-rejected valid position is used, then the least-violating one; both cases are counted and reported | In the first version colonies were silently lost after γ rejection, which shifted the entire count distribution. |
| 3.23 | `validate` subcommand compares generated layout against real data on ten measures | "It looks fine" is not an acceptance criterion. |
| 3.24 | **Synthetic mask = exactly the labelled disc. No margin** (`SYNTH_MARGIN = 1.00`) | Diffusion can only paint inside the mask. Enlarge it and a generated colony could exceed its label box, turning "zero label error" into "approximately zero" — the paper's central claim, not negotiable. |
| 3.25 | **Real colonies on the background go into an erase mask** (`ERASE_MARGIN = 1.35`); the full mask is synthetic discs ∪ erase regions | **AGAR has no empty plates** — the cleanest demo plate still holds 32 colonies. Using a background as-is leaves unlabelled real colonies in the image, which is injecting false negatives into the training data. |
| 3.26 | Background pool is selected by an `erase_ratio` threshold (default 0.06); fewer than 20 plates raises a red warning | The more area must be erased, the less "real background" the arm actually uses, and the weaker the A2 ablation becomes. |
| 3.27 | The `pool` command **checks for class bias** and warns above 15% | Found on demo data: at threshold 0.06 the pool was 100% small-colony species. The chain is erase area → colony size → class (3.14). A threshold on area is a threshold on class. |
| 3.28 | Every generated plate carries `provenance/<name>.json`: which background, which level, which list, how many real colonies erased | The leakage audit trail. If a reviewer asks where G25's backgrounds came from, the answer is in a file, not in memory. |

## Phase 3 — Diffusion pipeline (18 August 2026)

| # | Decision | Reason |
|---|---|---|
| 3.51 | **Inpainting is done at native resolution, in tiles.** Downscaling the plate to 512 is forbidden | A median *C. albicans* colony would go from 27.5 px to 6.8 px. The generator would be asked to synthesise a 7 px object and the upscaler would invent its texture; the small-object arms would then measure an artefact. |
| 3.52 | **A colony may never be split across two tiles.** `place_tiles()` raises on any layout that would; `test_generate.py` verifies it on generated plates | The two halves would be generated in separate passes from different noise. They would not match, and a seam would run through the middle of a colony. |
| 3.53 | **Cost model changed: not seconds per image but tiles per image × seconds per tile** | The old model assumed ~1 tile per image. Measured: 16.3 tiles per plate at 512. |
| 3.54 | Negative prompt for erase regions; the composite writes only where the mask allows | Erased areas must come back as **plain medium**. If the model invents a colony there, the image contains an unlabelled object — the exact failure 3.25 exists to prevent. |
| 3.55 | `inpaint.py` writes `gen_metrics.json` on every run, and a `bench` subcommand sweeps tile size × steps | Production cost depends on two levers and neither can be guessed. At ~58,000 images the difference between samplers is hundreds of GPU-hours. |
| 3.56 | `--dry` mode: a deterministic fake painter replaces the diffusion call, everything else runs for real | So the pipeline can be verified without a GPU — and so a plumbing error is never mistaken for a generation-quality problem. |
| 3.57 | `adapt.py crops` extracts and **reports** the training set before touching the GPU, with a red warning if a species is missing | A species absent from the adaptation set **cannot be generated**. At low levels this is a real risk and would silently turn a data-volume experiment into a class-imbalance one. |
| 3.58 | Crops mask **all colonies falling in the crop**, not only the assigned one | The first version left neighbouring colonies visible in the crop. A visual check caught it immediately. |
| 3.59 | 20% of the adaptation set is **empty holes**: a plain-medium patch, target identical, prompt "no colonies" | Otherwise the model learns that a hole is *always* a colony. |
| 3.60 | **Generation is two passes: erase first (plain medium), then synthesise.** Split masks became mandatory | With a single binary mask the model cannot know whether a hole should become a colony or stay empty — the ambiguity is at the level of the representation, not the prompt. |
| 3.61 | Per-step loss is **not reported or interpreted**; a running mean plus a validation loss on **fixed crops at fixed timesteps** is used instead | Diffusion training samples a random timestep per step, so per-step loss is dominated by that draw rather than by learning. |
| 3.62 | `--ckpt-every` (default 500) writes intermediate LoRA checkpoints with a validation loss at each | 1500 steps for 168 crops was a guess. With checkpoints one run answers "how many steps"; without them, every answer costs a full retrain. |
| 3.63 | **Proof that the LoRA loaded**, not assumption: a targeted weight is compared before and after merge, and the program **stops** if the difference is zero | On the first run the LoRA did not load at all. |
| 3.64 | Tile count is a measurement, not an estimate: **31.0 tiles per plate** (assumed 16.3) | A 90% error. The assumed figure came from demo-data size distributions. |
| 3.65 | The erase pass uses **classical inpainting** (`cv2.inpaint`, Telea r=7), not diffusion | The LoRA makes the erase pass *worse*, and unavoidably so: it was trained to turn a hole in the medium into a colony. `--erase-method diffusion` remains available for comparison but is not the default. Known limitation: large holes leave visible radial smearing. |
| 3.66 | Production settings: tile 512, **4 denoising steps** | 768 is worse on every axis — area grows 2.25× while tile count falls only 1.6×. And 4 steps are visually indistinguishable from 20 here, because the masks are small. Neither was knowable without measuring. |
| 3.68 | `species_check.py`: class fidelity becomes a permanent measurement with two verdicts — ordering (is the channel alive?) and separability (are classes distinguishable?), the latter computed both **with and without** the texture axis, with plate-level bootstrap intervals | Every other check asks whether a plate is *realistic*. None asked whether it is *correct*. A plate full of convincing colonies of the wrong species passes every other gate. |

## Code audit and budget correction (17 August 2026)

A read-through of the whole repository found 25 issues. The ones that became decisions:

| # | Decision | Reason |
|---|---|---|
| 3.29 | Budget arm definitions carry an explicit synthetic share instead of deriving it from the name | The old code looked for `"+S"` in the name; ablation arms do not contain it and were counted as needing zero synthetic data — the production volume was ~45% short. |
| 3.30 | `--lora-count` argument added, default 4; the hard-coded `3 *` removed | Decision 3.2 had been taken three days earlier and had no counterpart in the code. |
| 3.31 | Budget scenarios scale **training cost only** | The previous version scaled the grand total, which includes generation and XAI — costs that do not depend on epochs or image size. Savings looked larger than they were. |
| 3.32 | Baseline B: `degrees: 180.0` → `0.0` | Rotation is legitimate for plates, but Ultralytics rotates the four box corners and draws a new **axis-aligned** box; a square box grows by √2 at 45°. Baseline B would have trained on inflated labels and been tested on correct ones — a crippled baseline is a fake win for diffusion. |
| 3.33 | `scripts/collect.py` written; `results/` goes into git | `runs/` is gitignored and must stay so, but the metrics inside it are a few KB and are not reproducible without hundreds of GPU-hours. All of the paper's results were living outside version control. |
| 3.34 | `evaluate.py` matches predictions **by path**, not by order, and stops on a count mismatch | `zip(images, boxes_all)` was positional. One skipped image would shift every later pairing, with no error message. |
| 3.35 | `evaluate.py` paths are rooted at the repo root | It used relative paths and only worked from the repo root; a `splits/` directory elsewhere would have silently supplied the wrong split. |
| 3.36 | `--split train` requires `--conf-thr` | Previously it reached `counting_metrics` with `None` and raised a `TypeError`. |
| 3.37 | `src/generate/test_generate.py` written, added to the setup gate | `layout.py` contains at least as much subtle mathematics as the metrics code — inverse-CDF sampling, Strauss bisection — and had no tests at all. |
| 3.38 | `read_yolo_txt` test added to `test_metrics.py` | Every existing check built `ImageAnno` directly from pixel boxes, skipping the one place where the normalised↔pixel conversion lives. |
| 3.39 | Sampled checks removed: `written[:200]` and `lines[:50]` → the full list | On the full data these checked 2.5% and 0.6% of entries. The *entire purpose* of both checks is to catch silent label loss. |
| 3.40 | `null` forbidden in the `augment` block; the program stops pre-flight | `{k: v for k, v in ... if v is not None}` filtered nulls out and Ultralytics supplied its own default — exactly what 2.13 forbids. |
| 3.41 | W&B is driven from the config; `--smoke` really disables it | The `logging` block was read by nobody, and the help text claiming "W&B off" had no counterpart in the code. |
| 3.42 | `train.py --resume` | The notebook suggested `YOLO('last.pt').train(resume=True)`, which bypasses the wrapper entirely — so `run_metrics.json` is never written and time, VRAM and commit are lost. |
| 3.43 | `empty` and `uncountable` separated into different counters | "12,000 images were not countable" does not answer which was which, and the paper's data section must. |
| 3.44 | Broken symlinks are repaired (`is_symlink() and not exists()`) | `exists()` follows the link: if the source moved, the code tried to re-create the link and crashed with `FileExistsError`. |
| 3.45 | `check_labels.py` catches boxes extending past the edge | The old check tested centre and width in 0–1; `xc=0.99, bw=0.1` passed while the box reached 1.04. |
| 3.46 | `ultralytics>=8.4.118` | The config requires `yolo26n.pt` and `cutmix`, both introduced in 8.4. The old constraint allowed a version that could not run the config. |
| 3.47 | `setup.sh` installs `cu130`; batch suggestion corrected to the measured value | Every measurement was taken on `torch 2.13.0+cu130` while the script installed `cu124` — the cost table would have come from two different machines. |
| 3.48 | `max_det` and `nms_iou` are read from `base.yaml` | Both were hard-coded in `evaluate.py`, and the config values were read nowhere. Changing the config did nothing. |
| 3.49 | Radius normalised by **plate radius**, matching `layout.py` | It had been normalised by half the image width, with a comment claiming the plate fills the square. It does not; the out-of-plate check could never fire. |
| 3.50 | The duplicate-box IoU threshold is calibrated from the data | 0.5 was a guess, and real colonies touch (39%, overlap depth up to 0.63). On the full data the duplicate report would have filled with genuine touching pairs. |

## Phase 4 — Texture, and the gate that was not passed

| # | Decision | Reason |
|---|---|---|
| 3.69 | The texture defect is **not a quantity problem but a scale problem**: the spatial frequency of generated texture does not scale with colony size | Three measurements together: changing guidance does not move the ratio, changing the LoRA does not either. |
| 3.71 | Canvas rescaling works mechanically but the effect is insufficient. **Default off**; the option and the measurement stay on record | The control group did its job: the intervention moves correlation in the right direction but not far enough. |
| 3.72 | **3.69's framing was corrected**: the pooled −0.57 correlation is largely a *between-class* effect, not within-class | Within-class values are −0.22 / −0.42 / +0.11; the pooled figure is stronger than all of them. Small species are brighter, so pooling manufactured a correlation. |
| 3.81 | Background-pool species bias is **accepted**, the threshold is not loosened. `B.subtilis` is 0% of the pool; the A2 ablation will measure whether it matters | Loosening the threshold to fix the bias would weaken a different control. |
| 3.82 | `tiles.colony_boxes` clips context padding to the image | A colony near the plate edge produced a negative coordinate. |
| 3.83 | `adapt.py crops` prints progress and ETA | The loop reads 2,987 plates and writes ~33,000 crops while printing nothing. It was assumed hung and killed twice. **A silent long loop is itself a defect.** |
| 3.84 | **Colony count is not lognormal.** Switched to the empirical histogram; a MEAN row was added to `validate`. Supersedes 3.17 | The family was chosen on ten demo plates. On the full data it does not hold — and `validate` had not caught it because it compared medians and quantiles but not the mean, where the discrepancy lived. |
| 3.85 | Split masks are **always** written; `inpaint.py` stops without them | They had been behind a flag documented as "for visual checking", while decision 3.60 had made them mandatory in substance. |
| 3.86 | **The texture gate was not passed at any training length.** The hypothesis "the LoRA was undertrained" was eliminated by measurement. LoRA training length is not a scale but a **trade-off parameter** | Longer training makes texture worse, not better. The generator gains class separation and loses distribution fidelity. Reported as a finding. *(Two sentences of this decision were retracted in 3.90.)* |

## Phase 5 — Freeze the protocol (2–4 September 2026)

| # | Decision | Reason |
|---|---|---|
| 3.87 | **`--resume` works** — Phase 6's precondition passed. But the accounting was broken: an append-only `segments.jsonl` is now fsynced every epoch and survives `kill -9`; time, epochs and VRAM accumulate across segments | Three separate accounting errors surfaced during the test. A resumed run is not bit-identical to an uninterrupted one (0.0009 on the last epoch), and **no claim of negligibility was made** until the seed variance was known. |
| 3.88 | **Protocol frozen**: `requirements.lock` (with the torch `+cu130` trap documented in its header), `conf_thr = 0.35` written into `base.yaml` with `evaluate.py` **stopping** on a mismatch, nine tests fixed, Phase 4 outputs put under version control. Tag `phase5-protocol` (renamed from `faz5-protokol` in 3.95e) | Freezing revealed the rule-without-enforcement pattern in **three** places at once: a locked threshold sitting as `null`, a frozen environment with a missing tag, and an applied decision with un-updated tests. |
| 3.89 | **Production-LoRA selection deferred**, and tied to a measurement. Phase 6 starts with the 26 runs that need no synthetic data | Both arguments have support and neither has a measurement. "An unmeasured number is not a number" forbids locking a guess into the protocol. |
| 3.90 | A confidence interval was given to the **gate's own number**; the bootstrap's class set was fixed; the gate threshold was written into the code, and the verdict is issued over the interval. **Two sentences of 3.86 are retracted** | The gate could not decide. All four point estimates sit below the threshold, but only one interval lies entirely below it, and the bootstrap had been changing the definition of the quantity it measured between iterations — the error bar was itself measuring the wrong thing. |
| 3.91 | **Phase 5 closed.** All four levels have `bg_pool` + `crops` + `lora`. Tag `phase5-done` (renamed from `faz5-bitti` in 3.95e) | Unexpected confirmation: the crop pool changes tenfold across levels while the size of the adaptation does not (eval-loss drop 3.4–4.1% everywhere). Level differences on the substitution curve will not be coming from the LoRA. |

## Phase 6 — Run the grid (8 September 2026 onwards)

| # | Decision | Reason |
|---|---|---|
| 3.92 | **`significant_diff_threshold = 0.00399`** (2σ, three seeds, val, mAP50-95). The resume deviation (0.0009) sits below even the interval's lower bound — 3.87's suspended judgement is now made | The grid's first output had to be the threshold, not a results table. Two claims were blocked on it. |
| 3.92a | **The threshold applies to `mAP50-95` only.** Every sub-metric is judged by its own 2σ; for `mAP_large` that is 0.0877, twenty-two times the headline threshold | Principle 4, fourth occurrence: what a gate does not measure matters as much as what it does. |
| 3.92b | The three runs' **counting metrics are not comparable to each other** — the threshold selected on val moved between 0.35 and 0.40 across seeds. mAP is unaffected | 3.88's lock gained a measured justification: without it, every arm would have counted with its own ruler. |
| 3.92c | **The threshold is provisional.** σ's 90% interval spans eightfold; the grid requires five seeds for G100 anyway | Nothing better is possible with three runs, but the grid will produce more. |
| 3.92d | `train.py --help` crash fixed (`%` → `%%`); `scripts/threshold.py` added; `results/` entered git | A loud failure is better than a silent one — but a loud failure still gets fixed. |
| 3.93 | **`significant_diff_threshold = 0.007112`** (2σ, five seeds). The three-seed value under-estimated the spread by nearly half; it is kept beside the new one rather than deleted | 3.92c predicted the update. How far a small-sample sigma can stray is itself a finding (principle 3). |
| 3.93a | **Phase 2's "a chasm at mAP75" mechanism sentence is retracted.** Across five seeds the large-vs-small gap is 1.6σ at mAP75 — not distinguishable — while the same gap is 11.9σ at mAP50. Direction holds (5/5 seeds), magnitude cannot be claimed | The sentence was written from a single run, and that run was the highest of the five. |
| 3.93b | Phase 2's size table will be rewritten as five-seed mean ± σ | Small-object figures are stable (σ 0.010); large-object figures are not (σ 0.067, and 0.107 at mAP75). |
| 3.93c | Sub-metric differences are measured **paired**: take the difference per run, then look at the spread of the difference | Adding separate sigmas ignores that the two move together, and understates the comparison. |
| 3.93d | Why `large` is volatile was **not measured**; left as work to do rather than a hypothesis | *(Answered in 3.94c.)* |
| 3.94 | **`check()` now asserts under pytest.** None of the 115 checks in 30 test functions reported a verdict to pytest; "30 passed" only meant the functions had not crashed. After the fix: still 30 passed — they had all been passing | Principle 1, fourth occurrence: the rule existed in the code but had no verdict. Every commit of Phase 6 had trusted this suite. |
| 3.94a | **`eval.min_gt_for_cell: 10`.** A (class, size) cell with fewer ground-truth boxes is left out of a new `mAP*_minGT` column; dropped cells are written to `summary.json` and printed loudly | *S. aureus* has exactly **one** large box in val, and it carried a quarter of `mAP_large`. A cell with zero boxes was already excluded; one with a single box was not. |
| 3.94b | **`metrics.py` unchanged, still COCO-identical.** The filtered reading lives in `evaluate.py`, reported beside the COCO one | The two-implementation agreement of Phase 2 must hold. COCO is not wrong; it is pathological on this dataset. |
| 3.94c | **3.93d answered.** `mAP_large`'s volatility is not "large boxes clustered in few images" but one class having a nearly empty size cell | Measurement instead of hypothesis: the `n_GT_large` column of `class_ap.csv`. |
| 3.94d | `test_min_gt_cell` added (7 checks) | And this time the tests have a verdict. |
| 3.95 | **Repository made public.** Git history cleaned of personal working notes (2 documents) and of AGAR-derived images plus a model weight (312 files). 39 MB → 461 KB | Deleting a file does not remove it from history. AGAR is CC BY-NC and must not be redistributed. |
| 3.95a | **Licence is AGPL-3.0, not MIT**; `LICENSE` added | The code imports Ultralytics, which is AGPL-3.0 and requires derivative work to carry the same licence. |
| 3.95b | `KOD_HARITASI.md` → **`CODE_MAP.md`** (English, real filenames) | The old map did not know half of the 8,985 lines: filenames had been translated and the map had not been updated. A map that gives wrong information is worse than no map. |
| 3.95c | **Check count measured: 50 + 65 = 115.** README, `setup.sh` and 3.94 aligned to it | Four sources gave four different numbers; none had been obtained by running the suites. |
| 3.95d | Stale Turkish documents removed; `decisions.md` → `decisions.tr.md` with this English summary beside it | Two of the removed files were marked "provisional" in their own text; one lost its purpose when the translation it guided was completed. |
| 3.95e | **Commit messages and tags translated too.** 35 commits rewritten with `filter-repo --message-callback`; `faz5-protokol` → `phase5-protocol`, `faz5-bitti` → `phase5-done` | Filenames and file contents were English, but the file listing showed a Turkish commit message beside every row — the most visible place of all. |
| 3.96 | **G50 threshold: 0.00569** (2σ, five seeds, val, mAP50-95). Arm mean 0.68091 | Every arm of `main_grid` gets five seeds and its own measured threshold. |
| 3.96a | **First point of the substitution curve: halving the real training data costs 0.01810 mAP50-95.** The gap exceeds both arms' thresholds (t ≈ 8.9, df = 8) — it is real. Relative loss 2.6% | 2,987 → 1,491 images. The gap synthetic data has to close is small, which is the favourable case for the substitution claim. |
| 3.96b | **The loss looks larger for small objects (0.0302) but cannot be distinguished** — pooled 2σ ≈ 0.030, right at the boundary | 3.93a's lesson, repeated on the substitution curve: direction consistent, magnitude not claimable. |
| 3.96c | `G50_s0` and `G50_s1` printing the same value is **rounding**, not a seeding fault. `summary.json` rounds to five places; the two runs differ on every other measure | Two identical values would have depressed σ. Checked and ruled out; full values live under `results/raw/`. |
| 3.96d | **`threshold.py` reads the COCO column for sub-metrics, not the `minGT` one.** To be fixed | 3.94a added that column precisely to correct this pathology, and the threshold tool is unaware of it. |
| 3.97 | **G25 threshold 0.00762, G10 threshold 0.01910.** Arm means 0.66437 and 0.63705. The real-data axis is complete: 20 runs, 47.72 GPU-hours | Every arm of `main_grid` gets five seeds and carries its own threshold (3.96). |
| 3.97a | **A prediction written down before the measurement held on the primary metric.** A log-linear curve fitted to three points (G100/G50/G25) on 11 September said G10 = 0.64128; the measurement is 0.63705 — off by −0.00423, less than half of G10's own threshold | Fitting a curve to four points *after* seeing them and reporting R² = 0.998 proves nothing; four points always lie near a line. Saying the fourth point in advance is what proves something, and the curve is meant to be used as a predictor in the paper. |
| 3.97b | **The same prediction failed for small objects:** 0.46480 predicted, 0.48607 measured (+0.02127, 1.6× that metric's threshold). `mAP_small` **flattens** at the low end where the primary metric steepens (3.97e). What failed is the *shape*, not the magnitude — small-object loss is still the largest of the four (3.97h) | The deviation exceeds the threshold, so it is reported; no mechanism is claimed (3.93a, 3.96b). The two metrics do not behave alike at the low end, and the paper needs a separate sentence for each. |
| 3.97c | **Variance explodes at 299 images: σ = 0.00955, 2.5–3.4× the other three arms.** Synthetic arms at the G10 level will be compared against 0.01910, not the primary arm's 0.00711 | The first three arms sit in a narrow band (0.0028–0.0038); G10 is outside it entirely. A single shared threshold would have amounted to claiming 170% more sensitivity than the G10 data supports — 3.96's per-arm threshold decision, vindicated. |
| 3.97d | **Four-point fit: `mAP50-95 = 0.48573 + 0.01853·log2(N)`, R² = 0.99768.** 0.0185 lost per halving; all three pairwise gaps resolved (t = 8.9 / 7.8 / 5.9, df = 8) | This is the real-data axis of the substitution curve. The synthetic arms will be placed on it: "this much synthetic data is worth this many real images." |
| 3.97e | **The steps grow (0.0181 → 0.0166 → 0.0273):** on a log2 axis the curve **steepens** as data runs out | One straight line does not fully explain four points, and this is also why G10 landed slightly below prediction. Reported, not forced. |
| 3.97f | `G10_s2` selected a counting threshold of 0.45 where the other four selected 0.35. **This arm's counting metrics are not comparable seed to seed;** mAP is unaffected | mAP integrates over all confidences. `threshold.py` printed a NOTE and did not stop — correct, because the quantity being measured is mAP. |
| 3.97g | **3.96d fixed: `threshold.py` now reports both size columns** and prints which one to use. The fix is not cosmetic — `mAP_large`'s seed spread drops from 32× the primary metric to **1.1–1.7×**, R² rises from 0.86 to **0.99251**, and resolved pairwise gaps go from **0/3 to 3/3** | One ground-truth box — *S. aureus*'s only large object in val — was making the metric unusable. `mAP_small` and `mAP_medium` are identical in both columns; that one cell was the whole problem. The COCO column stays beside it because it is the number comparable to the literature (3.94b: `metrics.py` remains COCO-identical). |
| 3.97h | **Slopes by size cell: small 0.02714, primary 0.01853, medium 0.01715, large 0.01702** per halving. Small objects really do suffer ~60% more from data scarcity, and the primary metric's slope comes largely from them | The claim stands on the fit (R² = 0.97393) but only one of `mAP_small`'s three pairwise gaps is resolved. Direction consistent, single step not provable — 3.93a/3.96b for the third time. |

### The prediction test (12 September 2026)

The real-data axis of the substitution curve was measured in four arms of five
seeds each — twenty runs, 47.72 GPU-hours on one laptop GPU.

```
arm    images  n   mAP50-95    sigma   2sigma   mAP_small   GPU-h
G100     2987  5    0.69901  0.00356  0.00711     0.57422   21.74
G50      1491  5    0.68091  0.00284  0.00569     0.54398   13.79
G25       746  5    0.66437  0.00381  0.00762     0.50764    7.16
G10       299  5    0.63705  0.00955  0.01910     0.48607    5.02
```

Before any G10 run started, a curve was fitted to the first three points and
the fourth was written down:

```
mAP50-95  = 0.49893 + 0.01731 * log2(N)      R2 = 0.99938
predicted for N = 299:   0.64128 primary,  0.46480 small objects
measured:                0.63705            0.48607
                        -0.00423  HELD     +0.02127  MISSED
                    (threshold 0.01910)  (threshold 0.01344)
```

The primary metric landed inside its own measurement uncertainty; the
small-object metric did not, and missed on the high side — its curve flattens
at the low end rather than continuing to fall. The refit over all four points:

```
mAP50-95          = 0.48573 + 0.01853 * log2(N)   R2 = 0.99768   3/3 gaps resolved
mAP_medium_minGT  = 0.44705 + 0.01715 * log2(N)   R2 = 0.99807   3/3
mAP_large_minGT   = 0.54370 + 0.01702 * log2(N)   R2 = 0.99251   3/3
mAP_small         = 0.25762 + 0.02714 * log2(N)   R2 = 0.97393   1/3
```

Halving the real training data costs 0.0185 mAP50-95. Small objects lose most
(0.0271 per halving); medium and large are nearly identical and both below the
primary metric, so the headline slope comes largely from the small objects.


---

## The four design principles

These were not written down in advance. They were extracted from what kept going
wrong.

**1 — A methodological rule is a runtime check, not a comment.** A rule in prose
gets forgotten. In this repository they raise `SystemExit`. Found violated in
four separate places: a locked threshold sitting as `null` (3.88), a frozen
environment with a missing tag (3.88), an applied decision with un-updated tests
(3.88), and a test suite that could not fail (3.94).

**2 — A silent failure is worse than a loud one.** Every serious wound this
project took was silent: split lists pointing at the wrong directory while
Ultralytics warned and continued (2.0); a check that inspected a variable
instead of the file written (2.16); `max_det` set and never passed (2.17); a
colony silently dropped from a generated plate (3.22); a long loop that printed
nothing and was killed twice (3.83).

**3 — Measurement replaces guesswork, and the old guess stays next to it.**
`imgsz=1280` was a guess, then measured. The touching rate was a guess, then
measured on 387 colonies. Tile count was assumed 16.3 and measured at 31.0
(3.64). The significance threshold was 0.00399 on three seeds and 0.00711 on
five (3.93) — and the three-seed value is still in the file.

**4 — What a gate does not measure matters as much as what it does.** `validate`
was not measuring the mean, and a 42% count error hid there (3.84).
`species_check` gave its gate figure no error bar (3.90). That error bar was
itself measuring the wrong thing, because the bootstrap changed the definition
of the quantity between iterations (3.90). And the significance threshold does
not apply to sub-metrics, where the spread is up to twenty-two times larger
(3.92a).
