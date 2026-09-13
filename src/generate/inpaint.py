#!/usr/bin/env python3
"""
Mask-controlled inpainting -- the step that turns coordinates into pixels.

Input is what mask.py produced for one level:
    <in>/backgrounds/<name>.jpg   real plate, chosen from that level's subset
    <in>/masks/<name>.png         where diffusion is allowed to paint
    <in>/labels/<name>.txt        the labels, already final (decision 3.24)
    <in>/provenance/<name>.json   which background, which level, how many colonies

Output is the synthetic plate plus a copy of the label, plus timing:
    <out>/images/<name>.jpg
    <out>/labels/<name>.txt
    <out>/gen_metrics.json        tiles, sec/tile, peak VRAM, model, steps

WHY THE TIMING FILE MATTERS (decision 3.53)
-------------------------------------------
Production cost is tiles_per_image x seconds_per_tile, and seconds_per_tile
depends on the sampler and step count. At 58,200 synthetic images the difference
between 20 steps and a distilled 4-8 step sampler is ~290 GPU-hours. So this
script does not just generate; it MEASURES, and scripts/budget.py consumes the
measurement. Never hand budget.py a guessed --sec-per-tile once a real run
exists.

SUBCOMMANDS
    run    generate synthetic plates
    bench  sweep (tile size x steps) on a few plates and report cost, so the
           two budget levers are chosen from data instead of taste

THE DRY MODE
------------
--dry replaces the diffusion call with a deterministic stub painter. Everything
else -- tiling, prompting, compositing, the mask gate, label handling, timing --
runs for real. It exists so the plumbing can be verified on a machine without a
GPU, and so a plumbing bug is never mistaken for a generation-quality problem.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiles as T                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# SD 1.5 inpainting. Small enough for an 8 GB card at 512, and -- unlike SDXL --
# actually trained at the tile size we need. If this is ever changed, record it
# in decisions.md: the base model is part of the protocol, not a detail.
DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"

# Decision 3.54: the erase regions (decision 3.25) must come back as PLAIN AGAR.
# If the model invents a colony there, the image contains an unlabelled object --
# exactly the false negative that erasing the real colonies was meant to prevent,
# reintroduced by the generator itself. A negative prompt is a weak guard, not a
# guarantee; Phase 4 must measure the residual rate with a detector.
NEGATIVE_PROMPT = ("blurry, out of focus, text, watermark, drawing, illustration, "
                   "duplicate, deformed, low quality, jpeg artifacts")


def species_prompt(colonies, idx) -> str:
    """
    Prompt for one tile, naming only the species actually inside it.

    A tile can hold two species (decision 3.16: one dominant + a few secondary),
    so the prompt is built per tile rather than per plate.
    """
    names = sorted({colonies[i]["cls"] for i in idx})
    if not names:
        return "a clean agar plate surface, macro photograph"
    listed = " and ".join(names)
    return (f"macro photograph of {listed} bacterial colonies growing on an agar "
            f"plate, sharp focus, even studio lighting, top-down view")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def attach_lora(unet, lora: str):
    """
    Attach the LoRA that adapt.py wrote, and PROVE that it attached
    (decision 3.63).

    adapt.py adapts the UNet with peft (`get_peft_model`) and saves it with
    `save_pretrained`. That writes `adapter_config.json` +
    `adapter_model.safetensors`, whose keys are prefixed `base_model.model.`.
    `pipe.load_lora_weights()` expects the diffusers/kohya naming
    (`unet.down_blocks...`), finds no match, and prints

        No LoRA keys associated to UNet2DConditionModel found with prefix='unet'
        This is safe to ignore if ...

    as a WARNING. It then generates happily from the completely unadapted base
    model. This is the exact failure mode design principle 2 exists for: the
    whole 58,200-image grid could be produced by a model that never saw an agar
    plate, and nothing in the output would say so -- the images would just be
    "a bit disappointing", which is indistinguishable from "diffusion does not
    work for this", which is the paper's headline claim.

    (It already happened once, on the first real generation run of the project.)

    So: load through peft, merge into the base weights, and verify by comparing
    one targeted weight before and after. No change -> stop.
    """
    from peft import PeftModel

    p = Path(lora)
    if not (p / "adapter_config.json").exists():
        sys.exit(f"ERROR (decision 3.63): {p} has no adapter_config.json. "
                 f"adapt.py writes a peft adapter directory; this is not one.")

    # A module LoraConfig actually targets (to_q of a self-attention block).
    ref_name, ref_before = None, None
    for n, t in unet.named_parameters():
        if n.endswith("attn1.to_q.weight"):
            ref_name, ref_before = n, t.detach().clone()
            break
    if ref_name is None:
        sys.exit("ERROR (decision 3.63): no attn1.to_q.weight in the UNet; the "
                 "verification below cannot run, so the LoRA cannot be trusted.")

    unet = PeftModel.from_pretrained(unet, str(p))
    n_lora = sum(1 for n, _ in unet.named_parameters() if "lora_" in n)
    if n_lora == 0:
        sys.exit(f"ERROR (decision 3.63): {p} loaded but contains no lora_ "
                 f"tensors.")
    # merge_and_unload() folds the low-rank update into the base weights and
    # gives back a plain UNet2DConditionModel -- no wrapper in the hot path, so
    # inference speed and the pipeline's attribute access are unaffected.
    unet = unet.merge_and_unload()

    after = dict(unet.named_parameters())[ref_name]
    delta = (after - ref_before).abs().max().item()
    if delta == 0.0:
        sys.exit(
            f"ERROR (decision 3.63): {p} loaded ({n_lora} lora tensors) but "
            f"merging changed nothing -- {ref_name} is bit-identical. The "
            f"adapter is empty, or it targets modules this UNet does not have. "
            f"Refusing to generate from an unadapted model.")
    print(f"  [lora] {p.name}: {n_lora} tensors merged, "
          f"max |delta| = {delta:.5f}  ({ref_name})")
    return unet


def load_pipeline(model: str, lora: str | None, device: str, steps: int):
    """Load the inpainting pipeline. Imported lazily so --dry needs no torch."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    dtype = torch.float16 if device != "cpu" else torch.float32
    # adapt.py trains in fp32. Merging an fp32 low-rank update into fp16 base
    # weights rounds part of it away, so load fp32, merge, then cast.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model, torch_dtype=torch.float32 if lora else dtype,
        safety_checker=None, requires_safety_checker=False)
    if lora:
        # Decision 3.2: the LoRA is per level. inpaint.py does not check which
        # level it belongs to -- run() does, from the provenance file, before we
        # get here.
        pipe.unet = attach_lora(pipe.unet, lora)
        # Cast the modules directly: DiffusionPipeline.to()'s dtype keyword has
        # changed name across diffusers versions, nn.Module.to(dtype) has not.
        for m in (pipe.unet, pipe.vae, pipe.text_encoder):
            m.to(dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _stub_paint(crop_bgr, mask_crop, rng):
    """
    Deterministic fake 'generation' for --dry.

    Paints a plausible colony-ish blob inside the mask so the composite path,
    the mask gate and the label geometry can all be checked. It is intentionally
    ugly: nobody should ever mistake a dry-mode output for a real sample.
    """
    out = crop_bgr.copy()
    m = mask_crop > 0
    if not m.any():
        return out
    base = np.array([200, 205, 195], dtype=np.float32)
    noise = rng.normal(0, 12, size=(*crop_bgr.shape[:2], 3)).astype(np.float32)
    painted = np.clip(base + noise, 0, 255).astype(np.uint8)
    out[m] = painted[m]
    return out


# ---------------------------------------------------------------------------
# one plate
# ---------------------------------------------------------------------------

CLEAN_PROMPT = ("clean agar plate surface, smooth uniform medium, no colonies, "
                "even studio lighting, top-down view")

# Decision 3.65: the erase pass is CLASSICAL by default, not diffusion.
#
# Measured on the first real plate (35 erase regions, level 100):
#
#     erase pass                 regions that came back as a colony
#     diffusion + lora_100       23 / 35   (66 %)
#     diffusion, base model       6 / 35   (17 %)
#     cv2.inpaint (Telea, r=7)    0 / 35   ( 0 %)
#     (reference: untouched real background = 34 / 35)
#
# The LoRA makes the erase pass WORSE, and necessarily so: it was trained to
# turn a hole on agar into a colony, and pass 1 hands it exactly that. Prompting
# cannot undo a weight update. Classical inpainting cannot hallucinate a colony
# because it has nothing to hallucinate with -- it propagates the surrounding
# medium inward, which is precisely what "plain agar" means here.
#
# It is also free: the erase grid was 12 of the 31 tiles on that plate, so the
# whole pass leaves the GPU budget (-39 %).
#
# The cost is that classical inpainting blurs when the hole is large relative to
# its surroundings. The background pool only admits plates with erase ratio
# <= 0.06 (decision 3.26), so holes are small in practice -- but "in practice"
# is not a guarantee, so the diameter is measured and reported per plate.
ERASE_RADIUS = 7
ERASE_WARN_PX = 120

# Decision 3.70: the canvas the model paints on may be RESCALED per tile.
#
# Measured 19 August: real colonies get SMOOTHER as they get larger (rank
# correlation between diameter and texture: -0.57). Generated colonies do not
# (-0.18). The generator stamps texture at a fixed spatial frequency regardless
# of how big the colony is -- which is exactly what a fixed latent grid does.
# SD's VAE downsamples by 8, so the invented structure has a fixed size in
# CANVAS pixels; a 29 px S.aureus colony is ~3.6 latent pixels and cannot come
# out smooth, while a 155 px colony gets the same fine structure a small one
# does.
#
# Decision 3.51 solved this at the PLATE scale (tile instead of downscale).
# This is the same problem one level down, at the COLONY scale.
#
# The fix: resample the tile so that a colony occupies a CONSTANT number of
# canvas pixels. Then the model's fixed-frequency texture lands at the same
# RELATIVE scale on every colony, and in image space the texture frequency
# scales with the colony -- which is the real behaviour.
#
# The mask gate still uses the FULL-RESOLUTION mask (tiles.composite), so no
# resampling can move a colony outside its labelled disc. Decision 3.24 holds
# regardless of what happens on the canvas.
#
# Cost is quadratic in the canvas: a tile at 768 costs 2.25x one at 512. The
# clamp keeps that bounded, and run() prints the measured multiplier.
MIN_CANVAS = 256
MAX_CANVAS = 768
TARGET_DIAMETER = 128        # canvas px a colony should occupy, when enabled


def canvas_for(tile: int, gen_scale: float, target_diameter: int,
               colonies, idx, W: int) -> int:
    """
    Canvas size for one tile (decision 3.70).

    target_diameter > 0 -> per-colony scaling: the tile's median colony is made
    `target_diameter` canvas pixels wide. Otherwise the flat `gen_scale` is used
    (1.0 = the original behaviour, canvas == tile).
    """
    s = gen_scale
    if target_diameter and idx:
        d = float(np.median([colonies[i]["diameter"] * W for i in idx]))
        if d > 0:
            s = target_diameter / d
    c = int(round(tile * s / 8) * 8)
    return int(min(max(c, MIN_CANVAS), MAX_CANVAS))


def erase_classical(img: np.ndarray, erase: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Replace the erase regions with plain agar, without a generative model.

    Returns (image, stats). The stats are reported, not asserted: a large hole
    is a diversity/realism risk, not a correctness bug, and the right response
    is to look at the plate -- see design principle 2.
    """
    m = (erase > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    diam = [max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, n)]
    out = cv2.inpaint(img, m, ERASE_RADIUS, cv2.INPAINT_TELEA)
    return out, {
        "method": "classical",
        "regions": int(n - 1),
        "max_diameter_px": int(max(diam)) if diam else 0,
        "area_ratio": round(float(m.sum()) / m.size, 4),
    }


def generate_plate(bg_path: Path, mask_path: Path, plan: dict, pipe,
                   tile: int, steps: int, guidance: float, strength: float,
                   seed: int, dry: bool, rng,
                   erase_mask_path: Path | None = None,
                   erase_method: str = "classical",
                   gen_scale: float = 1.0,
                   target_diameter: int = 0
                   ) -> tuple[np.ndarray, list[float]]:
    """
    Returns (synthetic plate BGR, per-tile seconds).

    TWO PASSES (decision 3.60)
    --------------------------
    Pass 1 -- ERASE: the regions where mask.py removed the background plate's
              real colonies are repainted as PLAIN AGAR.
    Pass 2 -- SYNTHESISE: the synthetic colony disks are painted on the now-clean
              plate.

    Why not one pass over the combined mask, as the first version did:

        With a single binary mask the model CANNOT KNOW which holes should
        become colonies and which should become empty medium. Both are just
        holes. It will fill them all -- and every filled erase region is an
        UNLABELLED colony in the image, i.e. the exact false negative that
        erasing was introduced to prevent (decision 3.25), reintroduced by the
        generator.

    That is a representation problem, not a prompting problem: no negative
    prompt fixes an input that does not distinguish the two cases. Splitting the
    pass is what makes each question unambiguous. Each pass gets its own prompt,
    and the erase pass gets CLEAN_PROMPT.

    If erase_mask_path is None the erase pass is skipped, which is correct only
    when the background genuinely has no colonies -- which AGAR never does
    (decision 3.25).
    """
    bg = cv2.imread(str(bg_path))
    if bg is None:
        sys.exit(f"ERROR: cannot read background {bg_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit(f"ERROR: cannot read mask {mask_path}")
    H, W = bg.shape[:2]

    colonies = plan["colonies"]
    out = bg.copy()
    per_tile = []
    canvases: list[int] = []
    n = 0

    # ---- pass 1: erase -> plain agar ---------------------------------------
    erase = None
    if erase_mask_path is not None and Path(erase_mask_path).exists():
        erase = cv2.imread(str(erase_mask_path), cv2.IMREAD_GRAYSCALE)
    if erase is not None and (erase > 0).any() and erase_method == "classical":
        # Decision 3.65. Free, and measured to be strictly better at the one
        # thing this pass has to do: NOT produce a colony.
        out, est = erase_classical(out, erase)
        if est["max_diameter_px"] > ERASE_WARN_PX:
            print(f"  [!] largest erased region is {est['max_diameter_px']} px. "
                  f"Classical inpainting smooths holes this size; check the "
                  f"plate before trusting it (decision 3.65).")
    elif erase is not None and (erase > 0).any():
        # Grid over the erase regions: these are not colonies we placed, so the
        # colony-based tiler does not apply. A simple bounding-box grid is enough
        # and there is no split-invariant to preserve (nothing must line up).
        ys, xs = np.nonzero(erase)
        for gy in range(int(ys.min()), int(ys.max()) + 1, tile):
            for gx in range(int(xs.min()), int(xs.max()) + 1, tile):
                y0 = min(gy, H - tile); x0 = min(gx, W - tile)
                t = T.Tile(max(0, x0), max(0, y0), tile, [])
                if (erase[t.y0:t.y1, t.x0:t.x1] > 0).sum() == 0:
                    continue
                crop = out[t.y0:t.y1, t.x0:t.x1]
                t0 = time.perf_counter()
                if dry:
                    gen = _stub_paint(crop, erase[t.y0:t.y1, t.x0:t.x1], rng)
                else:
                    from PIL import Image
                    import torch
                    g = torch.Generator(device=pipe.device).manual_seed(seed * 1000 + n)
                    res = pipe(prompt=CLEAN_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                               image=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                               mask_image=Image.fromarray(erase[t.y0:t.y1, t.x0:t.x1]),
                               height=tile, width=tile, num_inference_steps=steps,
                               guidance_scale=guidance, strength=strength,
                               generator=g).images[0]
                    gen = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
                per_tile.append(time.perf_counter() - t0)
                T.composite(out, gen, erase, t, feather=32)
                n += 1

    # ---- pass 2: synthesise the colonies -----------------------------------
    windows = T.place_tiles(colonies, W, H, tile=tile)
    for t in windows:
        crop = out[t.y0:t.y1, t.x0:t.x1]
        mcrop = mask[t.y0:t.y1, t.x0:t.x1]
        if (mcrop > 0).sum() == 0:
            continue                       # nothing to paint here

        # Decision 3.70: the canvas may differ from the tile.
        canvas = canvas_for(tile, gen_scale, target_diameter,
                            colonies, t.colony_idx, W)
        canvases.append(canvas)
        if canvas != tile:
            down = canvas < tile
            crop_in = cv2.resize(crop, (canvas, canvas),
                                 interpolation=cv2.INTER_AREA if down else cv2.INTER_CUBIC)
            mask_in = cv2.resize(mcrop, (canvas, canvas),
                                 interpolation=cv2.INTER_NEAREST)
        else:
            crop_in, mask_in = crop, mcrop

        t0 = time.perf_counter()
        if dry:
            gen = _stub_paint(crop_in, mask_in, rng)
        else:
            from PIL import Image
            import torch
            g = torch.Generator(device=pipe.device).manual_seed(seed * 1000 + n)
            img = Image.fromarray(cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB))
            msk = Image.fromarray(mask_in)
            res = pipe(prompt=species_prompt(colonies, t.colony_idx),
                       negative_prompt=NEGATIVE_PROMPT,
                       image=img, mask_image=msk,
                       height=canvas, width=canvas,
                       num_inference_steps=steps, guidance_scale=guidance,
                       strength=strength, generator=g).images[0]
            gen = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
        if canvas != tile:
            gen = cv2.resize(gen, (tile, tile),
                             interpolation=cv2.INTER_CUBIC if canvas < tile else cv2.INTER_AREA)
        per_tile.append(time.perf_counter() - t0)
        n += 1

        # The mask gate (see tiles.composite): whatever the model did outside
        # its mask never reaches the plate. This is what keeps the untouched
        # background byte-identical to the real plate.
        T.composite(out, gen, mask, t, feather=32)

    if canvases and any(c != tile for c in canvases):
        # Cost is quadratic in the canvas; report the multiplier rather than
        # letting it appear later as an unexplained slowdown.
        mult = float(np.mean([(c / tile) ** 2 for c in canvases]))
        print(f"  [scale] canvas {min(canvases)}-{max(canvases)} px "
              f"(tile {tile}), cost x{mult:.2f}")

    return out, per_tile


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _peak_vram(dry: bool):
    if dry:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    except Exception:
        pass
    return None


def run(args):
    src = Path(args.input)
    prov_dir = src / "provenance"
    provs = sorted(prov_dir.glob("*.json"))
    if not provs:
        sys.exit(f"ERROR: no provenance files under {prov_dir}. Run mask.py build first.")
    if args.limit:
        provs = provs[:args.limit]

    # Decision 3.7 / 3.2: the level must agree everywhere, and the LoRA must be
    # the one trained on THIS level. A mismatch here is the leakage the whole
    # protocol exists to prevent, so it stops the program.
    first = json.loads(provs[0].read_text())
    if first["level"] != args.level:
        sys.exit(f"ERROR (decision 3.7): {provs[0].name} has level={first['level']}, "
                 f"--level {args.level} was given.")
    if args.lora and f"_{args.level}" not in Path(args.lora).name and not args.dry:
        sys.exit(f"ERROR (decision 3.2 / 3.7): --lora {args.lora} does not carry "
                 f"level {args.level} in its name. Per-level LoRA is what keeps "
                 f"'only X% real data' true; refusing to guess.")

    out = Path(args.out)

    # --- resume (decision 3.100) --------------------------------------------
    # Filtered BEFORE the pipeline is loaded: if there is nothing left to do
    # there is no reason to spend a model load, and more importantly no reason
    # to touch anything in --out.
    n_skipped = 0
    if args.resume:
        todo = []
        for pf in provs:
            name = json.loads(pf.read_text())["name"]
            if (out / "images" / f"{name}.jpg").exists() and \
               (out / "labels" / f"{name}.txt").exists():
                n_skipped += 1
            else:
                todo.append(pf)
        print(f"  [resume] {n_skipped} plates already present, "
              f"{len(todo)} to generate")
        if not todo:
            # Return WITHOUT writing gen_metrics.json. A cron-driven retry that
            # finds the level already finished would otherwise overwrite the
            # real measurement with a file describing zero plates -- destroying
            # the sec/tile figure that scripts/budget.py depends on and that
            # cost GPU-hours to obtain. Decision 3.53: this script measures,
            # and a measurement must not be clobbered by a no-op.
            print(f"  [resume] nothing to do; {out/'gen_metrics.json'} left "
                  f"untouched")
            return
        provs = todo

    pipe = None
    if not args.dry:
        pipe = load_pipeline(args.model, args.lora, args.device, args.steps)

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_tiles, all_secs = 0, []
    t_start = time.perf_counter()

    for i, pf in enumerate(provs):
        prov = json.loads(pf.read_text())
        name = prov["name"]
        plan_path = Path(args.plans) / "plans" / f"{name}.json"
        if not plan_path.exists():
            sys.exit(f"ERROR: layout plan missing for {name} ({plan_path})")
        plan = json.loads(plan_path.read_text())

        # decision 3.60: two passes need the two masks separately. mask.py
        # writes them with --split-masks; the combined mask is the fallback and
        # triggers a warning, because a single mask cannot express "fill this
        # one, empty that one".
        syn_p = src / "masks" / f"{name}_synthetic.png"
        er_p = src / "masks" / f"{name}_erase.png"
        if not syn_p.exists():
            # Decision 3.85. This used to warn once and then quietly carry on
            # with the combined mask. That is worse than stopping: the run looks
            # successful, writes a full set of plates, and every one of them can
            # contain a colony painted into an erase region -- an UNLABELLED
            # object, which is exactly the false negative decisions 3.25 and 3.60
            # exist to prevent. A warning that scrolls off the top of a 20-minute
            # log does not protect anyone.
            #
            # The one-pass behaviour is still reachable, but only as a DELIBERATE
            # choice via --allow-combined-mask, so that it is recorded in the
            # command rather than inferred from a missing file.
            if not args.allow_combined_mask:
                sys.exit(
                    f"ERROR (decision 3.85): split masks not found for '{name}'.\n"
                    f"  Expected: {syn_p}\n"
                    f"  inpaint.py generates in TWO passes (decision 3.60): erase\n"
                    f"  regions become plain agar, synthetic disks become colonies.\n"
                    f"  A single combined mask cannot express that difference, so the\n"
                    f"  model fills the erase regions too and every one of them is an\n"
                    f"  unlabelled colony.\n"
                    f"  Fix: re-run `mask.py build` (it now always writes the split\n"
                    f"  masks). Only pass --allow-combined-mask if you deliberately\n"
                    f"  want the one-pass ablation.")
            if i == 0:
                print("  [!] --allow-combined-mask: running the ONE-PASS ablation. "
                      "Erase regions\n      may be filled with unlabelled colonies "
                      "(decisions 3.60, 3.85). This\n      output must not be used "
                      "for the ghost-colony gate.")
            syn_p = src / "masks" / f"{name}.png"
            er_p = None

        img, secs = generate_plate(
            src / "backgrounds" / f"{name}.jpg", syn_p,
            plan, pipe, args.tile, args.steps, args.guidance, args.strength,
            args.seed, args.dry, rng, erase_mask_path=er_p,
            erase_method=args.erase_method,
            gen_scale=args.gen_scale, target_diameter=args.target_diameter)

        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(src / "labels" / f"{name}.txt", out / "labels" / f"{name}.txt")
        all_tiles += len(secs)
        all_secs.extend(secs)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(provs)} plates, {all_tiles} tiles, "
                  f"{np.mean(all_secs):.2f} s/tile")

    total = time.perf_counter() - t_start
    a = np.asarray(all_secs) if all_secs else np.zeros(1)
    metrics = {
        "mode": "dry" if args.dry else "real",
        "model": None if args.dry else args.model,
        "lora": args.lora,
        "level": args.level,
        "tile": args.tile,
        "steps": args.steps,
        "guidance": args.guidance,
        "strength": args.strength,
        # These describe THIS invocation. On a resumed run len(provs) is the
        # number generated now, not the number in --out; the two extra fields
        # say so rather than leaving a reader to assume. sec_per_image and
        # tiles_per_image stay honest because they divide by the generated
        # count -- dividing by the total would report a machine several times
        # faster than it is.
        "n_plates": len(provs),
        "n_plates_skipped_resume": n_skipped,
        "n_plates_in_out": len(provs) + n_skipped,
        "resumed": bool(args.resume and n_skipped),
        "n_tiles": all_tiles,
        "tiles_per_image": round(all_tiles / max(len(provs), 1), 2),
        "sec_per_tile_mean": round(float(a.mean()), 4),
        "sec_per_tile_p95": round(float(np.percentile(a, 95)), 4),
        "sec_per_image": round(total / max(len(provs), 1), 3),
        "total_sec": round(total, 1),
        "peak_vram_gb": _peak_vram(args.dry),
    }
    (out / "gen_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n[inpaint] {len(provs)} plates - {all_tiles} tiles - "
          f"level={args.level} - {'DRY' if args.dry else args.model}")
    print(f"          tiles/image      : {metrics['tiles_per_image']}")
    print(f"          sec/tile         : {metrics['sec_per_tile_mean']} "
          f"(p95 {metrics['sec_per_tile_p95']})")
    print(f"          sec/image        : {metrics['sec_per_image']}")
    if metrics["peak_vram_gb"]:
        print(f"          peak VRAM        : {metrics['peak_vram_gb']} GB")
    if args.dry:
        print("  [!] DRY MODE -- no diffusion ran. Timings measure the plumbing "
              "only.\n      Do NOT feed this sec/tile to budget.py.")
    else:
        h = 58200 * metrics["tiles_per_image"] * metrics["sec_per_tile_mean"] / 3600
        print(f"          -> 58,200 images would cost {h:.0f} GPU-hours "
              f"at this setting")
        print(f"          scripts/budget.py --tiles-per-image "
              f"{metrics['tiles_per_image']} --sec-per-tile "
              f"{metrics['sec_per_tile_mean']}")
    print(f"          -> {out}")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def bench(args):
    """
    Sweep the two budget levers (decision 3.53) and print the cost table.

    This is the command Phase 4 runs before anyone quotes a GPU-hour figure.
    Quality is NOT measured here -- that needs eyes and FID. This answers only
    'what does each setting cost', which is half the decision.
    """
    src = Path(args.input)
    provs = sorted((src / "provenance").glob("*.json"))[:args.plates]
    if not provs:
        sys.exit(f"ERROR: no provenance files under {src/'provenance'}")

    # The pipeline does not depend on tile size or step count, so it is loaded
    # ONCE. The first version reloaded it per combination -- six model loads,
    # several minutes of pure waste, and the fp32-load-then-merge path this file
    # now uses for LoRA made that worse.
    pipe = None if args.dry else load_pipeline(
        args.model, args.lora, args.device, 0)

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tile in [int(x) for x in args.tiles.split(",")]:
        for steps in [int(x) for x in args.steps_list.split(",")]:
            rng = np.random.default_rng(0)
            secs, n_tiles = [], 0
            for pf in provs:
                prov = json.loads(pf.read_text())
                name = prov["name"]
                plan = json.loads((Path(args.plans) / "plans" / f"{name}.json").read_text())
                syn_p = src / "masks" / f"{name}_synthetic.png"
                er_p = src / "masks" / f"{name}_erase.png"
                if not syn_p.exists():
                    syn_p, er_p = src / "masks" / f"{name}.png", None
                img, s = generate_plate(
                    src / "backgrounds" / f"{name}.jpg", syn_p,
                    plan, pipe, tile, steps, args.guidance, args.strength,
                    0, args.dry, rng, erase_mask_path=er_p,
                    erase_method=args.erase_method,
            gen_scale=args.gen_scale, target_diameter=args.target_diameter)
                # Cost without a picture is half an answer: the closing note
                # below says "pick a setting only after LOOKING at it", so the
                # cheapest candidate has to be lookable. One plate per setting.
                if out_dir is not None and pf is provs[0]:
                    cv2.imwrite(str(out_dir / f"tile{tile}_steps{steps}.jpg"), img)
                secs.extend(s); n_tiles += len(s)
            tpi = n_tiles / len(provs)
            spt = float(np.mean(secs)) if secs else 0.0
            rows.append((tile, steps, tpi, spt, 58200 * tpi * spt / 3600))
            print(f"  [bench] tile={tile} steps={steps}: {tpi:.1f} tiles, "
                  f"{spt:.3f} s/tile")

    print(f"\n{'tile':>6} {'steps':>6} {'tiles/img':>10} {'s/tile':>9} "
          f"{'58,200 images':>16}")
    print("-" * 52)
    for tile, steps, tpi, spt, hours in rows:
        print(f"{tile:>6} {steps:>6} {tpi:>10.1f} {spt:>9.3f} {hours:>13.0f} h")
    if args.dry:
        print("\n[!] DRY -- these seconds measure plumbing, not diffusion.")
    print("\nQuality is not measured here. Pick a setting only after LOOKING at "
          "the output\nof the cheapest candidate (Phase 3 gate: 'could this be "
          "real?').")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--input", required=True, help="mask.py build output")
        p.add_argument("--plans", required=True, help="layout.py sample output")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--lora", default=None,
                       help="LoRA weights for THIS level (decision 3.2)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--guidance", type=float, default=7.5)
        p.add_argument("--strength", type=float, default=1.0)
        p.add_argument("--gen-scale", type=float, default=1.0,
                       help="flat canvas rescale (decision 3.70). 1.0 = the "
                            "original behaviour. <1 makes the generated texture "
                            "COARSER in image pixels, >1 finer.")
        p.add_argument("--target-diameter", type=int, default=0,
                       help="canvas px a colony should occupy; overrides "
                            "--gen-scale per tile (decision 3.70). "
                            f"0 = off, {TARGET_DIAMETER} = the calibrated value.")
        p.add_argument("--allow-combined-mask", action="store_true",
                       help="run the ONE-PASS ablation when the split masks are "
                            "missing, instead of stopping (decision 3.85). The "
                            "erase regions may then be filled with UNLABELLED "
                            "colonies, so the output is not valid for the "
                            "ghost-colony gate. Never use it to get past an error.")
        p.add_argument("--erase-method", default="classical",
                       choices=["classical", "diffusion"],
                       help="how the erase regions become plain agar "
                            "(decision 3.65). classical is free and measured "
                            "not to hallucinate colonies; diffusion is kept "
                            "only so the comparison can be reproduced.")
        p.add_argument("--dry", action="store_true",
                       help="no diffusion; exercise the whole pipeline with a "
                            "stub painter. For plumbing checks on a GPU-less box.")

    p = sub.add_parser("run", help="generate synthetic plates")
    common(p)
    # decision 3.5: no default level, ever
    p.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100])
    p.add_argument("--tile", type=int, default=T.DEFAULT_TILE)
    # Decision 3.66: 4, not the usual 20. Measured -- visually indistinguishable
    # from 20 here (small masks, very strong surrounding context) and closer to
    # the real colonies in texture, at 1/3.6 of the cost. See `bench`.
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N plates")
    # Decision 3.100. Generation is the longest single step in this project --
    # one level costs 8 to 13 GPU-hours -- and without this a run killed in its
    # twelfth hour starts again from zero. That is not hypothetical: on
    # 13 September the kernel OOM killer took a python process mid-run and ten
    # and a half hours went idle before anyone noticed. Explicit rather than
    # automatic, so that a resumed run is visible in the command that produced
    # it (the same reason --allow-combined-mask is a flag).
    p.add_argument("--resume", action="store_true",
                   help="skip plates whose image AND label are already under "
                        "--out, so an interrupted generation can continue")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=run)

    p = sub.add_parser("bench", help="sweep tile size x steps, report cost")
    common(p)
    p.add_argument("--tiles", default="512,768")
    p.add_argument("--steps-list", default="4,8,20")
    p.add_argument("--plates", type=int, default=5)
    p.add_argument("--out", default=None,
                   help="write one plate per setting here, so the cheapest "
                        "candidate can be LOOKED at before it is chosen")
    p.set_defaults(fn=bench)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
