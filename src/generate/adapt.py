#!/usr/bin/env python3
"""
LoRA adaptation of the inpainting model -- teaches it what an AGAR colony looks
like, using ONLY the real images of one data level.

WHY THIS IS PER LEVEL (decision 3.2 / 3.3)
------------------------------------------
The G10+S arm claims "only 10% of the real data was used". If the generator that
produced its synthetic images was adapted on 100% of the real data, the model has
already seen the 90% that was supposedly held out, and that information leaks
into every synthetic image. The claim collapses and a reviewer will find it.

So there are FOUR adaptations, not one:

    lora_10   trained on train_10   -> feeds G10+S
    lora_25   trained on train_25   -> feeds G25+S, the amount sweep, ablations
    lora_50   trained on train_50   -> feeds G50+S
    lora_100  trained on train      -> feeds S100

Because the subsets are nested (decision 1.6), lora_10's data is a subset of
lora_25's. That is not a problem -- it is the point: every point on the
substitution curve is generated with its own information budget.

Training data comes from the TRAIN split only. val and test are never shown to
the diffusion model, in any level, at any stage (decision 3.3). Indirect contact
through a generator is still contact.

TRAINING MIMICS INFERENCE
-------------------------
At generation time we hand the model a plate with disk-shaped holes where
colonies should go, and ask it to fill them. So training does exactly that:
take a real tile, punch out the real colonies' disks, ask the model to
reconstruct them. Same window size, same mask shape, same prompt template as
inpaint.py -- if training and inference disagree on any of these, the adapted
model is being asked a question it was never trained on.

The crops are placed with the SAME tiles.place_tiles() used at generation time,
on the REAL labels. The model therefore sees exactly the kind of window it will
later be asked to paint.

SUBCOMMANDS
    crops  extract and inspect the training set (no GPU needed)
    train  run the LoRA fine-tune
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
import tiles as T                                   # noqa: E402
from inpaint import DEFAULT_MODEL, species_prompt   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Same margin as generation, so the training window matches the inference window.
CONTEXT_PAD = T.CONTEXT_PAD


def read_labels(stem: str, labels_dir: Path, classes: list[str]):
    """Real YOLO labels -> the colony dicts the tiling layer expects."""
    p = labels_dir / f"{stem}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, xc, yc, w, h = line.split()
        out.append({"cls": classes[int(c)], "xc": float(xc), "yc": float(yc),
                    "w": float(w), "h": float(h),
                    "diameter": (float(w) + float(h)) / 2})
    return out


def extract_crops(list_file: Path, labels_dir: Path, images_dir: Path,
                  classes: list[str], tile: int, out_dir: Path,
                  max_crops: int | None = None, empty_ratio: float = 0.20) -> dict:
    """
    Build the (image, mask, caption) training set.

    One crop per tile, so the training distribution of "how many colonies in a
    window, how big, which species" is the real one rather than something
    invented.
    """
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    stems = [Path(l).stem for l in list_file.read_text(encoding="utf-8").split() if l.strip()]
    records, skipped_big, per_species = [], 0, {}

    # Decision 3.83: this loop reads a 2048x2048 JPEG and writes ~19 crop pairs
    # per plate. Over the 2987-plate train split that is ~57k crops and roughly
    # an hour, and the previous version printed NOTHING until it finished. Twice
    # in a row it was mistaken for a hang and killed -- and because
    # metadata.jsonl is only written at the very end, everything on disk was
    # then useless. A silent long-running loop is a bug in its own right.
    n_plates = len(stems)
    print(f"[crops] {n_plates} plates -> {out_dir}")
    t_start = time.perf_counter()

    for pi, stem in enumerate(stems, 1):
        if pi % 25 == 0 or pi == n_plates:
            el = time.perf_counter() - t_start
            eta = el / pi * (n_plates - pi)
            print(f"\r  plate {pi}/{n_plates}  ({pi/n_plates*100:5.1f}%)  "
                  f"crops {len(records)}  elapsed {el/60:.1f} min  "
                  f"eta {eta/60:.1f} min", end="", flush=True)
        colonies = read_labels(stem, labels_dir, classes)
        if not colonies:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img = cv2.imread(str(cand))
                break
        if img is None:
            print(f"  WARNING: image not found for {stem}, skipped", file=sys.stderr)
            continue
        H, W = img.shape[:2]

        try:
            windows = T.place_tiles(colonies, W, H, tile=tile)
        except ValueError:
            # A colony larger than the tile. At generation time this is a hard
            # error; here we skip the plate and COUNT it, because a silently
            # missing plate would quietly bias the training set (decision 3.22).
            skipped_big += 1
            continue

        for n, t in enumerate(windows):
            crop = img[t.y0:t.y1, t.x0:t.x1]
            mask = np.zeros((tile, tile), np.uint8)
            # Decision 3.58: mask EVERY colony that lands in this window, not
            # only the ones the tiler assigned to it.
            #
            # The first version masked t.colony_idx alone, which left
            # neighbouring colonies visible inside the crop. A visual check made
            # the mismatch obvious: at generation time mask.py has already
            # ERASED every real colony from the background (decision 3.25), so
            # the model is handed a plate with no visible colony at all. Training
            # on crops that still showed colonies taught it to copy a neighbour's
            # appearance -- a cue that does not exist at inference.
            visible = []
            for i, k in enumerate(colonies):
                cx = int(round(k["xc"] * W)) - t.x0
                cy = int(round(k["yc"] * H)) - t.y0
                r = max(1, int(round(k["diameter"] / 2 * W)))
                if -r <= cx <= tile + r and -r <= cy <= tile + r:
                    cv2.circle(mask, (cx, cy), r, 255, -1)
                    visible.append(i)

            name = f"{stem}_{n:03d}"
            cv2.imwrite(str(out_dir / "images" / f"{name}.png"), crop)
            cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), mask)
            caption = species_prompt(colonies, visible)
            records.append({"name": name, "source_plate": stem,
                            "caption": caption, "kind": "colony",
                            "n_colonies": len(visible),
                            "mask_fraction": round(float((mask > 0).mean()), 5)})
            for i in visible:
                s = colonies[i]["cls"]
                per_species[s] = per_species.get(s, 0) + 1
            if max_crops and len(records) >= max_crops:
                break
        if max_crops and len(records) >= max_crops:
            break

    # ---- Decision 3.59: empty-hole crops -----------------------------------
    # Without these, the adaptation set contains only "hole -> colony" and the
    # model learns that a hole is ALWAYS a colony. But at generation time the
    # erase regions (decision 3.25) must come back as PLAIN AGAR: they are holes
    # where a real colony was removed and no synthetic colony was placed. A model
    # that has never seen an empty hole will fill them -- producing an UNLABELLED
    # object, which is exactly the false negative erasing was meant to prevent.
    #
    # So a fraction of the training set is: mask a colony-free patch of real
    # agar, target = that same plain agar, caption = "no colonies".
    rng_e = np.random.default_rng(0)
    n_empty = int(round(len(records) * empty_ratio))
    made = 0
    guard = 0
    while made < n_empty and guard < n_empty * 60:
        guard += 1
        r = records[int(rng_e.integers(0, max(len(records), 1)))]
        img = cv2.imread(str(out_dir / "images" / f"{r['name']}.png"))
        src_mask = cv2.imread(str(out_dir / "masks" / f"{r['name']}.png"), 0)
        if img is None:
            continue
        # a disk the size of a real colony, on a spot with no colony in it
        rad = int(rng_e.integers(12, 90))
        cx = int(rng_e.integers(rad, tile - rad))
        cy = int(rng_e.integers(rad, tile - rad))
        probe = np.zeros((tile, tile), np.uint8)
        cv2.circle(probe, (cx, cy), int(rad * 1.6), 255, -1)
        if (src_mask[probe > 0] > 0).any():
            continue                       # overlaps a real colony, try again
        m = np.zeros((tile, tile), np.uint8)
        cv2.circle(m, (cx, cy), rad, 255, -1)
        name = f"empty_{made:04d}"
        cv2.imwrite(str(out_dir / "images" / f"{name}.png"), img)
        cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), m)
        records.append({"name": name, "source_plate": r["source_plate"],
                        "caption": "clean agar plate surface, no colonies, "
                                   "even studio lighting, top-down view",
                        "kind": "empty", "n_colonies": 0,
                        "mask_fraction": round(float((m > 0).mean()), 5)})
        made += 1

    print()
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    return {"n_plates": len(stems), "n_crops": len(records),
            "n_empty": sum(1 for r in records if r.get("kind") == "empty"),
            "skipped_oversized_plates": skipped_big,
            "per_species": per_species}


def crops(args):
    classes = Path(args.classes).read_text(encoding="utf-8").split()
    stats = extract_crops(Path(args.list), Path(args.labels), Path(args.images),
                          classes, args.tile, Path(args.out), args.max_crops,
                          args.empty_ratio)

    print(f"[crops] level={args.level}  {stats['n_plates']} plates -> "
          f"{stats['n_crops']} crops of {args.tile}px "
          f"({stats['n_crops']-stats['n_empty']} colony + {stats['n_empty']} empty)")
    print(f"        source list      : {args.list}")
    if stats["skipped_oversized_plates"]:
        print(f"  [!] {stats['skipped_oversized_plates']} plates skipped: a colony "
              f"was larger than the {args.tile}px tile.")
    print("        colonies per species in the training set:")
    tot = sum(stats["per_species"].values()) or 1
    for s in classes:
        n = stats["per_species"].get(s, 0)
        flag = "   <-- NONE" if n == 0 else ""
        print(f"          {s:<14} {n:>6}  ({100*n/tot:4.1f}%){flag}")

    # Decision 3.57: a species absent from the adaptation set cannot be generated.
    missing = [s for s in classes if stats["per_species"].get(s, 0) == 0]
    if missing:
        print(f"  [!] These species have NO training crops: {missing}")
        print("      The adapted model cannot generate them. At low levels this "
              "is a real\n      risk: if C.albicans vanishes from train_10, the "
              "G10+S arm measures a\n      class-imbalance artefact, not a data-"
              "quantity effect (decision 1.5).")
    if stats["n_crops"] < 50:
        print(f"  [!] Only {stats['n_crops']} crops. LoRA works with few images, "
              f"but below ~50\n      the adaptation is unstable. Expected for the "
              f"demo package; check\n      again on the full dataset.")
    print(f"        -> {args.out}")


def train(args):
    """
    LoRA fine-tune. Needs a GPU; the crop extraction above does not, so the
    training set can be inspected before a single GPU-second is spent.
    """
    # ---- PRE-FLIGHT, before importing torch -------------------------------
    # These checks cost nothing and must fire on a machine without a GPU too.
    # (The same lesson as train.py's null-augment guard: a check placed after a
    # heavy import only runs where the import succeeds.)
    crops_dir = Path(args.crops)
    meta_path = crops_dir / "metadata.jsonl"
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found. Run `adapt.py crops` first.")
    meta = [json.loads(l) for l in
            meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not meta:
        sys.exit(f"ERROR: {meta_path} is empty. Run `crops` first.")

    # Decision 3.7: the level must be carried in the output name so a mismatched
    # LoRA can never be silently paired with the wrong level at generation time.
    out = Path(args.out)
    if f"_{args.level}" not in out.name:
        sys.exit(f"ERROR (decision 3.7): --out must carry the level, e.g. "
                 f"lora_{args.level}. Got '{out.name}'.")

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    print(f"[train] level={args.level}  {len(meta)} crops  {args.steps} steps  "
          f"rank={args.rank}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False).to(args.device)

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank,
                     target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                     lora_dropout=0.0, bias="none")
    unet = get_peft_model(pipe.unet, cfg)
    unet.print_trainable_parameters()
    unet.train()
    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad],
                            lr=args.lr)

    vae, txt, tok, sched = pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    vae.requires_grad_(False); txt.requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    # Decision 3.61: the per-step loss is UNINTERPRETABLE and must not be read
    # as a training curve.
    #
    # Every step samples a random timestep. A very noisy timestep makes noise
    # prediction easy (low loss); a nearly clean one makes it hard (high loss).
    # With batch=1 that variance swamps any learning signal -- the first run
    # printed 0.0006, 0.0147, 0.0016, 0.0241 in consecutive logs, which says
    # nothing at all. Two fixes:
    #   * a running mean over the last `log_every` steps, and
    #   * a validation loss on a FIXED set of crops at FIXED timesteps, so the
    #     only thing that changes between evaluations is the model.
    # The second is the one to actually watch.
    recent = []
    eval_idx = rng.integers(0, len(meta), min(16, len(meta)))
    eval_ts = torch.linspace(50, sched.config.num_train_timesteps - 50, 8).long()

    def fixed_eval():
        """Loss on fixed crops at fixed timesteps -- comparable across steps."""
        unet.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in eval_idx:
                r = meta[int(i)]
                im = cv2.imread(str(crops_dir / "images" / f"{r['name']}.png"))
                mk = cv2.imread(str(crops_dir / "masks" / f"{r['name']}.png"), 0)
                x = torch.from_numpy(
                    (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1)
                )[None].permute(0, 3, 1, 2).to(args.device)
                m = torch.from_numpy((mk > 0).astype(np.float32))[None, None].to(args.device)
                lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
                masked = vae.encode(x * (1 - m)).latent_dist.mean * vae.config.scaling_factor
                ids = tok([r["caption"]], padding="max_length",
                          max_length=tok.model_max_length, truncation=True,
                          return_tensors="pt").input_ids.to(args.device)
                emb = txt(ids)[0]
                g = torch.Generator(device=args.device).manual_seed(int(i))
                noise = torch.randn(lat.shape, generator=g, device=args.device)
                m_lat = torch.nn.functional.interpolate(m, size=lat.shape[-2:])
                for ts_v in eval_ts:
                    ts = ts_v[None].to(args.device)
                    noisy = sched.add_noise(lat, noise, ts)
                    pred = unet(torch.cat([noisy, m_lat, masked], 1), ts,
                                encoder_hidden_states=emb).sample
                    tot += torch.nn.functional.mse_loss(pred, noise).item(); n += 1
        unet.train()
        return tot / max(n, 1)

    base_eval = fixed_eval()
    print(f"  eval loss before training: {base_eval:.5f}  "
          f"(fixed crops, fixed timesteps -- THIS is the number to watch)")

    for step in range(args.steps):
        batch = [meta[i] for i in rng.integers(0, len(meta), args.batch)]
        imgs, masks, caps = [], [], []
        for r in batch:
            im = cv2.imread(str(crops_dir / "images" / f"{r['name']}.png"))
            mk = cv2.imread(str(crops_dir / "masks" / f"{r['name']}.png"), 0)
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1)
            masks.append((mk > 0).astype(np.float32))
            caps.append(r["caption"])

        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(args.device)
        m = torch.from_numpy(np.stack(masks))[:, None].to(args.device)

        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            masked = vae.encode(x * (1 - m)).latent_dist.sample() * vae.config.scaling_factor
            ids = tok(caps, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(args.device)
            emb = txt(ids)[0]

        noise = torch.randn_like(lat)
        ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],),
                           device=args.device).long()
        noisy = sched.add_noise(lat, noise, ts)
        m_lat = torch.nn.functional.interpolate(m, size=lat.shape[-2:])
        model_in = torch.cat([noisy, m_lat, masked], dim=1)

        pred = unet(model_in, ts, encoder_hidden_states=emb).sample
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward(); opt.step(); opt.zero_grad()

        recent.append(loss.item())
        if (step + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  step {step+1}/{args.steps}  train(mean {len(recent)}) "
                  f"{np.mean(recent):.4f}  {el/(step+1):.2f} s/step  "
                  f"eta {(args.steps-step-1)*el/(step+1)/60:.1f} min")
            recent = []

        # Decision 3.62: checkpoint periodically. 1500 steps on 168 crops is a
        # GUESS. Whether it is under- or over-trained can only be seen by
        # generating from it, and without checkpoints that means retraining from
        # scratch for every answer. Saving lets one training run answer "how many
        # steps" instead of three.
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
            ck = out / f"step_{step+1:05d}"
            ck.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(str(ck))
            ev = fixed_eval()
            print(f"    checkpoint {ck.name}  eval {ev:.5f}  "
                  f"(start {base_eval:.5f})")

    final_eval = fixed_eval()
    out.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out))
    hours = (time.perf_counter() - t0) / 3600
    (out / "adapt_metrics.json").write_text(json.dumps({
        "level": args.level, "n_crops": len(meta), "steps": args.steps,
        "rank": args.rank, "lr": args.lr, "batch": args.batch,
        "base_model": args.model,
        "eval_loss_start": round(base_eval, 6),
        "eval_loss_end": round(final_eval, 6),
        "source_list": str(args.source_list) if args.source_list else None,
        "hours": round(hours, 3),
        "peak_vram_gb": (round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
                         if torch.cuda.is_available() else None),
    }, indent=2), encoding="utf-8")

    print(f"\n[train] done in {hours*60:.1f} min -> {out}")
    print(f"        eval loss {base_eval:.5f} -> {final_eval:.5f} "
          f"({100*(final_eval-base_eval)/base_eval:+.1f}%)")
    if final_eval >= base_eval:
        print("  [!] The eval loss did not fall. Either the adaptation is not "
              "learning\n      (check lr / rank) or the base model already fits "
              "these crops.\n      Generate from it before trusting it either way.")
    print(f"        scripts/budget.py --lora-hours {hours:.2f} --lora-count 4")
    print(f"        (4 adaptations are needed, one per level -- decision 3.2)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crops", help="extract the training set (no GPU)")
    # decision 3.5: never a default level
    p.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100])
    p.add_argument("--list", required=True, help="THIS level's train list only")
    p.add_argument("--labels", default="data/processed/labels")
    p.add_argument("--images", default="data/processed/images")
    p.add_argument("--classes", default="data/processed/classes.txt")
    p.add_argument("--tile", type=int, default=T.DEFAULT_TILE)
    p.add_argument("--max-crops", type=int, default=None)
    p.add_argument("--empty-ratio", type=float, default=0.20,
                   help="share of empty-hole crops (decision 3.59). 0 disables "
                        "them, but then the model learns 'a hole is always a "
                        "colony' and will invent colonies in the erase regions.")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=crops)

    p = sub.add_parser("train", help="LoRA fine-tune (needs a GPU)")
    p.add_argument("--level", type=int, required=True, choices=[10, 25, 50, 100])
    p.add_argument("--crops", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500,
                   help="save an intermediate LoRA every N steps (decision 3.62). "
                        "0 disables. Lets one run answer 'how many steps'.")
    p.add_argument("--source-list", default=None,
                   help="recorded in adapt_metrics.json as the leakage audit trail")
    p.add_argument("--out", required=True, help="must contain the level, e.g. lora_25")
    p.set_defaults(fn=train)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
