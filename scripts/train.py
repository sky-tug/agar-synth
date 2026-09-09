#!/usr/bin/env python3
"""
Training wrapper -- the SAME protocol on every run, and TIME/VRAM are logged.

Why not `yolo train` directly:
  1. For outline section 6 (computational cost) the time, GPU-hours and peak
     VRAM must be on record FROM THE FIRST run onwards. They cannot be measured
     retroactively.
  2. The augmentation policy must come from the configuration file, not from the
     command line. Otherwise the question "what was turned on in which run"
     stays unanswered.
  3. Ultralytics' defaults change from release to release. The config here
     writes out ALL augmentation fields EXPLICITLY -- no implicit default is
     left behind.
  4. When the protocol is frozen in Phase 5, what gets frozen is this file + the
     config.

Usage:
    # smoke test (with the demo data: is there a GPU / does the pipeline run)
    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\
        --epochs 10 --name smoke_test --smoke

    # G100 full training, single seed -- the main job of Phase 2
    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\
        --name G100_s0

    # classic arm
    python scripts/train.py --config configs/base.yaml \\
        --overlay configs/aug_c_copypaste.yaml --level 25 --seed 0 --name BC_G25_s0

Output: runs/<name>/run_metrics.json  ->  scripts/budget.py reads this
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Decision 3.40: fields written as `null` in the config used to be filtered out
# SILENTLY and Ultralytics would then use its own default -- exactly the thing
# decision 2.13 ("implicit default = irreproducibility") forbids. From now on
# only the fields below may be null (in Ultralytics None means "off"); if any
# other field is null the program STOPS.
AUGMENT_NULL_ALLOWED = {"auto_augment"}


# ---------------------------------------------------------------------------

def deep_merge(a: dict, b: dict) -> dict:
    """Overlay b on top of a (including nested dicts)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if (
            isinstance(v, dict) and isinstance(out.get(k), dict)) else v
    return out


def gpu_info() -> dict:
    info = {"cuda": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:
        info["error"] = str(e)
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception:
        pass
    return info


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True, timeout=10).strip()
    except Exception:
        return "unknown"


def list_path(data_root: Path, level: int) -> Path:
    """
    The absolute-path list that Ultralytics will read.
    level 100 -> train.txt, the others -> train_<level>.txt
    """
    name = "train" if level == 100 else f"train_{level}"
    p = data_root / "lists" / f"{name}.txt"
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist.\n"
                 f"  first run: python src/make_splits.py --data {data_root}")
    return p


def validate_list(p: Path, data_root: Path) -> int:
    """
    Can a label be found NEXT TO each path in the list?
    Ultralytics looks for the label by replacing the last '/images/' segment of
    the path with '/labels/'. If the list points at the raw AGAR folder the
    label is NOT FOUND and the model silently learns 'there are no objects' --
    this is the most insidious failure of all.
    """
    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        sys.exit(f"ERROR: {p} is empty. (In the demo package val/test come out "
                 f"empty -- the full dataset is required.)")
    # Previously only the first 50 lines were checked. On the full dataset train
    # has ~8000 images -> 0.6% of it was being checked. Path.exists() takes
    # microseconds; 8000 of them do not add up to a second. There is no reason
    # to keep the limit (decision 3.39).
    missing = 0
    missing_examples = []
    for s in lines:
        ip = Path(s)
        if f"{'/'}images{'/'}" not in str(ip):
            sys.exit(
                f"ERROR: the list points at the raw data path:\n    {ip}\n"
                f"  Ultralytics finds the label via '/images/' -> '/labels/'.\n"
                f"  The lists must point below {data_root/'images'}.\n"
                f"  Fix: write_paths() inside src/make_splits.py")
        lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists():
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(lp))
    if missing:
        sys.exit(f"ERROR: {missing} out of {len(lines)} images have no label file.\n"
                 + "\n".join(f"    missing: {e}" for e in missing_examples))
    return len(lines)


def write_dataset_yaml(target: Path, data_root: Path, train_list: Path,
                       val_list: Path, names: list[str]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(data_root),
        "train": str(train_list),
        "val": str(val_list),
        "test": str(data_root / "lists" / "test.txt"),
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    target.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    return target


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overlay", help="augmentation arm (e.g. aug_b_classic.yaml)")
    ap.add_argument("--level", type=int, default=100,
                    choices=[100, 50, 25, 10], help="real data level (%%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True, help="run name -> runs/<name>")
    ap.add_argument("--epochs", type=int, help="override the config (for the smoke test)")
    ap.add_argument("--imgsz", type=int, help="override the config")
    ap.add_argument("--batch", type=int, help="override the config (according to VRAM)")
    ap.add_argument("--workers", type=int,
                    help="override the config (according to SYSTEM RAM, not VRAM). "
                         "Each worker decodes a full-size 2048x2048 plate before "
                         "resizing, so on a 16 GB machine 8 workers can exhaust RAM.")
    ap.add_argument("--patience", type=int,
                    help="override the config. Give a high value in the "
                         "memorisation test (so that early stopping does not "
                         "kick in while mAP is stuck at 0)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: W&B off, the result is marked as 'smoke' "
                         "in run_metrics.json")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run (runs/<name>/weights/last.pt). "
                         "Use THIS one when Colab drops -- if you call "
                         "YOLO(...).train(resume=True) directly, run_metrics.json "
                         "IS NOT WRITTEN (time/VRAM/commit are lost, decision 3.42)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not start training, only do the preparation and the checks")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.overlay:
        cfg = deep_merge(cfg, yaml.safe_load(
            Path(args.overlay).read_text(encoding="utf-8")))

    t = cfg["train"]
    if args.epochs:
        t["epochs"] = args.epochs
    if args.imgsz:
        t["imgsz"] = args.imgsz
    if args.batch:
        t["batch"] = args.batch
    if args.workers is not None:
        t["workers"] = args.workers
    if args.patience is not None:
        t["patience"] = args.patience

    # --- augment: null check (decision 3.40) --------------------------------
    # PRE-FLIGHT: must run with --dry-run as well, it has to be caught before
    # training starts.
    aug = dict(cfg["augment"])
    disallowed = [k for k, v in aug.items() if v is None and k not in AUGMENT_NULL_ALLOWED]
    if disallowed:
        sys.exit(f"ERROR (decision 2.13 / 3.40): these augment fields are null: {disallowed}\n"
                 "  If you write null, Ultralytics uses its own DEFAULT and the\n"
                 "  augmentation policy collapses SILENTLY "
                 "(e.g. mosaic: null -> mosaic=1.0).\n"
                 "  If you want it off, write 0.0. The ones that may be null: "
                 f"{sorted(AUGMENT_NULL_ALLOWED)}")

    data_root = (ROOT / cfg["data"]["root"]).resolve()
    names = cfg["data"]["names"]

    train_list = list_path(data_root, args.level)
    n_train = validate_list(train_list, data_root)

    val_list = data_root / "lists" / "val.txt"
    val_empty = (not val_list.exists()
                 or not val_list.read_text(encoding="utf-8").strip())
    if val_empty and args.smoke:
        # In the demo package (10 images, 7 plates) the val set comes out empty.
        # The smoke test is not run to produce results, but to see THAT THE
        # PIPELINE WORKS; train is used as val. This measurement is NEVER
        # reported.
        yellow = "\033[1;33m%s\033[0m"
        print(yellow % "! val set is empty -- using val=train for the smoke test.")
        print(yellow % "  The mAP value of this run is MEANINGLESS (the model is tested on its own training data).")
        print(yellow % "  The only purpose is: are the labels found, is the VRAM enough, how many seconds is an epoch.")
        val_list = train_list
        n_val = n_train
    else:
        n_val = validate_list(val_list, data_root)

    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_yaml = write_dataset_yaml(out_dir / "dataset.yaml", data_root,
                                 train_list, val_list, names)

    gpu = gpu_info()
    print("=" * 62)
    print(f"RUN         : {args.name}")
    print(f"level       : G{args.level}   seed {args.seed}")
    print(f"train / val : {n_train} / {n_val} images")
    print(f"imgsz       : {t['imgsz']}   batch {t['batch']}   epochs {t['epochs']}")
    print(f"augment     : {args.overlay or 'base (minimum: flip + HSV)'}"
          f"   [{len(aug)} fields from the config]")
    log_on = bool((cfg.get("logging") or {}).get("wandb")) and not args.smoke
    print(f"W&B         : {(cfg.get('logging') or {}).get('project') if log_on else 'OFF'}")
    print(f"GPU         : {gpu.get('gpu_name', 'NONE -- CPU')}"
          f"{'  ' + str(gpu.get('vram_gb')) + ' GB' if gpu.get('cuda') else ''}")
    print("=" * 62)

    if not gpu.get("cuda"):
        print("! WARNING: CUDA is not visible. Training on CPU is meaningless for Phase 2.")
    if args.dry_run:
        print("\n--dry-run: preparation done, training was not started.")
        print(f"dataset yaml -> {ds_yaml}")
        return

    # ---------------- W&B (decision 3.41) ----------------
    # The `logging` block in base.yaml used to be READ NOWHERE and the help text
    # of --smoke that says "W&B off" had no counterpart in the code. The moment
    # you run wandb login, Ultralytics detects it by itself and logs to ITS OWN
    # default project name -- and the smoke tests would get mixed in there.
    log_cfg = cfg.get("logging") or {}
    wandb_on = bool(log_cfg.get("wandb")) and not args.smoke
    os.environ["WANDB_MODE"] = "online" if wandb_on else "disabled"
    if wandb_on and log_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(log_cfg["project"])
    # ---------------- Training ----------------
    import torch
    from ultralytics import YOLO

    if gpu.get("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # --- Decision 3.87: segment accounting ----------------------------------
    # run_metrics.json is written ONLY when a run finishes. On a spot/preemptible
    # card an interrupted segment therefore leaves NO time record at all, and the
    # --resume segment then OVERWRITES the file with its own partial numbers.
    # Measured 2 Sep: a run that really took ~4.3 min reported total_min 2.32 and
    # epochs_actual 4 out of 6. Section 6 of the paper (computational cost) is
    # written from this file, so a run that survives a preemption must not
    # silently lose half of its own cost.
    #
    # segments.jsonl is append-only and fsync'd after EVERY epoch: it is the only
    # part of the accounting that survives kill -9.
    seg_path = out_dir / "segments.jsonl"

    def read_segments() -> list[dict]:
        """Last line per segment id = that segment's final state."""
        if not seg_path.exists():
            return []
        by_id: dict[int, dict] = {}
        for line in seg_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue        # a half-written last line after kill -9
            by_id[rec["seg"]] = rec
        return [by_id[k] for k in sorted(by_id)]

    prev_segments = read_segments()
    seg_id = (prev_segments[-1]["seg"] + 1) if prev_segments else 0
    accounting_complete = True
    if args.resume and not prev_segments:
        accounting_complete = False
        print("\033[1;33m%s\033[0m" % (
            "! WARNING: --resume, but segments.jsonl is missing.\n"
            "  The interrupted segment left no time record; total_sec covers THIS\n"
            "  segment only. run_metrics.json is marked accounting_complete=false."))
    elif prev_segments:
        print(f"SEGMENTS   : {len(prev_segments)} earlier segment(s), "
              f"{sum(s['elapsed_sec'] for s in prev_segments) / 60:.1f} min carried over")

    epoch_times = []
    last = {"t": time.perf_counter()}

    def on_epoch_end(trainer):
        now = time.perf_counter()
        epoch_times.append(round(now - last["t"], 3))
        last["t"] = now
        rec = {
            "seg": seg_id,
            "resumed": bool(args.resume),
            "epochs": len(epoch_times),
            "elapsed_sec": round(now - t0, 3),
            "peak_vram_gb": (round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
                             if gpu.get("cuda") else None),
            "git_commit": commit,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        with seg_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    if args.resume:
        last_pt = out_dir / "weights" / "last.pt"
        if not last_pt.exists():
            sys.exit(f"ERROR: --resume was given but {last_pt} does not exist.")
        print(f"RESUME     : {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO(cfg["model"]["weights"])
    model.add_callback("on_fit_epoch_end", on_epoch_end)

    kwargs = dict(
        data=str(ds_yaml),
        epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
        patience=t["patience"], optimizer=t["optimizer"],
        lr0=t["lr0"], lrf=t["lrf"], momentum=t["momentum"],
        weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"],
        cos_lr=t["cos_lr"], deterministic=t["deterministic"],
        workers=t["workers"], amp=t["amp"], val=t["val"], plots=t["plots"],
        # It must hold for the validation step inside training too: there are up
        # to 125 colonies on a single plate. If it is written in the config but
        # not passed through here, the Ultralytics default (300) is used
        # silently.
        max_det=t.get("max_det", 300),
        seed=args.seed, device=args.device,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True,
        **aug,
    )

    started_at = datetime.now(timezone.utc)
    commit = git_commit()       # resolved once; the callback must not fork git per epoch
    t0 = time.perf_counter()
    result = model.train(resume=True) if args.resume else model.train(**kwargs)
    total = time.perf_counter() - t0

    peak_vram = (round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
                 if gpu.get("cuda") else None)

    # ---------------- Metrics record ----------------
    # --- cumulative accounting over all segments (decision 3.87) ------------
    segments = read_segments()
    cum_sec = sum(s["elapsed_sec"] for s in prev_segments) + total
    prev_vram = [s["peak_vram_gb"] for s in prev_segments
                 if s.get("peak_vram_gb") is not None]
    cum_peak_vram = max(prev_vram + ([peak_vram] if peak_vram is not None else [])) \
        if (prev_vram or peak_vram is not None) else None

    # Epochs: results.csv is the only source that counts REAL epochs. The
    # on_fit_epoch_end callback fires for the final validation pass too, so it
    # over-counts by exactly one PER SEGMENT -- harmless at one segment (measured
    # 2 Sep: 6 planned, 7 counted), but it multiplies once segments are summed.
    # The old number is kept next to the measurement (principle 3).
    callback_epochs = sum(s["epochs"] for s in prev_segments) + len(epoch_times)
    try:
        real_epochs = max(
            sum(1 for _ in (out_dir / "results.csv").open(encoding="utf-8")) - 1, 0)
    except OSError:
        real_epochs = 0
    actual_epochs = real_epochs or callback_epochs

    metrics = {
        "run": args.name,
        "smoke_test": bool(args.smoke),
        "resumed": bool(args.resume),   # the measurements are partial -- decision 3.42
        "resume_count": max(len(segments) - 1, 0),
        "accounting_complete": accounting_complete,
        "date_utc": started_at.isoformat(),
        "git_commit": commit,
        "level": args.level,
        "seed": args.seed,
        "overlay": args.overlay,
        "n_train": n_train,
        "n_val": n_val,
        "imgsz": t["imgsz"],
        "batch": t["batch"],
        "epochs_planned": t["epochs"],
        "epochs_actual": actual_epochs,              # from results.csv
        "epochs_callback_raw": callback_epochs,      # old count, +1 per segment
        # --- section 6 (computational cost) will be written from here ---
        # All four are CUMULATIVE across segments; the per-segment values sit
        # beside them so a resumed run stays auditable (decision 3.87).
        "total_sec": round(cum_sec, 2),
        "total_min": round(cum_sec / 60, 2),
        "sec_per_epoch": round(cum_sec / max(actual_epochs, 1), 2),
        "gpu_hours": round(cum_sec / 3600, 4),
        "peak_vram_gb": cum_peak_vram,
        "segment_sec": round(total, 2),
        "segment_peak_vram_gb": peak_vram,
        "segments": segments,
        "epoch_secs": epoch_times,                   # this segment only
        "images_per_sec": round(n_train * max(actual_epochs, 1) / cum_sec, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **gpu,
        },
        "augment": cfg["augment"],
    }
    try:
        metrics["ultralytics_metrics"] = {
            k: float(v) for k, v in result.results_dict.items()
            if isinstance(v, (int, float))
        }
    except Exception:
        pass

    (out_dir / "run_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"total time     : {metrics['total_min']:.1f} min "
          f"({metrics['gpu_hours']:.3f} GPU-hours)"
          f"{f'  [{len(segments)} segments]' if len(segments) > 1 else ''}")
    if len(segments) > 1:
        print(f"  this segment : {total / 60:.1f} min")
    print(f"per epoch      : {metrics['sec_per_epoch']:.1f} s "
          f"({actual_epochs} epochs ran)")
    print(f"peak VRAM      : {cum_peak_vram} GB"
          f"{f'  (this segment {peak_vram})' if len(segments) > 1 else ''}")
    if not accounting_complete:
        print("! accounting_complete = false -- an interrupted segment left no record")
    print(f"metrics -> {out_dir / 'run_metrics.json'}")
    print("=" * 62)
    print("\nNEXT:")
    print(f"  python src/eval/evaluate.py --weights runs/{args.name}/weights/best.pt "
          f"--split val  --out runs/{args.name}/eval --imgsz {t['imgsz']}")
    print(f"  python scripts/budget.py --metrics runs/{args.name}/run_metrics.json")


if __name__ == "__main__":
    main()
