#!/usr/bin/env python3
"""
One screen that answers "where is everything".                         (3.102)

Read-only. Touches nothing, decides nothing, and is safe to run at any moment
including while training and generation are going.

    python scripts/status.py

Until now the answer was spread over six places -- runs/, results/,
logs/queue_status.tsv, logs/gen_status.tsv, the synth_g* directories and
nvidia-smi -- and reconstructing it by hand took longer than reading it. That
is a bad property for a project meant to be left running for days.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
GEN = ROOT / "src" / "generate"
LISTS = ROOT / "data" / "processed" / "lists"
LOGS = ROOT / "logs"
QUEUE = ROOT / "queue.txt"

# level: plates needed -- mirrors gen_queue.sh
GEN_TARGETS = [(100, 2987), (10, 2688), (25, 4481), (50, 1494)]

# Arms already measured before the queue existed, so that the totals below
# describe Phase 6 and not just what happens to be in queue.txt.
HISTORICAL = ["G100", "G50", "G25", "G10"]

# A run whose results.csv has not moved in this long is not running.
STALE_SEC = 30 * 60

W = 68


def rule(ch="=") -> str:
    return ch * W


def bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "-" * width
    n = min(width, int(width * done / total))
    return "#" * n + "-" * (width - n)


def gpu_line() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return ["GPU    nvidia-smi bulunamadi"]
    out = []
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,"
             "memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        out.append(f"GPU    {q}")
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        out.append(f"       hesaplama sureci: {p if p else 'YOK -- kart bos'}")
    except Exception as e:                                  # noqa: BLE001
        out.append(f"GPU    okunamadi ({e})")
    return out


def run_state(name: str) -> tuple[str, str]:
    """(state, detail) for one run directory."""
    d = RUNS / name
    if not d.is_dir():
        return "bekliyor", ""
    if (d / "eval_val" / "summary.json").exists():
        try:
            s = json.loads((d / "eval_val" / "summary.json").read_text())
            return "bitti", f"mAP50-95 {s.get('mAP50-95', '?')}"
        except Exception:                                   # noqa: BLE001
            return "bitti", ""
    if (d / "run_metrics.json").exists():
        return "olculuyor", "egitim tamam, degerlendirme bekliyor"
    # Training in flight: results.csv grows one line per epoch. But a run
    # directory that was abandoned (a cancelled run, a crash) also has a
    # results.csv and no run_metrics.json, and would read as "training" for
    # ever -- G100_s1_iptal_1730 did exactly that. So require the file to have
    # been touched recently: an epoch here takes under two minutes even on the
    # largest arm, so nothing older than half an hour is still running.
    csv = d / "results.csv"
    if csv.exists():
        try:
            n = max(sum(1 for _ in csv.open(encoding="utf-8")) - 1, 0)
            age = time.time() - csv.stat().st_mtime
            if age > STALE_SEC:
                return "yarim kalmis", f"epoch {n}, {age/3600:.0f} saat dokunulmadi"
            return "egitiliyor", f"epoch {n}"
        except OSError:
            return "egitiliyor", ""
    return "basladi", ""


def read_queue() -> list[tuple[str, str]]:
    """[(run name, train list or '-')] in queue order."""
    if not QUEUE.exists():
        return []
    rows = []
    for line in QUEUE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) >= 3:
            rows.append((f[0], f[4] if len(f) >= 5 else "-"))
    return rows


def arm_of(name: str) -> str:
    """Group a run name into its arm: B_G25_s1 -> B_G25."""
    return name.rsplit("_s", 1)[0] if "_s" in name else name


def main() -> int:
    print(rule())
    print(f"AGAR-SYNTH DURUM{' ' * (W - 16 - 20)}{datetime.now():%d.%m.%Y %H:%M}")
    print(rule())
    for l in gpu_line():
        print(l)

    # --- what is running right now -----------------------------------------
    active, stale = [], []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir()):
            if not d.is_dir() or d.name.startswith("smoke"):
                continue
            st, detail = run_state(d.name)
            if st in ("egitiliyor", "olculuyor"):
                active.append(f"{d.name}  {st}  {detail}")
            elif st == "yarim kalmis":
                stale.append(f"{d.name}  {detail}")
    print(f"\nSU AN   {active[0] if active else 'egitim yok (uretim olabilir)'}")
    for a in active[1:]:
        print(f"        {a}")
    for st in stale:
        print(f"        (yarim kalmis, kuyrukta degil: {st})")

    # --- generation ---------------------------------------------------------
    print(f"\n{rule('-')}\nURETIM (sentetik plaka)")
    gen_done = gen_want = 0
    for lvl, target in GEN_TARGETS:
        d = GEN / f"synth_g{lvl}"
        have = len(list((d / "images").glob("*.jpg"))) if (d / "images").is_dir() else 0
        gen_done += min(have, target)
        gen_want += target
        if have >= target:
            note = "TAMAM"
        elif have == 0:
            note = "bekliyor"
        else:
            note = f"%{100 * have / target:.0f}"
        print(f"  level {lvl:>3}  {bar(have, target)}  {have:>5}/{target:<5} {note}")
    print(f"  {'':>9}  {'':<20}  {gen_done:>5}/{gen_want:<5} "
          f"toplam %{100 * gen_done / max(gen_want, 1):.0f}")

    # --- arm lists ----------------------------------------------------------
    print(f"\n{rule('-')}\nKOL LISTELERI (sentetik kollarin egitim kumesi)")
    any_list = False
    for p in sorted(LISTS.glob("arm_*.txt")):
        any_list = True
        side = p.with_suffix(p.suffix + ".json")
        if side.exists():
            s = json.loads(side.read_text())
            print(f"  {p.name:<26} hazir   {s['n_real']} gercek + "
                  f"{s['n_synth']} sentetik")
        else:
            print(f"  {p.name:<26} hazir   (kenar dosyasi yok)")
    if not any_list:
        print("  henuz yok -- uretim bitince gen_queue.sh kendisi kuruyor")

    # --- runs by arm --------------------------------------------------------
    print(f"\n{rule('-')}\nKOSULAR")
    print(f"  {'kol':<22}{'bitti':>7}{'suruyor':>9}{'bekliyor':>10}   not")

    queue = read_queue()
    order, groups = [], {}
    for name, tl in queue:
        a = arm_of(name)
        if a not in groups:
            groups[a] = []
            order.append(a)
        groups[a].append((name, tl))

    # historical arms first -- they are finished and part of the total
    hist_done = 0
    for a in HISTORICAL:
        names = [d.name for d in RUNS.iterdir()
                 if d.is_dir() and arm_of(d.name) == a] if RUNS.is_dir() else []
        done = sum(1 for n in names if run_state(n)[0] == "bitti")
        hist_done += done
        if done:
            print(f"  {a:<22}{done:>7}{'-':>9}{'-':>10}   olculdu")

    tot_done = hist_done
    tot_all = hist_done
    for a in order:
        rows = groups[a]
        done = busy = wait = 0
        note = ""
        for name, tl in rows:
            st = run_state(name)[0]
            if st == "bitti":
                done += 1
            elif st == "bekliyor":
                wait += 1
                if tl != "-" and not (LISTS / tl).exists():
                    note = "uretim bekliyor"
            else:
                busy += 1
        if not note and wait and all(tl != "-" for _, tl in rows):
            note = "liste hazir"
        tot_done += done
        tot_all += len(rows)
        print(f"  {a:<22}{f'{done}/{len(rows)}':>7}{busy or '-':>9}"
              f"{wait or '-':>10}   {note}")

    print(f"  {rule('-')[:W - 2]}")
    print(f"  {'FAZ 6 TOPLAM':<22}{f'{tot_done}/{tot_all}':>7}")

    # --- problems -----------------------------------------------------------
    print(f"\n{rule('-')}\nSORUNLAR")
    # Only what is STILL wrong. A transient failure followed by a success is
    # the retry working as designed, and listing it here would have someone
    # worrying about a problem that fixed itself hours ago. Keep the LAST
    # status per subject and report it only if that one is bad.
    last: dict[str, tuple[str, str]] = {}
    for f in (LOGS / "queue_status.tsv", LOGS / "gen_status.tsv"):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                last[parts[1]] = (parts[0], parts[2])

    BAD = ("fail", "missing")
    bad = [f"  {t[:16]}  {subj:<18} {st}"
           for subj, (t, st) in sorted(last.items())
           if any(b in st.lower() for b in BAD)]
    deferred = [subj for subj, (_, st) in last.items() if "deferred" in st]

    if bad:
        for b in bad:
            print(b)
        print("  (ilgili log: logs/<kosu adi>.log  ya da  logs/gen_<level>.log)")
    else:
        print("  yok -- cozulmemis hata yok")
    if deferred:
        print(f"  ({len(deferred)} kosu sirasini bekliyor, bu normal)")

    print(rule())
    return 0


if __name__ == "__main__":
    sys.exit(main())
