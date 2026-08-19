#!/usr/bin/env bash
# =============================================================================
# AGAR-SYNTH environment setup  --  A SINGLE COMMAND
#
#   bash scripts/setup.sh
#
# What it does:
#   1. Checks the GPU
#   2. Creates the conda environment 'agar' (Python 3.11)  -- leaves it alone if it exists
#   3. Installs PyTorch (CUDA) + Ultralytics + the metrics dependencies
#   4. Verifies the GPU from inside torch, suggests a batch size based on VRAM
#   5. Runs both sanity suites as a GATE
#        src/eval/test_metrics.py       36 checks (42 with pycocotools)
#        src/generate/test_generate.py  61 checks
#      A non-zero exit here means: do not start training.
#
# Idempotent: running it twice does no harm.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"

blue(){ printf "\033[1;34m%s\033[0m\n" "$*"; }
green(){ printf "\033[1;32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[1;33m%s\033[0m\n" "$*"; }
red(){ printf "\033[1;31m%s\033[0m\n" "$*"; }
heading(){ echo; blue "====== $* ======"; }

ENV=agar
PY=3.11

# ---------------------------------------------------------------- 1. GPU ----
heading "1/5  GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
else
  red "nvidia-smi not found. Is the NVIDIA driver installed?"
  yellow "Setup continues but training will run on the CPU (pointless for Phase 2)."
  VRAM_MB=0
fi

# --------------------------------------------------------- 2. environment ----
heading "2/5  python environment"

# CAUTION: if a venv is already active, conda activate CANNOT override it; the
# packages get installed into the venv and saying "conda activate agar" at the
# end would be misleading (that environment would stay empty). That is why we
# detect the active venv first.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  yellow "There is an active venv: $VIRTUAL_ENV"
  yellow "The packages will be installed into THIS environment (conda 'agar' will not be used)."
  ACTIVATE="source ${VIRTUAL_ENV}/bin/activate"
else
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && . "$c/etc/profile.d/conda.sh" && break
  done
  if ! command -v conda >/dev/null 2>&1; then
    red "conda not found. anaconda3 may be installed but not on the PATH."
    echo "  Try:  source ~/anaconda3/etc/profile.d/conda.sh  &&  bash scripts/setup.sh"
    exit 1
  fi
  if conda env list | grep -qE "^${ENV}\s"; then
    green "conda environment '$ENV' already exists"
  else
    echo "creating: $ENV (Python $PY)"
    conda create -y -n "$ENV" "python=$PY" || exit 1
  fi
  conda activate "$ENV" || exit 1
  ACTIVATE="conda activate $ENV"
fi
echo "active python: $(which python)  ($(python -V 2>&1))"

# ------------------------------------------------------------ 3. packages ----
heading "3/5  packages"
python -m pip install --upgrade pip -q

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  green "torch with CUDA is already installed, skipping"
else
  # Decision 3.47: cu124 used to be installed, but all the measurements were made
  # with torch 2.13.0+cu130. A different CUDA build -> the durations are not
  # comparable, and the table of section 6 cannot be defended if it comes from
  # two machines. It can be changed with CUDA_CHANNEL.
  CUDA_CHANNEL="${CUDA_CHANNEL:-cu130}"
  echo "installing PyTorch ($CUDA_CHANNEL) -- this may take a few minutes..."
  python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_CHANNEL" \
    || { yellow "$CUDA_CHANNEL failed, trying cu124"
         python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
         || { yellow "that failed too, default wheel"; python -m pip install torch torchvision; }; }
fi

echo "other packages..."
python -m pip install -q -r requirements.txt || exit 1
green "packages done"

# -------------------------------------------------------- 4. verification ----
heading "4/5  GPU verification"
python - <<'PY'
import torch, platform
print(f"  torch      : {torch.__version__}")
print(f"  CUDA build : {torch.version.cuda}")
print(f"  CUDA avail.: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024**3
    print(f"  card       : {p.name}")
    print(f"  VRAM       : {gb:.1f} GB")
    # imgsz=1280, YOLO26 nano, AMP on. MEASURED in decision 2.19:
    # on an 8 GB card batch 8 -> peak VRAM 6.05 GB (80% of 7.6), that is the ceiling.
    # The old heuristic suggested 4 for 8 GB; it stayed below the measured value.
    suggested = 2 if gb < 5 else 4 if gb < 7 else 8 if gb < 12 else 16
    print()
    print(f"  >>> suggested starting batch for imgsz=1280: {suggested}")
    print(f"      Lower it if it does not fit. DO NOT LOWER imgsz -- the colonies are too small.")
else:
    print("  ! No GPU. Use Colab for training.")
PY

python -c "import ultralytics; ultralytics.checks()" 2>&1 | sed 's/^/  /'

# --------------------------------------------------------------- 5. test ----
heading "5/5  sanity tests"
echo "--- metrics code (src/eval/test_metrics.py)"
python src/eval/test_metrics.py
STATUS=$?
echo
echo "--- generation pipeline (src/generate/test_generate.py)"
python src/generate/test_generate.py
STATUS=$(( STATUS + $? ))

echo
if [ $STATUS -eq 0 ]; then
  green "====== SETUP COMPLETE ======"
  echo
  yellow "Activate the environment WITH THIS COMMAND every time you open a shell:"
  echo "    $ACTIVATE"
  echo "  (the packages were installed here -- torch will not be found in another environment)"
  echo
  echo "Next step -- smoke test with the demo data:"
  echo "    python src/convert.py --src data/AGAR_representative --out data/processed"
  echo "    python src/make_splits.py --data data/processed --seed 42"
  echo "    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\"
  echo "        --epochs 10 --batch 2 --name smoke_test --smoke"
  echo
  echo "Then the budget estimate:"
  echo "    python scripts/budget.py --metrics runs/smoke_test/run_metrics.json --full-size 8000 --xai"
else
  red "====== SANITY TESTS FAILED ======"
  echo "Look at the ERROR lines above. Do not start training before this passes."
  exit 1
fi
