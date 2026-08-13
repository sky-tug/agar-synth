#!/usr/bin/env bash
# =============================================================================
# AGAR-SYNTH ortam kurulumu  --  TEK KOMUT
#
#   bash scripts/kurulum.sh
#
# Ne yapar:
#   1. GPU'yu kontrol eder
#   2. conda ortami 'agar' (Python 3.11) kurar  -- varsa dokunmaz
#   3. PyTorch (CUDA) + Ultralytics + olcum bagimliliklarini kurar
#   4. GPU'yu torch icinden dogrular, VRAM'e gore batch onerir
#   5. Olcum kodunun sanity testini kosturur (33 test)
#
# Idempotent: iki kez calistirmak zarar vermez.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
KOK="$PWD"

mavi(){ printf "\033[1;34m%s\033[0m\n" "$*"; }
yesil(){ printf "\033[1;32m%s\033[0m\n" "$*"; }
sari(){ printf "\033[1;33m%s\033[0m\n" "$*"; }
kirmizi(){ printf "\033[1;31m%s\033[0m\n" "$*"; }
baslik(){ echo; mavi "══════ $* ══════"; }

ORTAM=agar
PY=3.11

# ---------------------------------------------------------------- 1. GPU ----
baslik "1/5  GPU kontrolu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
else
  kirmizi "nvidia-smi bulunamadi. NVIDIA surucusu kurulu mu?"
  sari "Kuruluma devam ediliyor ama egitim CPU'da kosar (Faz 2 icin anlamsiz)."
  VRAM_MB=0
fi

# -------------------------------------------------------------- 2. ortam ----
baslik "2/5  python ortami"

# DIKKAT: Zaten etkin bir venv varsa conda activate onu EZEMEZ; paketler
# venv'e kurulur ve sonda "conda activate agar" demek yaniltici olur
# (o ortam bos kalir). O yuzden once etkin venv'i tespit ediyoruz.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  sari "Etkin bir venv var: $VIRTUAL_ENV"
  sari "Paketler BU ortama kurulacak (conda 'agar' kullanilmayacak)."
  ETKINLESTIRME="source ${VIRTUAL_ENV}/bin/activate"
else
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && . "$c/etc/profile.d/conda.sh" && break
  done
  if ! command -v conda >/dev/null 2>&1; then
    kirmizi "conda bulunamadi. anaconda3 kurulu ama PATH'te degil olabilir."
    echo "  Deneyin:  source ~/anaconda3/etc/profile.d/conda.sh  &&  bash scripts/kurulum.sh"
    exit 1
  fi
  if conda env list | grep -qE "^${ORTAM}\s"; then
    yesil "conda ortami '$ORTAM' zaten var"
  else
    echo "olusturuluyor: $ORTAM (Python $PY)"
    conda create -y -n "$ORTAM" "python=$PY" || exit 1
  fi
  conda activate "$ORTAM" || exit 1
  ETKINLESTIRME="conda activate $ORTAM"
fi
echo "aktif python: $(which python)  ($(python -V 2>&1))"

# ------------------------------------------------------------- 3. paket ----
baslik "3/5  paketler"
python -m pip install --upgrade pip -q

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  yesil "CUDA'li torch zaten kurulu, atlaniyor"
else
  echo "PyTorch (CUDA 12.4) kuruluyor -- birkac dakika surebilir..."
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    || { sari "cu124 basarisiz, varsayilan tekerlek deneniyor"; python -m pip install torch torchvision; }
fi

echo "diger paketler..."
python -m pip install -q -r requirements.txt || exit 1
yesil "paketler tamam"

# ----------------------------------------------------------- 4. dogrulama ----
baslik "4/5  GPU dogrulamasi"
python - <<'PY'
import torch, platform
print(f"  torch      : {torch.__version__}")
print(f"  CUDA derl. : {torch.version.cuda}")
print(f"  CUDA var mi: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024**3
    print(f"  kart       : {p.name}")
    print(f"  VRAM       : {gb:.1f} GB")
    # imgsz=1280, YOLO nano, AMP acik icin kaba batch onerisi
    oneri = 2 if gb < 7 else 4 if gb < 10 else 8 if gb < 14 else 16
    print()
    print(f"  >>> imgsz=1280 icin onerilen baslangic batch: {oneri}")
    print(f"      Yetmezse dusur. imgsz'yi DUSURME -- koloniler cok kucuk.")
else:
    print("  ! GPU yok. Egitim icin Colab kullan.")
PY

python -c "import ultralytics; ultralytics.checks()" 2>&1 | sed 's/^/  /'

# ---------------------------------------------------------------- 5. test ----
baslik "5/5  olcum kodu sanity testi"
python src/eval/test_metrics.py
DURUM=$?

echo
if [ $DURUM -eq 0 ]; then
  yesil "══════ KURULUM TAMAM ══════"
  echo
  sari "Ortami her acilista SU KOMUTLA etkinlestir:"
  echo "    $ETKINLESTIRME"
  echo "  (paketler buraya kuruldu -- baska bir ortamda torch bulunmaz)"
  echo
  echo "Siradaki adim -- demo veriyle duman testi:"
  echo "    python src/convert.py --src data/AGAR_representative --out data/processed"
  echo "    python src/make_splits.py --data data/processed --seed 42"
  echo "    python scripts/train.py --config configs/base.yaml --level 100 --seed 0 \\"
  echo "        --epochs 10 --batch 2 --name duman_testi --duman"
  echo
  echo "Sonra butce tahmini:"
  echo "    python scripts/butce.py --olcum runs/duman_testi/olcum.json --tam-veri 8000 --xai"
else
  kirmizi "══════ SANITY TESTI BASARISIZ ══════"
  echo "Yukaridaki HATA satirlarina bak. Bu gecmeden egitime baslama."
  exit 1
fi
