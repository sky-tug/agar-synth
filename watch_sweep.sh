#!/usr/bin/env bash
# Tarama izleyici. Tek seferlik yazar; canli takip icin:
#     watch -n 15 bash ~/agar-synth/watch_sweep.sh
# Buradaki Ctrl+C GUVENLI -- sadece izlemeyi kapatir, taramayi oldurmez.
cd ~/agar-synth || exit 1

TARGET=100                       # her kontrol noktasi icin uretilecek plak
# en son yazilan sweep*.log dosyasi -- taramayi parca parca kosarsak
# (sweep.log, sweep6000.log ...) izleyici kendiliginden dogru dosyaya bakar
LOG=$(ls -t sweep*.log 2>/dev/null | head -1)
[ -z "$LOG" ] && LOG=sweep.log

printf '  TARAMA  %s\n' "$(date '+%H:%M:%S')"
printf '  ─────────────────────────────────────────────────────────\n'

# 1) LoRA gercekten takildi mi? (karar 3.63 -- bu satir yoksa taban model)
n_lora=$(grep -c '\[lora\]' "$LOG" 2>/dev/null)
if [ "${n_lora:-0}" -eq 0 ]; then
  printf '  LoRA      henuz yok (model yukleniyor ya da PROBLEM)\n'
else
  grep '\[lora\]' "$LOG" | sed 's/^ *//;s/^/  /'
fi
printf '\n'

# 2) Her nokta icin ilerleme
done_any=0
for TAG in ck3000 ck4500 ck6000; do
  d=src/generate/synth_$TAG/images
  n=0; [ -d "$d" ] && n=$(ls "$d" 2>/dev/null | wc -l)
  json=src/generate/species_check_$TAG.json

  if [ -f "$json" ]; then
    state='BITTI  species_check hazir'
  elif [ "$n" -ge "$TARGET" ]; then
    state='uretim bitti, species_check calisiyor'
  elif [ "$n" -gt 0 ]; then
    state='uretiliyor'
  else
    state='beklemede'
  fi

  filled=$(( n * 30 / TARGET )); [ "$filled" -gt 30 ] && filled=30
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  pad=$(printf '%*s' $(( 30 - filled )) '')
  printf '  %-8s [%s%s] %3d/%d  %s\n' "$TAG" "$bar" "$pad" "$n" "$TARGET" "$state"
  done_any=$(( done_any + n ))
done
printf '\n'

# 3) Hiz ve kalan sure -- OLCUM, tahmin degil: uretilmis plak / gecen sure
start=$(stat -c %Y "$LOG" 2>/dev/null)
first=$(ls -t src/generate/synth_ck3000/images 2>/dev/null | tail -1)
if [ "$done_any" -gt 2 ] && [ -n "$first" ]; then
  t0=$(stat -c %Y "src/generate/synth_ck3000/images/$first" 2>/dev/null)
  now=$(date +%s)
  el=$(( now - t0 ))
  if [ "$el" -gt 0 ]; then
    total=$(( TARGET * 3 ))
    left=$(( total - done_any ))
    printf '  hiz       %s sn/plak  ·  %d/%d plak  ·  kalan ~%d dk\n' \
      "$(awk "BEGIN{printf \"%.1f\", $el/$done_any}")" \
      "$done_any" "$total" \
      "$(awk "BEGIN{printf \"%d\", $el/$done_any*$left/60}")"
  fi
fi

# 4) GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU       %s C  ·  %%%s kullanim  ·  %s/%s MB\n",$1,$2,$3,$4}'
fi

# 5) Log'un son anlamli satiri (tqdm ve HF gurultusu atiliyor)
printf '\n  son satir\n'
grep -vE 'it/s\]|Loading|Defaulting|float32|^\s*$|HF_TOKEN' "$LOG" 2>/dev/null | tail -2 | sed 's/^/    /'

# 6) Proses yasiyor mu?
if pgrep -f 'inpaint.py run|species_check.py' >/dev/null; then
  printf '\n  durum     CALISIYOR\n'
else
  printf '\n  durum     PROSES YOK -- ya hepsi bitti ya da coktu, son satira bak\n'
fi
