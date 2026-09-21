#!/usr/bin/env bash
# Waits for the GPU to be free, then trains RTMDet-R tiny on the v1 split:
#   A: v1 train only            -> work_dirs/rtmdet_tiny_v1
#   B: v1 train + augmented/    -> work_dirs/rtmdet_tiny_v1_aug
# and evaluates both best checkpoints on v1 test. Run from rtmdet_mps/.
set -u
PY=../.venv/Scripts/python
DATA="../Annotated banknotes 2.yolov8-obb"
INIT=../checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth
LOG=work_dirs/gpu_experiments.log
mkdir -p work_dirs
echo "$(date) waiting for GPU (memory.used < 600 MiB for 5 consecutive minutes)" | tee -a $LOG
free=0
while [ $free -lt 5 ]; do
  sleep 60
  if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 600 ]; then free=$((free+1)); else free=0; fi
done
echo "$(date) GPU free, starting" | tee -a $LOG

train() {  # name, extra args...
  local name=$1; shift
  for bs in 8 4; do
    echo "$(date) === $name batch $bs" | tee -a $LOG
    $PY -W ignore train.py --data "$DATA" --split-dir ../v1 --init $INIT --work-dir work_dirs/$name \
        --epochs 36 --batch $bs --img-size 640 --device cuda --workers 4 "$@" > work_dirs/$name.log 2>&1 && return 0
    echo "$(date) $name failed with batch $bs (see work_dirs/$name.log)" | tee -a $LOG
    grep -q "out of memory" work_dirs/$name.log || return 1
    rm -rf work_dirs/$name
  done
  return 1
}

train rtmdet_tiny_v1
train rtmdet_tiny_v1_aug --extra-train ../augmented

for name in rtmdet_tiny_v1 rtmdet_tiny_v1_aug; do
  best=$(ls work_dirs/$name/best_epoch_*.pth 2>/dev/null | head -1)
  [ -z "$best" ] && continue
  for ck in "$best" work_dirs/$name/epoch_36.pth; do
    echo "$(date) === test $ck" | tee -a $LOG
    $PY -W ignore evaluate.py "$ck" --data "$DATA" --split-dir ../v1 --split test --img-size 640 --device cuda --workers 2 --nms-iou 0.3 0.5 2>&1 | grep -v Warning | tee -a $LOG
  done
done
echo "$(date) all done" | tee -a $LOG
