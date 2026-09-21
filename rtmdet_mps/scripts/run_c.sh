#!/usr/bin/env bash
# Run C: strong augmentation (mosaic/resize/rotate/crop/HSV/flip/mixup), 512 px, batch 8, 100 epochs, last 30 light.
set -u
PY=../.venv/Scripts/python
DATA="../Annotated banknotes 2.yolov8-obb"
LOG=work_dirs/gpu_experiments.log
NAME=rtmdet_tiny_v1_aug_strong512
echo "$(date) === $NAME batch 8 @512, 100 ep, stage2 30" | tee -a $LOG
$PY -W ignore train.py --data "$DATA" --split-dir ../v1 --extra-train ../augmented \
    --resume work_dirs/$NAME/latest.pth --work-dir work_dirs/$NAME \
    --epochs 100 --stage2-epochs 30 --strong-aug --batch 8 --img-size 512 --device cuda --workers 4 \
    --val-interval 2 >> work_dirs/$NAME.log 2>&1 || echo "$(date) $NAME FAILED" | tee -a $LOG
best=$(ls work_dirs/$NAME/best_epoch_*.pth 2>/dev/null | head -1)
for ck in "$best" work_dirs/$NAME/epoch_100.pth; do
  [ -f "$ck" ] || continue
  for sz in 512 640; do
    echo "$(date) === test v1 $ck @$sz" | tee -a $LOG
    $PY -W ignore evaluate.py "$ck" --data "$DATA" --split-dir ../v1 --split test --img-size $sz --device cuda --workers 2 --nms-iou 0.3 2>&1 | grep nms | tee -a $LOG
  done
  for sz in 512 640 800 1024; do
    echo "$(date) === billetesprueba $ck @$sz" | tee -a $LOG
    $PY -W ignore evaluate.py "$ck" --data ../data/billetesprueba --dota --img-size $sz --batch 2 --device cuda --workers 2 --nms-iou 0.3 2>&1 | grep nms | tee -a $LOG
  done
done
echo "$(date) run C done" | tee -a $LOG
