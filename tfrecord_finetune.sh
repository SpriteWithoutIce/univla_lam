export PYTHONPATH=/home/linyihan/linyh/univla_lam:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATA_ROOT="/home/linyihan/linyh/datasets/robotwin/lift_pot/1.0.0"

if [ $# -lt 1 ]; then
  TFRECORD_PATH="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
else
  TFRECORD_PATH="$1"
  shift
fi

if [ ! -e "$TFRECORD_PATH" ]; then
  echo "TFRecord path not found: $TFRECORD_PATH" >&2
  exit 1
fi

python genie/tfrecord_finetune.py "$TFRECORD_PATH" \
  --checkpoint /home/linyihan/linyh/univla_lam/checkpoints/lam-stage-2.ckpt \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-epochs "${MAX_EPOCHS:-5}" \
  --devices "${DEVICES:-1}" \
  --num-workers "${NUM_WORKERS:-1}" \
  --frame-interval "${FRAME_INTERVAL:-32}" \
  --subset-fraction "${SUBSET_FRACTION:-1.0}" \
  --image-size "${IMAGE_SIZE:-56}" \
  --action-loss-weight "${ACTION_LOSS_WEIGHT:-1.0}" \
  --action-hidden "${ACTION_HIDDEN:-256}" \
  "$@"
