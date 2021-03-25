#!/bin/bash
set -e
set -x
set -uo pipefail

function train_barcode() {
  python3 ./train.py --dataset barcode \
    --batch_size 16 \
    --num_workers 4 \
    --cuda true \
    --lr 1e-5 \
    --img_list_file_path "./data/barcode/train.txt"
}
function train_voc() {
  python3 ./train.py --dataset VOC \
    --batch_size 16 \
    --num_workers 4 \
    --cuda true \
    --img_list_file_path "./data/barcode/CorrectDetect.txt"
}
"$1"
