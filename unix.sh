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
function l() {
  python3 ./eval.py --dataset barcode \
    --save_folder eval/ \
    --cuda true \
    --cleanup false
}
function test_small() {
  python3 ./test.py --dataset barcode \
    --trained_model weights/ssd300_barcode/349.pth \
    --save_folder test \
    --cuda true
}
function test_all() {
  python3 ./test.py --dataset barcode \
    --trained_model weights/ssd300_barcode/349.pth \
    --img_list_file_path data/barcode/train.txt \
    --save_folder test \
    --cuda true
}
function test_voc() {
  python3 ./test.py \
    --trained_model weights/ssd300_VOC/0.pth \
    --save_folder test \
    --cuda true
}
"$1"
