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
    --img_list_file_path data/barcode/CorrectDetect.txt \
    --trained_model weights/ssd300_barcode/349.pth \
    --save_folder test \
    --cuda true \
    --test_or_eval false
}
function test_all() {
  python3 ./test.py --dataset barcode \
    --trained_model weights/ssd300_barcode/349.pth \
    --img_list_file_path data/barcode/train.txt \
    --save_folder test \
    --cuda true
}
function ap_small() {
  python3 ./ap_test.py \
    --img_list_file_path data/barcode/CorrectDetect.txt \
    --pred_label_path test/barcode5/labels
}
function ap_all() {
  python3 ./ap_test.py \
    --img_list_file_path data/barcode/train.txt \
    --pred_label_path test/barcode6/labels
}
"$1"
