#!/bin/bash
set -e
set -x
set -uo pipefail

function train_voc() {
  python3 ./train.py --dataset VOC
}
"$1"
