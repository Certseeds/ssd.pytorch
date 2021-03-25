#!/usr/bin/env python3
# coding=utf-8
# config.py
import os.path

# gets home dir cross platform
from typing import Dict

HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc: Dict = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
    'max_epoch': 300,
}

coco: Dict = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
    'max_epoch': 300,
}

barcode_cfg_dict: Dict = dict(num_classes=2, lr_steps=(280000, 350000, 400000), lr_epoch=(150, 220, 275),
                              min_dim=300, steps=(8, 16, 32, 64, 100, 300), min_sizes=(30, 60, 111, 162, 213, 264),
                              max_sizes=(60, 111, 162, 213, 264, 315), variance=(0.1, 0.2), clip=True,
                              max_iter=120000, max_epoch=350, feature_maps=(38, 19, 10, 5, 3, 1),
                              aspect_ratios=((2), (2, 3), (2, 3), (2, 3), (2), (2)),
                              name='barcode')
