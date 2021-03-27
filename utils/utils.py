#!/usr/bin/env python3
# coding=utf-8
import glob
import re
from pathlib import Path
from typing import Tuple, List

import numpy
import torch
import os

from Cython import typeof
import cv2


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def init_torch_tensor(args_cuda: bool = False):
    if torch.cuda.is_available():
        if args_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                  CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def get_output_dir(name: str, phase: str) -> str:
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_output_dirs(*args: Tuple[str]) -> str:
    filedir = os.path.join(*args)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def coco_to_yolo(rect: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Args:
        rect: (x_min,y_min,x_max,y_max)
        shape: (x_net_len,h_net_len)
    Returns:
        (x_mid,y_mid,x_len,y_len)
    """
    # print(f'[x_min,y_min,x_max,y_max] is {rect[0]} {rect[1]} {rect[2]} {rect[3]}')
    will_return = [-1, -1, -1, -1]
    will_return[0], will_return[1] = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    will_return[2], will_return[3] = (rect[2] - rect[0]), (rect[3] - rect[1])
    will_return[0], will_return[2] = will_return[0] / shape[0], will_return[2] / shape[0]
    will_return[1], will_return[3] = will_return[1] / shape[1], will_return[3] / shape[1]
    # print(f'[x_mid,y_mid,x_len,y_len] is {will_return[0]}, {will_return[1]}, {will_return[2]}, {will_return[3]}')
    return will_return[0], will_return[1], will_return[2], will_return[3]


def coco_to_percent(rect: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[
    float, float, float, float]:
    """
    Args:
        rect: (x_min,y_min,x_max,y_max)
        shape: (x_net_len,h_net_len)
    Returns:
        (x_mid,y_mid,x_len,y_len)
    """
    # print(f'[x_min,y_min,x_max,y_max] is {rect[0]} {rect[1]} {rect[2]} {rect[3]}')
    will_return = list(rect)
    will_return[0], will_return[2] = will_return[0] / shape[0], will_return[2] / shape[0]
    will_return[1], will_return[3] = will_return[1] / shape[1], will_return[3] / shape[1]
    # print(f'[x_mid,y_mid,x_len,y_len] is {will_return[0]}, {will_return[1]}, {will_return[2]}, {will_return[3]}')
    return will_return[0], will_return[1], will_return[2], will_return[3]


def draw_picture_with_label(img, labels: List[Tuple[int, int, int, int]] or Tuple[int, int, int, int]):
    """
    Args:
        img: come from cv2.read()
        labels: List[Tuple[int,int,int,int]]
         in each Tuple[], order is (x_min,y_min,x_max,y_max)
         Ps: each number is percent,
    Returns:
        a drawed labels image
    """
    x_net, y_net, _ = img.shape
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        if len(label) < 4:
            continue
        cv2.rectangle(img, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (0, 255, 0), 6)
        # cv2.imshow('x', img)
        print(label)
    cv2.imshow('x', cv2.resize(img, (img.shape[0] // 5, img.shape[1] // 5)))
    cv2.waitKey(0)
    return img


def save_picture_with_label(file_name_with_dir: str,
                            img, labels: List[Tuple[int, int, int, int]] or Tuple[int, int, int, int]):
    """
    Args:
        img: come from cv2.read()
        labels: List[Tuple[int,int,int,int]]
         in each Tuple[], order is (x_min,y_min,x_max,y_max)
         Ps: each number is percent,
    Returns:
        a drawed labels image
    """
    x_net, y_net, _ = img.shape
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        if len(label) < 4:
            continue
        cv2.rectangle(img, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (0, 255, 0), 6)
        # cv2.imshow('x', img)
    cv2.imwrite(file_name_with_dir, cv2.resize(img, (img.shape[0] // 5, img.shape[1] // 5)))
    return img
