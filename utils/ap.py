#!/usr/bin/env python3
# coding=utf-8

# what is the input of ap?
# first, a list of img should be send
# then, it's pred label is a List[List[(x_min,y_min,x_max,y_max,score)]]
# for each img, a List[(x_min,y_min,x_max,y_max,score)] it's the pred result
# then for this img, a List[(x_min,y_min,x_max,y_max,score)] is the real-labels
from pathlib import Path
from typing import Tuple, List, Set

from utils.iou import iot_geter

x = 1

geter = iot_geter(0.5)


def file_labels(GroundTruths: List[List], predruths: List[List]) -> Tuple[List[Tuple[int, float]], int]:
    global x
    predList: List[Tuple[int, float]] = list()
    for predruth in predruths:
        bb = geter.filter_by_threshold(predruth, GroundTruths)
        predList.append((bb + x, predruth[4]))
        print(bb + x)
    x += len(GroundTruths)
    return predList, len(GroundTruths)


# First, read a txt to get the imgnames.
# then, read labels by pointer path and the imgnames by f'{path}/{imgname}.txt'
# then, the imgnames can be replace by labels txts
# then, we have double List[List[int]]
# TP: place all bb in a Set(), then len(Set())
# FP: the number of -1
# FN: len(GroundTruths) - TP
def get_dataset_ap(dataset, detect_labespath: str):
    annos = len(dataset)
    ground_truth_sum = 0
    pred_listes: List[Tuple[int, float]] = []
    for i in range(0, annos, 1):
        _, ground_truth = dataset.pull_anno(i)
        filestem = Path(dataset.imgLists[i]).stem
        with open(f'{detect_labespath}/{filestem}.txt', 'r') as file:
            pred_truth = file.readlines()
            pred_truth = [pred_trut.split(" ") for pred_trut in pred_truth]
        pred_truth = [
            [float(pred_trut[1]), float(pred_trut[2]), float(pred_trut[3]), float(pred_trut[4]), float(pred_trut[5])]
            for pred_trut in pred_truth]
        print(f"{filestem}")
        print(f"{ground_truth} \n {pred_truth}")
        pred_list, groud_truth_num = file_labels(ground_truth, pred_truth)
        ground_truth_sum += groud_truth_num
        pred_listes.extend(pred_list)
    pred_listes.sort(key=lambda conf: -conf[1])
    pres, recas = [], []
    for i in range(0, len(pred_listes), 1):
        temp = tp_fp_fn_by_predlistes_ground_truth_and_p_threshold(pred_listes, ground_truth_sum, i + 1)
        print(temp)
        _, _, _, pre, reca = temp
        print(f"{pre} {reca}\n")
        pres.append(pre)
        recas.append(reca)
    return pres, recas


def tp_fp_fn_by_predlistes_ground_truth_and_p_threshold(pred_listes: List[Tuple[int, float]], ground_truth_sum: int,
                                                        rank: int) -> Tuple[int, int, int, float, float]:
    ground_truth_numberset = set()
    for i in range(0, rank, 1):
        ground_truth_numberset.add(pred_listes[i][0])
    tp = len(ground_truth_numberset)
    fp = rank - tp
    fn = ground_truth_sum - tp
    precision = tp / rank
    recall = tp / ground_truth_sum
    return tp, fp, fn, precision, recall


def sort_by_p_conf(s1: Tuple[int, float], s2: Tuple[int, float]) -> int:
    if s1[1] > s2[1]:
        return 1
    elif s1[1] < s2[1]:
        return -1
    return 0


if __name__ == '__main__':
    test_pred_listes = [(1, 0.9), (3, 0.5), (2, 0.8)]
    test_pred_listes.sort(key=lambda conf: conf[1])
    print(test_pred_listes)
    test_pred_listes = sorted(test_pred_listes, key=lambda conf: -conf[1])
    print(test_pred_listes)
