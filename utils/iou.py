#!/usr/bin/env python3
# coding=utf-8
from typing import Tuple, List, Union

import numpy as np

getitem_able = Union[List, Tuple]


def get_iou(pred_bbox: getitem_able, gt_bbox: getitem_able) -> float:
    """
    Args:
        pred_bbox: it should be iterable, len >=4
        gt_bbox: it should be iterable, len >=4
    Returns:
        the iou of double box, float between [0,1)
    """
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) +
           (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni
    return overlaps


def get_max_iou(pred_bboxes: getitem_able, gt_bbox: getitem_able):
    """
        given 1 gt bbox, >1 pred bboxes, return max iou score for the given gt bbox and pred_bboxes
    Args:
        pred_bboxes: List[getitem_able]
            predict bboxes coordinates, we need to find the max iou score with gt bbox for these pred bboxes
        gt_bbox: getitem_able
             ground truth bbox coordinate
    Returns:
        max iou score
    """
    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and (abs(gt_bbox[3] - gt_bbox[1]) > 0))
    pred_bboxes = np.array(pred_bboxes)
    gt_bbox = np.array(gt_bbox)
    if len(pred_bboxes) > 0:
        # -----0---- get coordinates of inters, but with multiple predict bboxes
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # -----1----- intersection
        inters = iw * ih

        # -----2----- union, uni = S1 + S2 - inters
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)

        # -----3----- iou, get max score and max iou index
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    return overlaps, ovmax, jmax


class iot_geter(object):
    def __init__(self, iou_threshold: float):
        super(iot_geter, self).__init__()
        self.threshold = iou_threshold

    def filter_by_threshold(self, pred_bbox: getitem_able, gt_bboxes: getitem_able):
        _, iou_percent, iou_index = get_max_iou(gt_bboxes, pred_bbox)
        if iou_percent > self.threshold:
            return iou_index
        else:
            return -1


if __name__ == "__main__":
    # test1
    pred_bbox = np.array([50, 50, 90, 100])  # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    gt_bbox = np.array([70, 80, 120, 150])
    print(get_iou(pred_bbox, gt_bbox))

    # test2
    pred_bboxes = np.array([[15, 18, 47, 60],
                            [50, 50, 90, 100],
                            [70, 80, 120, 145],
                            [130, 160, 250, 280],
                            [25.6, 66.1, 113.3, 147.8]])
    gt_bbox = np.array([70, 80, 120, 150])
    print(get_max_iou(pred_bboxes, gt_bbox))
