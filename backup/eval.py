#!/usr/bin/env python3
# coding=utf-8
"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
# ! This DO NOT BE USE IN THIS REPO
from __future__ import print_function

from typing import List, Dict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import BARCODEDetection, BARCODEAnnotationTransform, MEANS
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import BARCODE_CLASS as labelmap

import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import argparse
import numpy as np
import pickle
import cv2

from utils import str2bool, init_torch_tensor, get_output_dir, get_output_dirs

import xml.etree.ElementTree as ET

from utils.Timer import Timer


def init_parser():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float, help='Detection confidence threshold')
    parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
    parser.add_argument('--cleanup', default=True, type=str2bool,
                        help='Cleanup and remove results files following eval')
    parser.add_argument("--img_list_file_path", default="./data/barcode/CorrectDetect.txt", type=str,
                        help="a file path")
    parser.add_argument("--dataset", default="barcode", choices=['VOC', 'COCO', 'barcode'], type=str,
                        help="You know the rules")

    args = parser.parse_args()
    return parser, args


parser, args = init_parser()
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

init_torch_tensor(args.cuda)

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def write_voc_results_file(all_boxes: List[List[np.ndarray]], dataset: BARCODEDetection):
    for cls_ind, cls in enumerate(labelmap):
        print(f'Writing {cls} VOC results file')
        filename = get_output_dirs("eval", "test")
        file = open(f'{filename}.result', 'w')
        for im_ind, index in enumerate(dataset.imgLists):
            dets: np.ndarray = all_boxes[cls_ind + 1][im_ind]
            if len(dets) == 0:
                continue
            # the VOCdevkit expects 1-based indices
            for k in range(dets.shape[0]):
                file.write(f'{index[1]:s} {dets[k, -1]:.3f} '
                           f'{(dets[k, 0] + 1):.1f} {(dets[k, 1] + 1):.1f} '
                           f'{(dets[k, 2] + 1):.1f} {(dets[k, 3] + 1):.1f}\n')
        file.close()


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_output_dirs("eval", "test", "barcode")
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format('test'), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(dataset: BARCODEDetection,
             detpath: str,
             annopath: str,
             imagesetfile: str,
             classname: str,
             ovthresh: float = 0.5,
             use_07_metric: bool = True):
    """
    @detpath: str ,Path to detections
        detpath.format(classname) should produce the detection results file.
    @annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    @imagesetfile: Text file containing the list of images, one image per line.
    @classname: Category name (duh)
    @cachedir: Directory for caching the annotations
    @[ovthresh]: Overlap threshold (default = 0.5)
    @[use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    @Returns:
       rec, prec, ap
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = dict()
    for i in dataset.imgLabelMap:
        recs[dataset.imgLists[i]] = dataset.imgLabelMap[i]

    # extract gt objects for this class
    class_recs: Dict[str, Dict] = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


DETECTIONS = [BARCODEDetection]


def test_net(save_folder: str, net, cuda: bool, dataset: DETECTIONS, transform, top_k: int,
             im_size: int = 300, thresh: float = 0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes: List[List[np.ndarray]] = [[[] for _ in range(num_images)]
                                         for _ in range(len(labelmap) + 1)]

    # timers
    _t = dict(im_detect=Timer(), misc=Timer())
    output_dir = get_output_dirs('eval', f'ssd300_{args.dataset}', 'test')
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1), 1):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32,
                                                                                      copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    with open(det_file, 'wb') as file:
        pickle.dump(all_boxes, file, pickle.HIGHEST_PROTOCOL)
        # all_boxes serilies to file
    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list: List[List[np.ndarray]], output_dir: str, dataset: BARCODEDetection):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = BARCODEDetection(img_list_path=args.img_list_file_path,
                               transform=BaseTransform(300, MEANS))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, MEANS), args.top_k, 300,
             thresh=args.confidence_threshold)
