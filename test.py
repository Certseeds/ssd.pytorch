#!/usr/bin/env python3
# coding=utf-8
from __future__ import print_function
import sys
import os
import time
import datetime
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import BARCODEDetection, BARCODE_CLASS
from data import BARCODE_CLASS as labelmap
from data import BaseTransform, MEANS
import torch.utils.data as data
from ssd import build_ssd
from utils import init_torch_tensor, increment_path, Path, coco_to_yolo, str2bool, draw_picture_with_label, \
    save_picture_with_label, coco_to_percent
from typing import Dict, List, Tuple


def init_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval', type=str, help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.6, type=float, help='Final confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
    parser.add_argument("--img_list_file_path", default="./data/barcode/CorrectDetect.txt", type=str,
                        help="a file path")
    parser.add_argument("--dataset", default="barcode", choices=['VOC', 'COCO', 'barcode'], type=str,
                        help="You know the rules")
    parser.add_argument("--test_or_eval", default=True, type=str2bool,
                        help="True: output for test, label format is yolo, "
                             "False: Output for eval, label format is '0 x_min y_min x_max y_max conf' per line")
    args_r = parser.parse_args()
    print(args_r.test_or_eval)
    args_r.test_save_path = increment_path(Path("test") / args_r.dataset, exist_ok=False)
    os.makedirs(args_r.test_save_path)
    os.makedirs(Path(args_r.test_save_path) / 'labels')
    return parser, args_r


parser, args = init_parser()


# DONE 计算AP
def test_net(save_folder: str, net: nn.Module, cuda: bool, testset: BARCODEDetection, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    begintime = time.time()
    for i in range(num_images):
        prev_time = time.time()
        print(f'Testing image {(i + 1):d}/{num_images:d}....')
        img = testset.pull_image(i)
        # img_id, annotation = testset.pull_anno(i)
        print(testset.imgLists[i])
        file_stem = str(Path(testset.imgLists[i]).stem)
        file = open(f'{args.test_save_path}/labels/{file_stem}.txt', mode='w')

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(0).cuda(non_blocking=True)
        y = net(x)  # forward pass
        print(f"\t+ Batch {i}, net Time:{time.time()- prev_time}",flush=True)
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0
        will_draw = list()
        for i2 in range(detections.size(1)):
            j = 0
            print(detections.shape)
            while detections[0, i2, j, 0] >= args.visual_threshold:
                origin_rect = detections[0, i2, j, 1:]
                score = detections[0, i2, j, 0]
                # label_name = labelmap[i2 - 1]
                pt = (origin_rect * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                # img = draw_picture_with_label(img, coords)
                # file.write(f'0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f} {score:.6f}\n')
                will_draw.append(coords)
                if not args.test_or_eval:
                    coords = coco_to_percent(coords, (img.shape[1], img.shape[0]))
                    file.write(f'0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f} {score:.6f}\n')
                    # this is eval
                pred_num += 1
                print(score)
                if args.test_or_eval:
                    coords = coco_to_yolo(coords, (img.shape[1], img.shape[0]))
                    file.write(f'0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n')
                j += 1
        # draw_picture_with_label(img, will_draw)
        inference_time = datetime.timedelta(seconds=time.time()- prev_time)
        print("\t+ Batch %d, Inference Time: %s" % (i, inference_time),flush=True)
        # save_picture_with_label(f'{args.test_save_path}/{file_stem}.jpg', img, will_draw)
        #file.close()
    net_time = time.time() - begintime
    print(f"sum time {net_time}\n")
    print(f"avg time {net_time / num_images}\n")
    print(f"avg frame { num_images /net_time}\n")

def test_voc():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    # load net
    num_classes = len(BARCODE_CLASS) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net = net.to(device)
    net.eval()
    print('Finished loading model!')
    # load data
    # default net.size == 300
    testset = BARCODEDetection(img_list_path=args.img_list_file_path, transform=BaseTransform(net.size, MEANS))
    # TODO,remeber do it
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.test_save_path, net, args.cuda, testset, BaseTransform(net.size, MEANS),
             thresh=args.visual_threshold)


def main() -> None:
    init_torch_tensor(args.cuda)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    test_voc()


if __name__ == '__main__':
    main()
