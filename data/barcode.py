#!/usr/bin/env python3
# coding=utf-8
"""Barcode Dataset Classes
Updated by: Ellis Brown, Max deGroot

"""
from typing import Tuple, List, Any, Union

from utils import SSDAugmentation
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path
from random import randint

BARCODE_CLASS = ('barcode',)  # dont delete the ,


# note: if you used our download scripts, this should be right

class BARCODEAnnotationTransform(object):
    def __init__(self):
        super(BARCODEAnnotationTransform).__init__()

    def __call__(self, filepath: str, shape: Tuple[int, int]):
        """
        Arguments:
            filepath: string
                a file's path that contain the yolo format anno
            shape: Tuple[int,int]
                x_len,y_len
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
            [[xmin, ymin, xmax, ymax, label_ind], ... ]
        """
        res: List[Tuple[float, float, float, float, int]] = []
        with open(filepath, 'r') as file:
            for line in file.readlines():
                _, x, y, diffx, diffy = line.split(" ")
                x, y, diffx, diffy = float(x), float(y), float(diffx), float(diffy)
                res.append((x - diffx / 2, y - diffy / 2, x + diffx / 2, y + diffy / 2, 0))
        return res


class BARCODEDetection(data.Dataset):
    """Barcode Detection Dataset Object
    input is image, target is annotation

    Arguments:
        img_list_path: str
        filepath to a txt that contains pictures list
        transform (callable, optional): transformation to perform on the
            input image
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, img_list_path: str, transform=None, dataset_name='barcode'):
        self.transform = transform
        self.name: str = dataset_name
        self.target_transform = BARCODEAnnotationTransform()
        self.imgLists: List[str] = list()
        with open(img_list_path, 'r') as imglistfile:
            self.imgLists = [str(Path(imgfile).absolute()).replace("\n", "")
                             for imgfile in imglistfile.readlines()]
        self.labelLists: List[str] = [path.replace("images", "labels")
                                          .replace(".png", ".txt")
                                          .replace(".jpg", ".txt")
                                          .replace(".JPG", ".txt")
                                          .replace(".JPEG", ".txt")
                                      for path in self.imgLists]
        self.imgMap = {}
        self.imgLabelMap = {}
        self.origin_data_length: int = len(self.imgLists)
        self.aimNum: int = int(1401)
        # self.increase()

    def __len__(self) -> int:
        # return self.aimNum - 1
        return len(self.imgLists)

    def __getitem__(self, index):
        # print(f"get {index} item")
        img, gt, h, w = self.pull_item(index)
        # print(img.shape, ' -- ', gt.shape)
        # print(gt)
        return img, gt

    @staticmethod
    def transformRotate(img: np.ndarray, labels: List) -> Tuple[np.ndarray, List]:
        """
        Args:
            img: image read by cv2, it's a ndarray
            labels: the labels of image
        Returns:
            image and labels that transferred
            randomInt = 3,保持不变
            randomInt = 0,
        """
        randomInt = randint(0, 3)
        # print(randomInt)
        if randomInt == 3:  # 3 for no transfer
            return img, labels
        height, weight = img.shape[:2]
        returnImg = cv2.rotate(img, rotateCode=randomInt)
        returnLabels: List[Tuple]
        if randomInt == 0:
            returnLabels = [(1 - label[1], label[0], 1 - label[3], label[2], label[4]) for label in labels]
        elif randomInt == 1:
            returnLabels = [(1 - label[0], 1 - label[1], 1 - label[2], 1 - label[3], label[4]) for label in labels]
        else:
            returnLabels = [(label[1], 1 - label[0], label[3], 1 - label[2], label[4]) for label in labels]
        # print(height, weight)
        return returnImg, returnLabels

    @staticmethod
    def transformMerge(imgList: Tuple[np.ndarray], labels: Tuple[List]) -> Tuple[np.ndarray, List]:
        assert (len(imgList) == 4)
        assert (len(labels) == 4)
        img_h1 = cv2.hconcat([cv2.resize(imgList[0], (300, 300)), cv2.resize(imgList[1], (300, 300))])
        img_h2 = cv2.hconcat([cv2.resize(imgList[2], (300, 300)), cv2.resize(imgList[3], (300, 300))])
        img_return = cv2.vconcat([img_h1, img_h2])
        labels_return = []
        for label1 in labels[0]:
            labels_return.append((label1[0] / 2, label1[1] / 2, label1[2] / 2, label1[3] / 2, 0))
        for label2 in labels[1]:
            labels_return.append((0.5 + label2[0] / 2, label2[1] / 2, 0.5 + label2[2] / 2, label2[3] / 2, 0))
        for label3 in labels[2]:
            labels_return.append((label3[0] / 2, 0.5 + label3[1] / 2, label3[2] / 2, 0.5 + label3[3] / 2, 0))
        for label4 in labels[3]:
            labels_return.append(
                (0.5 + label4[0] / 2, 0.5 + label4[1] / 2, 0.5 + label4[2] / 2, 0.5 + label4[3] / 2, 0))
        return img_return, labels_return

    def increase(self) -> None:
        for index in range(0, self.origin_data_length, 1):
            self.pull_anno(index)
            print(f"prepareing {index}", flush=True)
        for index in range(self.origin_data_length, self.aimNum, 1):
            randomlist: List[int] = [randint(0, self.origin_data_length - 1) for _ in range(0, 4, 1)]
            img1, label1 = self.pull_anno(randomlist[0])
            img2, label2 = self.pull_anno(randomlist[1])
            img3, label3 = self.pull_anno(randomlist[2])
            img4, label4 = self.pull_anno(randomlist[3])
            imgnet, labelsnet = self.transformMerge((img1, img2, img3, img4), (label1, label2, label3, label4))
            imgnet = cv2.resize(imgnet, (300, 300))
            self.imgMap[index], self.imgLabelMap[index] = imgnet, labelsnet
            print(f"Mergeing {index}", flush=True)
            if index % 100000 == 0:
                draw_labels_in_image(imgnet, labelsnet)
                cv2.imshow("img", imgnet)
                cv2.waitKey(0)
        print("length", self.origin_data_length)

    def pull_item(self, index):
        img, img_labels = self.pull_anno(index)
        height, width, channels = img.shape
        if self.transform is not None:
            img_labels = np.array(img_labels)
            try:
                img, boxes, labels = self.transform(img, img_labels[:, :4], img_labels[:, 4])
                img_labels = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            except IndexError as ie:
                print(f"index Error {index}")
                exit(-1)
            # img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1), img_labels, height, width
        # RGB to BRG

    def pull_image(self, index) -> np.ndarray:
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            opencv img
        """
        if index in self.imgMap:
            img = self.imgMap[index]
        else:
            img_path = self.imgLists[index]
            img = cv2.imread(img_path)
            img = img[:, :, (2, 1, 0)]  # BGR to RGB
            img = cv2.resize(img, (300, 300))
            self.imgMap[index] = img
        return img

    def pull_anno(self, index) -> Tuple[np.ndarray, List[Tuple[float, float, float, float, int]]]:
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img = self.pull_image(index)
        height, width, channels = img.shape
        if index in self.imgLabelMap:
            img_labels = self.imgLabelMap[index]
        else:
            label_path = self.labelLists[index]
            img_labels = self.target_transform(label_path, (width, height))
            self.imgLabelMap[index] = img_labels
        return img, img_labels

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def main() -> None:
    dataset = BARCODEDetection(
        img_list_path="barcode/train.txt",
        transform=SSDAugmentation(300, (104, 117, 123)))
    dataset.pull_item(0)
    img, labels, = dataset.pull_anno(0)

    print(img is None)
    img, labels = dataset.transformRotate(img, labels)
    print(img is None)
    shape = img.shape
    draw_labels_in_image(img, labels)
    img = cv2.resize(img, (512, 512))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    img, labels, = dataset.pull_anno(0)
    img2, labels2, = dataset.pull_anno(1)
    img3, labels3, = dataset.pull_anno(2)
    img4, labels4, = dataset.pull_anno(3)
    imgnet, labelsnet = dataset.transformMerge((img, img2, img3, img4), (labels, labels2, labels3, labels4))
    draw_labels_in_image(imgnet, labelsnet)
    # cv2.imshow("img", imgnet)
    #
    # cv2.waitKey(0)


def draw_labels_in_image(img: np.ndarray, labels: List) -> np.ndarray:
    y_len, x_len = img.shape[0], img.shape[1]
    for label in labels:
        x_min, y_min, x_max, y_max, _ = label
        x_min, y_min, x_max, y_max = int(x_min * x_len), int(y_min * y_len), int(x_max * x_len), int(y_max * y_len),
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
    return img


if __name__ == '__main__':
    main()
