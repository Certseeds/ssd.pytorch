#!/usr/bin/env python3
# coding=utf-8
"""Barcode Dataset Classes
Updated by: Ellis Brown, Max deGroot

"""
from typing import Tuple, List

from utils import SSDAugmentation
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path

BARCODE_CLASS = ('barcode')


# note: if you used our download scripts, this should be right

class BARCODEAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self):
        super(BARCODEAnnotationTransform).__init__()

    def __call__(self, filepath: str, shape: Tuple[int, int]):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res: List[Tuple[float, float, float, float, int]] = []
        with open(filepath, 'r') as file:
            for line in file.readlines():
                _, x, y, diffx, diffy = line.split(" ")
                x, y, diffx, diffy = float(x), float(y), float(diffx), float(diffy)
                res.append((x - diffx / 2, y - diffy / 2, x + diffx / 2, y + diffy / 2, 0))
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class BARCODEDetection(data.Dataset):
    """Barcode Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, img_list_path: str, transform=None, dataset_name='barcode'):
        self.transform = transform
        self.name = dataset_name
        self.target_transform = BARCODEAnnotationTransform()
        self.imgLists: List[str] = list()
        with open(img_list_path, 'r') as imglistfile:
            self.imgLists = [str(Path(imgfile).absolute()).replace("\n", "")
                             for imgfile in imglistfile.readlines()]
        self.labelLists: List[str] = [path.replace("images", "labels")
                                          .replace(".png", ".txt")
                                          .replace(".jpg", ".txt")
                                          .replace(".JPG", ".txt")
                                      for path in self.imgLists]
        self.imgMap = {}
        self.imgLabelMap = {}

    def __len__(self):
        return len(self.imgLists)

    def __getitem__(self, index):
        img, gt, h, w = self.pull_item(index)
        # print(img.shape, ' -- ', gt.shape)
        # print(gt)
        return img, gt

    def pull_item(self, index):
        if index in self.imgMap:
            img = self.imgMap[index]
        else:
            img_path = self.imgLists[index]
            img = cv2.imread(img_path)
            self.imgMap[index] = img
        height, width, channels = img.shape
        if index in self.imgLabelMap:
            img_labels = self.imgLabelMap[index]
        else:
            label_path = self.labelLists[index]
            img_labels = self.target_transform(label_path, (width, height))
            self.imgLabelMap[index] = img_labels
        # print(img_labels)
        if self.transform is not None:
            target = np.array(img_labels)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        if index in self.imgMap:
            img = self.imgMap[index]
        else:
            img_path = self.imgLists[index]
            img = cv2.imread(img_path)
            self.imgMap[index] = img
        return img

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
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
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def main() -> None:
    dataset = BARCODEDetection(
        img_list_path="barcode/CorrectDetect.txt",
        transform=SSDAugmentation(300, (104, 117, 123)))


if __name__ == '__main__':
    main()
