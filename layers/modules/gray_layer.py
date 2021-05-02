import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class grayLayer(nn.Module):
    def __init__(self):
        super(grayLayer, self).__init__()
        self.red = torch.cuda.FloatTensor([0.299]).reshape(1, 1, 1, 1)
        self.green = torch.cuda.FloatTensor([0.587]).reshape(1, 1, 1, 1)
        self.blue = torch.cuda.FloatTensor([0.114]).reshape(1, 1, 1, 1)
        self.weights = nn.Parameter(data=torch.cat((self.red, self.green, self.blue), 1), requires_grad=False)
        # self.weights = nn.Parameter(data=self.red, requires_grad=False)

    def forward(self, x):
        # print("gray input", x.shape)
        # x = x.cpu()
        # x = np.transpose(x, (2, 0, 1))
        # x = torch.FloatTensor(x)
        # x = x.reshape(1, shape[0], shape[1], shape[2])
        # print(x.shape)
        # return F.conv2d(x, self.weights, padding=0, stride=1)

        return F.conv2d(x, self.weights)
        # return x



def check():
    input = torch.ones(1, 3, 5, 5)
    print(input)
    input = Variable(input)
    x = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, groups=1)
    red = torch.FloatTensor([0.299]).reshape(1, 1, 1, 1)
    green = torch.FloatTensor([0.587]).reshape(1, 1, 1, 1)
    blue = torch.FloatTensor([0.114]).reshape(1, 1, 1, 1)
    weights = nn.Parameter(data=torch.cat((red, green, blue), 1), requires_grad=False)
    print(weights.shape)
    out = F.conv2d(input, weights)
    print(out)
    exit(-1)


def pltshow(img):
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(np.array(img))
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()


def main() -> None:
    # check()
    layer = grayLayer()
    path = "..\\..\\barcode_detection_dataset\\pictures\\images\\20210316_2\\IMG_20210316_121311.jpg"
    picture = cv2.imread(path)
    pltshow(picture)

    # picture = cv2.imread(os.path.join('.\\..\\data\\barcode\\images', 'train.jpg'), cv2.IMREAD_GRAYSCALE)
    temp = layer(picture)
    pltshow(temp.detach().numpy()[0, 0])
    print(temp)


if __name__ == '__main__':
    main()
