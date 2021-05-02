import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2.dnn as cvdnn


class expandLayer(nn.Module):
    def __init__(self):
        super(expandLayer, self).__init__()
        self.kernel1 = torch.cuda.FloatTensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]).reshape(1, 1, 3, 3)
        self.kernel2 = torch.cuda.FloatTensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).reshape(1, 1, 3, 3)
        self.empty = torch.cuda.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(1, 1, 3, 3)
        self.weights = nn.Parameter(data=torch.cat((self.kernel1, self.kernel2, self.empty), 0), requires_grad=False)

    def forward(self, x):
        # shape = x.shape
        # print("conv input", shape)
        # print(x.shape)
        # print(shape)
        # if len(shape) != 4:
        #     x = x.reshape(1, 1, shape[0], shape[1])
        # print(x.shape)
        # x = torch.tensor(x).float()
        # height_gradient.weight = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        # print(F.conv2d(x, self.weights, padding=1, stride=1).shape)
        return F.conv2d(x, self.weights, padding=1, stride=1)
        # return x


def main() -> None:
    layer = expandLayer()
    path = "..\\..\\barcode_detection_dataset\\pictures\\images\\20210316_2\\IMG_20210316_121311.jpg"
    picture = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # picture = cv2.imread(os.path.join('.\\..\\data\\barcode\\images', 'train.jpg'), cv2.IMREAD_GRAYSCALE)
    temp = layer(picture)
    print(temp.shape)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(np.array(temp[0, 0]))
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(np.array(temp[0, 1]))
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(np.array(temp[0, 2]))
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
    print(temp)


if __name__ == '__main__':
    main()
