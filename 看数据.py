import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from d2l import torch as d2l
# 本函数已保存在d2lzh包中方便以后使用
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root= "./data", train=True, transform=trans, download= False
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=False
)

def get_fashion_mnist_labels(labels): 
    '''返回Fashion-Mnist数据集的标签'''
    text_labels = ['t-shirt', 'trouser','pullover', 'dress','coat','sandal',
                    'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


