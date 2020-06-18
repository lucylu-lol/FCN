import numpy as np
import cv2
import glob
import os
import itertools
import matplotlib.pyplot as plt
import random


def getImageArr(img):
    img = img.astype(np.float32)
    # 三通道去均值
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img

def getSegmentationArr(seg,nClasses,input_height,input_width):
    seg_labels = np.zeros((input_height,input_width,nClasses))
    # seg为编码好的图片由（0，nClasses）组成，完成one-hot编码[True,False,False]
    for c in range(nClasses):
        seg_labels[:,:,c] = (seg==c).astype(int)

    # [height*weight,nclasses],不用reshape也可以，对应的网络结构也不用，两者对应即可
    seg_labels = np.reshape(seg_labels,(-1,nClasses))
    return seg_labels

def imageSegmentationGenerator(images_path,segs_path,batch_size,n_classes,input_height,input_width):
    # images_path = "CamVid/train"
    # segs_path = "CamVid/trainannot"
    # batch_size = 2
    # n_classes = 11
    # input_height = 320
    # input_width = 320
    # 排序，使图片和标签相对应
    # os.path.join生成CamVid/train*.png，glob生成符合条件图片列表
    images = sorted(glob.glob(os.path.join(images_path,'*.png')))
    segmentations = sorted(glob.glob(os.path.join(segs_path,"*.png")))
    zipped = itertools.cycle(zip(images,segmentations))
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img, seg = zipped.__next__()
            # 读取成彩色图像
            img = cv2.imread(img, 1)
            # 读取成灰度图像
            seg = cv2.imread(seg, 0)
            # 原图太大训练不方便，resize也可以，但可能改变图像标签像素对应，使用裁剪
            # 如：原图[512,512],裁剪成{256,256]在512-256=256之间生成初始行列值
            idx = random.randint(0,img.shape[0] - input_height)
            idy = random.randint(0,img.shape[1] - input_width)
            img = img[idx:idx+input_height, idy:idy+input_width]
            seg = seg[idx:idx+input_height,idy:idy+input_width]
            X.append(getImageArr(img))
            Y.append(getSegmentationArr(seg, n_classes,input_height,input_width))
        # X ：[batch_size,h,w,3] Y :[batch_size,h,w,nclasses]/[batch_size,h*w,nclasses]
        yield np.array(X), np.array(Y)

def segSegmentationGenerator():
    # images_path,segs_path,batch_size,n_classes,input_height,input_width
    images_path = "CamVid/val"
    segs_path = "CamVid/valannot"
    batch_size = 2
    n_classes = 11
    input_height = 320
    input_width = 320
    # 排序，使图片和标签相对应
    # os.path.join生成CamVid/train*.png，glob生成符合条件图片列表
    images = sorted(glob.glob(os.path.join(images_path,'*.png')))
    segmentations = sorted(glob.glob(os.path.join(segs_path,"*.png")))
    zipped = itertools.cycle(zip(images,segmentations))
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img, seg = zipped.__next__()
            # 读取成彩色图像
            img = cv2.imread(img, 1)
            # 读取成灰度图像
            seg = cv2.imread(seg, 0)
            # 原图太大训练不方便，resize也可以，但可能改变图像标签像素对应，使用裁剪
            # 如：原图[512,512],裁剪成{256,256]在512-256=256之间生成初始行列值
            idx = random.randint(0,img.shape[0] - input_height)
            idy = random.randint(0,img.shape[1] - input_width)
            img = img[idx:idx+input_height, idy:idy+input_width]
            seg = seg[idx:idx+input_height,idy:idy+input_width]
            X.append(getImageArr(img))
            Y.append(getSegmentationArr(seg, n_classes,input_height,input_width))
        # X ：[batch_size,h,w,3] Y :[batch_size,h,w,nclasses]/[batch_size,h*w,nclasses]
        yield np.array(X), np.array(Y)

if __name__ == "__main__":
    imgPath = "CamVid/train"
    segPath = "CamVid/trainannot"

    Genertor = imageSegmentationGenerator(imgPath,segPath,batch_size=4,n_classes=15,input_height=320,input_width=320)
    for (x,y) in Genertor:
        print(x,y)
        print(x.shape,y.shape)









