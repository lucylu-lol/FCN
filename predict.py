from load_batchs import *
from tensorflow.python.keras.models import load_model
from fcn32 import FCN32
from fcn16 import FCN16
from fcn8 import  FCN8
import glob
import cv2
import numpy as np
import random
import os

n_classes = 11

imgPath = "CamVid/train"
segPath = "CamVid/trainannot"

input_height = 320
input_weight = 320

colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_classes)]

def label2color(colors,n_classes,seg):
    seg_color = np.zeros((seg.shape[0],seg.shape[1],3))
    for c in range(n_classes):
        seg_color[:, :, 0] +=((seg==c)*(colors[c][0])).astype('uint8')
        seg_color[:, :, 1] +=((seg==c)*(colors[c][1])).astype('uint8')
        seg_color[:, :, 2] +=((seg==c)*(colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color

def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy
# id = random.randint(0,200)
id = 30
images = sorted(glob.glob(os.path.join(imgPath,"*.png")))[id:id+1]
segs = sorted(glob.glob(os.path.join(segPath,"*.png")))[id:id+1]

# model = FCN32(11,320,320)
# model.load_weights("model.h5")

# model = FCN16(11,320,320)
# model.load_weights("model16.h5")

model = FCN8(11,320,320)
model.load_weights("model8.h5")

for i,(img,seg) in enumerate(zip(images,segs)):
    img = cv2.imread(img,1)
    xx,yy = getcenteroffset(img.shape,input_height,input_weight)
    img = img[xx:xx+input_height,yy:yy+input_weight,:]
    seg = cv2.imread(seg,0)
    seg = seg[xx:xx+input_height,yy:yy+input_weight]

    pr = model.predict(np.expand_dims(getImageArr(img),0))[0]
    pr = pr.reshape((input_height,input_weight,n_classes)).argmax(axis=2)
    cv2.imshow("img",img)
    cv2.imshow("seg_predict",label2color(colors,n_classes,pr))
    cv2.imshow("seg",label2color(colors,n_classes,seg))
    cv2.imwrite("results/img8.png",img)
    cv2.imwrite("results/seg_predict8.png",label2color(colors,n_classes,pr))
    cv2.imwrite("results/seg8.png",label2color(colors,n_classes,seg))
    cv2.waitKey(0)








