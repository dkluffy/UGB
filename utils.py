import matplotlib.pyplot as plt

import numpy as np
import random
import math


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

def im_slicer(im,image_size):
    W = image_size[0]
    H = image_size[1]
    tiles = [im[x:x+W,y:y+H] for x in range(0,im.shape[0],W) for y in range(0,im.shape[1],H)]
    tiles = np.array(tiles)
    return tiles

def calc_iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x2,box2_x2)
    yi1 = max(box1_y2,box2_y2)
    xi2 = min(box1_x1,box2_x1)
    yi2 = min(box1_y1,box2_y1)
    
    inter_width = (box1_x2-box1_x1+box2_x2-box2_x1)-(xi1-xi2)
    inter_height = (box1_y2-box1_y1+box2_y2-box2_y1)-(yi1-yi2)
    inter_area = max(inter_width,0)*max(inter_height,0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)


    box1_area = (box1_y2-box1_y1)*(box1_x2-box1_x1)
    #box2_area = (box2_y2-box2_y1)*(box2_x2-box2_x1)
    #union_area = box1_area+box2_area-inter_area

    # compute the IoU

    #iou = inter_area/union_area

    #截取的方块，在标注范围内的比例
    iou_ = inter_area/box1_area
    if iou_>0.5:
        s = "found big iou: %s @" % iou_
        print(s ,box2)
  
    
    return iou_


def im_slice_with_label(im,image_size,label_rect=None,channels=3):
    """
    Args：
        im (h,w,c)- 图片数组
        image_size:(128,128) - 目标图片大小
        label_rect:x,y,h,w - 标记的区域
        **注意：TF读取的图片格式是(h,w,c)
    """
    if label_rect:
        x_,y_,h_,w_=label_rect
        box = (x_,y_,x_+w_,y_+h_)
    H = image_size[0]
    W = image_size[1]
    outy,outx = math.ceil(im.shape[0]//H),math.ceil(im.shape[1]//W)
    lables = np.zeros((outx,outy))
    tiles = np.zeros((outx,outy,H,W,channels))
    for x in range(0,outx):
        for y in range(0,outy):
            tile = im[y*H:(y+1)*H,x*W:(x+1)*W]
            tiles[x,y]=tile
            
            if label_rect:
                iou = calc_iou(box,(x*W,y*H,(x+1)*W,(y+1)*H))
                lables[x,y] = iou
    
    return tiles,lables

def im_crop_by_center(im,cx,cy,image_size):
    """
    cx,cy:中心坐标
    """
    h,w = image_size
    return im[cx-w//2:cx+w//2,cy-h//2:cy+h//2]

def im_crop_in_area(im,area,image_size):
    """
    area应该在im中心，并且面积远小于im
    return:
       返回裁剪后的图片，用于tf.random_crop，使得其可以始终截取包含中心点的切片    
    """
    cx,cy,ch,cw = area
    h,w = image_size
    #TODO: pad im, if (cx**2+cy**2)<=(h**2+w**2)
    im = im[cx-cw//2-w//2:cx+cw//2+w//2,cy-ch//2-h//2:cy+ch//2+h//2]
    return im



def im_slicer_after_crop(im,image_size,k=1):
    """
    #备用，APP实际使用的时候，通过多次截图来匹配
    #把 im 切成N个image_size大小的小块
    Args:
        im - image array
        image_size - target size
        k - int, >0
    """
    if k < 1:
        k = random.random() 
    ws = int(k*image_size[0]//2)
    hs = int(k*image_size[1]//2)
    return im_slicer(im[ws:-ws,hs:-hs],image_size)

import json
def parsejson(jsonfile):
    with open(jsonfile) as f:
        prjoson = json.load(f)
    return prjoson["_via_img_metadata"]

def get_rect_by_filename(jdict,filename):
    for k in jdict:
        if filename in k:
            p = jdict[k]["regions"][0]["shape_attributes"]
            return (p["x"],p["y"],p["height"],p["width"])


import pickle

def savepickle(fname,*args):
    with open(fname+"_pk","wb") as f:
        pickle.dump(args,f)
        
def loadpickle(fname):
    with open(fname,"rb") as f:
        obj = pickle.load(f)
        
    return obj
    