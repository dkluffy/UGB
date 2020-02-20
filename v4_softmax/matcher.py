import tensorflow as tf

import base_cfg as bcfg

import numpy as np

from utils import im_slice_with_label

import json
import cv2
from functools import partial

#TODO: improve resizer, if img.size <target.size ; create a board of target.size,put img on board
resizer = partial(tf.image.resize_with_pad,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def load_model(weights_filename):
    model = tf.keras.models.load_model(bcfg.BASE_MODEL_FILENAME)
    model.load_weights(weights_filename)
    return model

def matcher(src_img_tiles,template_code,model,threadhold=0.6,max_only=True):
    """
    src_img_tiles : shape like (None,128,128,3)
    template_code : shaple (256,)
    """
    src_codes = model.predict(src_img_tiles)
    distance = np.array([ np.linalg.norm(x-template_code,ord=2) for x in src_codes ])
    if max_only:
        max_ind = np.argmin(distance)
        if distance[max_ind] < threadhold:
            return max_ind
        else:
            return None
    
    mask = distance > threadhold
    return mask

#def preprocess pipeline
    #img1 = cv2.imread(image_path, 1)
    #img = img1[...,::-1]
    #img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)

from datetime import datetime
import os
run_logdir = "tb_logs"
def write_images_tb(images):
    # Sets up a timestamped log directory.
    logdir = os.path.join(run_logdir,"test_data",datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
      tf.summary.image("Test data examples", images, max_outputs=10000,step=0)

def encode_with_labels(src,label_rect,model,target_size=128):
    
    img_tiles,labels = im_slice_with_label(src,(target_size,target_size),label_rect=label_rect)
    
    # reshape img_tiles,labels to 1-D
    l_size = labels.shape[0]*labels.shape[1]
    labels = np.reshape(labels,(l_size,1))
    img_tiles = np.reshape(img_tiles,(l_size,target_size,target_size,3))

    write_images_tb(img_tiles)
    
    y_pred = model.predict(img_tiles)
    #y_pred = model.predict_on_batch(img_tiles)
    #labels = labels>0
    labels = labels*10
    labels = labels.astype(int)

    return labels,y_pred

def _save_tsv(labels,y_pred,vecs_file="vecs.tsv",meta_file="meta.tsv"):
    np.savetxt(vecs_file,y_pred,delimiter="\t")
    np.savetxt(meta_file,labels,delimiter="\t")

######################
from utils import parsejson,get_rect_by_filename
import cv2

def test_on_img2():

    src_file = "full04.png"
    src = cv2.imread("trash\\"+src_file)/255.0

    jsonfile = "trash\\yys_test_01.json"
    jdict = parsejson(jsonfile)
    label_rect = get_rect_by_filename(jdict,src_file)

    x,y,h,w = label_rect
    label_rect2 = (x-64,y-64,h,w)
    src2 = src[64:,64:]
    
    # print(label_rect2)
    # _,labels2 = im_slice_with_label(src2,(128,128),label_rect2)
    # print(label_rect2,labels2)
    
    model = load_model("__weights__\\yys_model_weights.20200201_1304.100.chkpt.h5")


    labels,y_pred = encode_with_labels(src,label_rect,model)
    labels2,y_pred2 = encode_with_labels(src2,label_rect2,model)

    target_file = "targets\\1.0.jpeg"
    target = cv2.imread(target_file)/255.0
    target = resizer(target,128,128)
    t_pred = model.predict(np.expand_dims(target, axis=0))
    
    labels = np.concatenate((labels,labels2))
    y_pred = np.concatenate((y_pred,y_pred2))

    distance = np.array([ np.linalg.norm(x-t_pred,ord=2) for x in y_pred ])

    print(distance)
    print("------------------")
    print("----- <0.6",distance[distance<0.6])

    _save_tsv(labels,y_pred)

    return distance,labels



if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 

    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True) 
      assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
      pass

    dist,lables = test_on_img2()
    from utils import savepickle
    savepickle("dist",dist,lables)