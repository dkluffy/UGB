import math
import tensorflow as tf 
import random
import numpy as np

from tensorflow.keras.utils import Sequence

import baseconf

from functools import partial

resize_with_pad = partial(tf.image.resize_with_pad,
                        target_height=baseconf.TARGET_SIZE,
                        target_width=baseconf.TARGET_SIZE,
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 

AUTOTUNE = tf.data.experimental.AUTOTUNE

def im_random_rescale(image,ratio=None,t_size=baseconf.TARGET_SIZE):
    """
    BUG(tf.image.resize):tf.image.resize不能处理batch，必须放在tf.image.random_crop之后
    """
    if ratio is None:
        ratio = tf.random.uniform(())*2
    else:
        ratio = np.random.choice(ratio,1)[0]
    ns = int(t_size*ratio)
    image = tf.image.resize(image,[ns,ns])
    image = tf.image.resize_with_crop_or_pad(image,t_size,t_size)

    return image


def im_rescale(image,ratio=None,t_size=baseconf.TARGET_SIZE):
    """
    ratio - 数组
    return:
        images list
    """
    im_list = []
    for rt in ratio:
        ns = int(t_size*rt)
        tmp_im = image
        tmp_im = tf.image.resize(tmp_im,[ns,ns])
        tmp_im = tf.image.resize_with_crop_or_pad(tmp_im,t_size,t_size)
        im_list.append(tmp_im)
    return im_list

def preprocess_image_with_scale(image):
    #image -= np.mean(image,keepdims=True)
    image = tf.image.random_crop(image,baseconf.TARGET_SHAPE)
    if tf.random.uniform(()) > 0.5:
        return im_random_rescale(image,baseconf.RESCAL_RATIO)
    return image

def preprocess_image(image):
    return tf.image.random_crop(image,baseconf.TARGET_SHAPE)
    
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return preprocess_image(image)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return image

def load_test_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return resize_with_pad(image,baseconf.TARGET_SIZE,baseconf.TARGET_SIZE)

def create_dataset(images,lables,im_loader=load_and_preprocess_image,buffer_size=128):
    path_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = path_ds.map(im_loader, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(lables)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()


def create_ds_temp(images,lables,buffer_size=128):
    m_size=baseconf.MID_SIZE
 
    img_list = []
    for im in images:
        im = load_image(im)
        img_list.append(resize_with_pad(im,m_size,m_size))
        

    assert len(img_list) == len(lables),"img_list,new_lables is not match!"

    image_ds = tf.data.Dataset.from_tensor_slices(img_list)
    label_ds = tf.data.Dataset.from_tensor_slices(lables)

    image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()

def create_dataset_in_mem(images,lables,gan=False,buffer_size=128):
    ratio = bcf.RESCAL_RATIO
    m_size=bcf.MID_SIZE
    
    copys_len = len(ratio)
    
    img_list = []
    for im in images:
        im = load_image(im)
        tmp_list=im_rescale(im,ratio=ratio,t_size=m_size)
        img_list+= [tf.image.resize_with_pad(img,m_size,m_size) for img in tmp_list]
        
    new_lables = []
    for lb in lables:
        new_lables+=[lb]*copys_len

    assert len(img_list) == len(new_lables),"img_list,new_lables is not match!"

    image_ds = tf.data.Dataset.from_tensor_slices(img_list)
    label_ds = tf.data.Dataset.from_tensor_slices(new_lables)

    image_ds = image_ds.map(preprocess_image)


    if gan:
        new_images = []
        for imgp in images:
            new_images+=[imgp]*copys_len
        assert len(img_list) == len(new_images),"img_list,new_images is not match!"
        path_ds = tf.data.Dataset.from_tensor_slices(new_images)
        img_orig_ds = path_ds.map(load_image)
        img_orig_ds = img_orig_ds.map(resize_with_pad)
        img_lab_ds = tf.data.Dataset.zip((img_orig_ds,image_ds,label_ds))
    else:
        img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()
import os

def fetch_image_label(fdir,fix_lable=None):
    file_names = os.listdir(fdir)
    image_paths = [os.path.join(fdir,f) for f in file_names]
    if fix_lable is not None:
        labels = [fix_lable] * len(file_names)
    else:
        labels = [ int(x.split(".")[0]) for x in file_names ]

    return image_paths,labels



