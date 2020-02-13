import math
import tensorflow as tf 
import random
import numpy as np

from tensorflow.keras.utils import Sequence

import baseconf

from functools import partial

resizer = partial(tf.image.resize_with_pad,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 

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

def preprocess_image(image):
    #image -= np.mean(image,keepdims=True)
    image = tf.image.random_crop(image,baseconf.TARGET_SHAPE)
    if tf.random.uniform(()) > 0.5:
        return im_random_rescale(image,baseconf.RESCAL_RATIO)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return preprocess_image(image)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    #image = tf.image.decode_jpeg(image,channels=3)
    #image /= 255.0
    return image

def create_dataset(images,lables,buffer_size=100):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(lables)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.repeat().shuffle(buffer_size=buffer_size)

import os

def fetch_image_label(fdir,fix_lable=None):
    file_names = os.listdir(fdir)
    image_paths = [os.path.join(fdir,f) for f in file_names]
    if fix_lable is not None:
        labels = [fix_lable] * len(file_names)
    else:
        labels = [ int(x.split(".")[0]) for x in file_names ]

    return image_paths,labels



