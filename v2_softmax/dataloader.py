# %%
import math
import tensorflow as tf 
import random
import numpy as np

from tensorflow.keras.utils import Sequence

import base_cfg

class DataGenerator(Sequence):
    def __init__(self,
                targets_ds,
                noise_ds,
                t_size=128,
                batch_size=16,
                noise_rate=0.2,
                steps_per_epoch=100):
        self.targets_ds = targets_ds
        self.noise_ds = noise_ds
        self.t_size = [t_size,t_size]
        
        self.noise_batch_size = math.ceil(batch_size*noise_rate)
        self.target_batch_size = batch_size - self.noise_batch_size

        self.codings_size = [self.noise_batch_size,t_size,t_size,3]

        self.steps_per_epoch=steps_per_epoch
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self,idx):
        targets_xy = list(self.targets_ds.take(self.target_batch_size))        
        noise_xy = list(self.noise_ds.take(self.noise_batch_size)) 
        xy = targets_xy+noise_xy
        random.shuffle(xy)
        x,y = zip(*xy)
        return tf.convert_to_tensor(x),tf.convert_to_tensor(y)

class DataGeneratorRandomNoise(DataGenerator):
    """
    =。= 感觉这个轮子又白造了，随机的noise可能没用
    """
    def __init__(self,
                targets_ds,
                t_size=128,
                batch_size=16,
                noise_rate=0.5,
                steps_per_epoch=100):
        super.__init__(self)
    
    def __getitem__(self,idx):
        targets_xy = list(self.targets_ds.take(self.target_batch_size))
        
        noise_x = tf.random.uniform(shape=self.codings_size,dtype=tf.float32)
        noise_y = tf.constant([0.]*self.noise_batch_size)
        noise_xy = list(zip(noise_x,noise_y))

        xy = targets_xy+noise_xy
        random.shuffle(xy)
        x,y = zip(*xy)
        return tf.convert_to_tensor(x),tf.convert_to_tensor(y)

from functools import partial

resizer = partial(tf.image.resize_with_pad,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 

# data augmentation:
# tf.image.flip_left_right,tf.image.flip_up_down,
# tf.image.random_jpeg_quality,tf.image.rot90
# skimage.transform.rescale

def im_random_rescale(image,ratio=None,t_size=base_cfg.TARGET_SIZE):
    if ratio is None:
        ratio = tf.random.uniform(())*2
    else:
        ratio = np.random.choice(ratio,1)[0]
    ns = int(t_size*ratio)
    image = tf.image.resize(image,[ns,ns])
    image = tf.image.resize_with_crop_or_pad(image,t_size,t_size)

    return image

def preprocess_image(image):
    image = tf.image.random_crop(image,base_cfg.TARGET_SHAPE)
    #image /= 255.0  # normalize to [0,1] range
    #image -= np.mean(image,keepdims=True)
    if tf.random.uniform(()) > 0.5:
        return im_random_rescale(image,base_cfg.RESCAL_RATIO)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return preprocess_image(image)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
    return image

def create_dataset(images,lables,buffer_size=100):
    """
    直接先把所有图片读入内存
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    images = [load_image(p) for p in images]
    images_zip = zip(list(range(len(images))),images)
    image_ds = tf.data.Dataset.from_generator( lambda: images_zip,(tf.int8,tf.float32))
    image_ds = image_ds.map(lambda i,im: preprocess_image(im),num_parallel_calls=AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(lables)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()

def create_dataset_large(images,lables,buffer_size=100):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(lables)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()

import os

def fetch_image_label(fdir,fix_lable=None):
    file_names = os.listdir(fdir)
    image_paths = [os.path.join(fdir,f) for f in file_names]
    if fix_lable is not None:
        labels = [fix_lable] * len(file_names)
    else:
        labels = [ int(x.split(".")[0])*1.0 for x in file_names ]

    return image_paths,labels

######################################




