import tensorflow as tf 

from tensorflow.keras import layers

import baseconf as bcf

from functools import partial

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
  
    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())
  
    result.add(tf.keras.layers.LeakyReLU())
  
    return result
  
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
  
    result.add(tf.keras.layers.BatchNormalization())
  
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
  
    result.add(tf.keras.layers.ReLU())
  
    return result
  
def make_generator_model():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
  
    down_stack = [
      downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
      downsample(128, 4), # (bs, 64, 64, 128)
      downsample(256, 4), # (bs, 32, 32, 256)
      downsample(512, 4), # (bs, 16, 16, 512)
      downsample(512, 4), # (bs, 8, 8, 512)
      downsample(512, 4), # (bs, 4, 4, 512)
      downsample(512, 4), # (bs, 2, 2, 512)
      downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
      upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
      upsample(512, 4), # (bs, 16, 16, 1024)
      upsample(256, 4), # (bs, 32, 32, 512)
      upsample(128, 4), # (bs, 64, 64, 256)
      upsample(64, 4), # (bs, 128, 128, 128)
    ]
  
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)
  
    x = inputs
  
    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)
  
    skips = reversed(skips[:-1])
  
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])
  
    x = last(x)
  
    return tf.keras.Model(inputs=inputs, outputs=x)

def make_discriminator_model():
    initializer = tf.random_normal_initializer(0., 0.02)
  
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
  
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
  
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
  
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
  
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
  
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def model_base_on_mobilenet(input_shape=bcf.TARGET_SHAPE,
                            fine_tune_at=bcf.FINE_TUNE_AT,
                            class_num=bcf.CLS_NUM):

    #Model file location: ~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
    #or from internet
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    
    # Freeze base_model weights
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])

    return model

def create_model(base_model,base_filename=None,**kwargs):

    from datetime import datetime
    import os

    #create model
    model = base_model(**kwargs)
    model.summary()

    #save
    if base_filename:
        model.save(base_filename)

if __name__ == "__main__":

    #create_model(model_base_on_mobilenet,base_filename=bcf.BASE_MODEL_FILENAME)
    #create_model(model_base_on_mobilenet,base_filename="model_freeze_all.h5",fine_tune_at=10000)
    create_model(model_base_on_mobilenet,base_filename="model_freeze_none.h5",fine_tune_at=0)
    create_model(model_base_on_mobilenet,base_filename="model_freeze_100.h5",fine_tune_at=100)