import tensorflow as tf 

from tensorflow.keras import layers

import baseconf as bcf

from functools import partial

def downsample(filters, size, apply_batchnorm=True,apply_pool=True):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
  
    if apply_batchnorm:
      result.add(layers.BatchNormalization())
  
    result.add(layers.LeakyReLU())

    if apply_pool:
      result.add(layers.MaxPooling2D())
    
  
    return result

def make_discriminator_model():
    
    inp = layers.Input(shape=[128, 128, 3], name='input_image')
    tar = layers.Input(shape=[128, 128, 3], name='target_image')
  
    x = layers.concatenate([inp, tar]) # (bs, 128, 128, channels*2)
  
    x = downsample(64, 4, False)(x) 
    x = downsample(128, 4)(x) 
    x = downsample(256, 4)(x)
    x = downsample(512, 1,True,False)(x)
    x = layers.Dense(512,activation='selu')(x)
    last = layers.Dense(1,activation='sigmoid')(x)
  
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
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
    inputs = tf.keras.layers.Input(shape=[512])
    up_stack = [
      upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
      upsample(512, 4), # (bs, 16, 16, 1024)
      upsample(256, 4), # (bs, 32, 32, 512)
      upsample(128, 4), # (bs, 64, 64, 256)
      upsample(64, 4), # (bs, 128, 128, 128)
    ]
    x = layers.Reshape((1, 1, 512))(inputs)
    for up in up_stack:
      x = up(x)

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
    
# if __name__ == "__main__":

#     import baseconf

#     model = make_discriminator_model()
#     model.summary()
#     model.save(baseconf.BASE_MODEL_FILENAME)