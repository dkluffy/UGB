import tensorflow as tf 

from tensorflow.keras import layers

import baseconf as bcf

from functools import partial

def downsample(filters, size, strides=2,apply_batchnorm=False,apply_pool=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False))
  
    if apply_batchnorm:
      result.add(layers.BatchNormalization())
  
    result.add(layers.ReLU())

    if apply_pool:
      result.add(layers.MaxPooling2D())

    return result


def make_discriminator_model():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = layers.Input(shape=[128, 128, 3], name='input_image')
    tar = layers.Input(shape=[128, 128, 3], name='target_image')
  
    x = layers.concatenate([inp, tar]) # (bs, 128, 128, channels*2)
  
    x = downsample(64, 4, 1,True)(inp) 
    x = downsample(128, 4,1)(x) 
    x = downsample(256, 4,1)(x)
    x = downsample(512, 4,1,False,False)(x)
    x = downsample(512, 4,1,False,False)(x)
    x = downsample(512, 4,1,False,False)(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024,activation='relu',kernel_initializer=initializer)(x)
    x = layers.Dense(512,activation='relu',kernel_initializer=initializer)(x)
    #x = layers.BatchNormalization()(x)
    last = layers.Dense(20,activation='sigmoid')(x)
  
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


DefaultConv2D = partial(layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    
def make_ResNet34():

    """
    Use ResNet34 instead of the orig

    """ 
    initializer = tf.keras.initializers.he_uniform()
    #CNN network
    inp = layers.Input(shape=[128, 128, 3], name='input_image')
    X = DefaultConv2D(64, kernel_size=7, strides=2)(inp)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(X)
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        X = ResidualUnit(filters, strides=strides)(X)
        prev_filters = filters
    X = layers.GlobalAvgPool2D()(X)
    x = layers.Flatten()(X)
    x = layers.Dense(1024,activation='relu',kernel_initializer=initializer)(x)
    #x = layers.BatchNormalization()(x)
    last = layers.Dense(64,activation='softmax')(x)

    return tf.keras.Model(inputs=inp, outputs=last)

if __name__ == "__main__":

#     import baseconf
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU') 

    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True) 
      assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
      pass
    model = make_discriminator_model()
    model.summary()
    #g=make_generator_model()
    #g.summary()
#     model.save(baseconf.BASE_MODEL_FILENAME)