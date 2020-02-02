import tensorflow as tf 

import baseconf as bcf


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