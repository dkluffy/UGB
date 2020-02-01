import tensorflow as tf 

import base_cfg as bcfg



def model_v1(input_shape=(128,128,3),fine_tune_at=100):

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

    encoder_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
    ])

    return encoder_model

def create_model(base_model,save_path = ".",base_filename = "model",is_save=True,**kwargs):

    from datetime import datetime
    import os

    #create model
    model = base_model(**kwargs)
    model.summary()

    #save
    if is_save:
        model_filename = "".join([base_filename, datetime.now().strftime("%Y%m%d-%H%M%S"),".h5"] )
        model_filename = os.path.join(save_path,model_filename)
        model.save(model_filename)

if __name__ == "__main__":
    create_model(model_v1)