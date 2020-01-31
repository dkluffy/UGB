import os

import tensorflow as tf
from tensorflow_addons.losses import TripletSemiHardLoss

import numpy as np
from datetime import datetime

from dataloader import DataGenerator,fetch_image_label,create_dataset,create_dataset_large
import base_cfg as bcfg

def create_model(input_shape=(128,128,3),fine_tune_at=100):

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

def _generate_base_model(filname):
    model = create_model(bcfg.TARGET_SHAPE,fine_tune_at=bcfg.FINE_TUNE_AT)
    model.summary()
    model.save(filname)


#############################params for train#################################
run_logdir = "tb_logs"
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,write_images=True,histogram_freq=5)

logdir = os.path.join(run_logdir,"scalars" , datetime.now().strftime("%Y%m%d-%H%M%S") )
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

def scheduler(epoch):
  learning_rate = bcfg.BASE_LR
  if epoch > 10:
    learning_rate = bcfg.BASE_LR/2
  if epoch > 20:
    learning_rate = bcfg.BASE_LR/10
  if epoch > 30:
    learning_rate = bcfg.BASE_LR * tf.math.exp(0.1 * (10 - epoch))
  #for tensorboard
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

lr_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

def mean_dist(y_true,y_pred):
    #用几张固定图ds,tf.print,返回 acc (y_true.shape[0],)
    pass

def write_images_tb(ds):
    # Sets up a timestamped log directory.
    logdir = os.path.join(run_logdir,"train_data",datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
      for images,_ in ds.take(2):
        tf.summary.image("a batch training data examples", images, max_outputs=100,step=0)

#############################params for train#################################

def train(weights_filename=None,lr=None,initial_epoch=1,
                        val_dir = "vals",
                        targets_dir = "targets",
                        noise_dir = "noise",
                        batch_size = 16,
                        steps_per_epoch = 100,
                        epoch_num = 10):
    
    # load data
    images,labels = fetch_image_label(targets_dir)
    target_ds = create_dataset(images,labels)
    
    images_n,labels_n = fetch_image_label(noise_dir)
    noise_ds = create_dataset_large(images_n,labels_n)

    X_train = tf.data.experimental.sample_from_datasets([target_ds, noise_ds], 
                                                          [0.9, 0.1]).repeat().batch(batch_size)
    
    # write image ds example to tb
    write_images_tb(X_train)

    #load base model from model file
    loss = TripletSemiHardLoss()
    model = tf.keras.models.load_model(bcfg.BASE_MODEL_FILENAME)


    #for checkpoint callbacks
    chkfilename = weights_filename.split(".")[0] \
            +"."+datetime.now().strftime("%Y%m%d_%H%M") \
            +".{epoch:02d}.chkpt.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(chkfilename,save_weights_only=True)

    #for first run 
    if not os.path.exists(weights_filename):
        print("Creating model for first run ...")

        optimizer_first_run = tf.keras.optimizers.Adam(learning_rate=bcfg.BASE_LR/1000)
        model.compile(optimizer_first_run,loss)

        model.summary()

        history = model.fit(X_train,epochs=1, verbose=1,steps_per_epoch=steps_per_epoch)
        model.save_weights(weights_filename)

        return history
    
    # set learning rate / lr callbacks
    cbs = [checkpoint_cb,tensorboard_cb]
    if lr is None:
        cbs = cbs+[lr_cb]
        lr = bcfg.BASE_LR

    #load and compile
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.8)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.load_weights(weights_filename)
    model.compile(optimizer,loss)

    model.summary()
    
    #fire...
    history = model.fit(X_train,epochs=epoch_num, verbose=1,
                  steps_per_epoch=steps_per_epoch,
                  initial_epoch=initial_epoch,
                  callbacks=cbs)

    return history


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU') 

    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True) 
      assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
      pass

    history = train(weights_filename="__weights__\\yys_model_weights.first_run.chkpt.h5",epoch_num=50,initial_epoch=1,lr=bcfg.BASE_LR)

    

    

