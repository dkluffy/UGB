import os

import tensorflow as tf
from tensorflow_addons.losses import TripletSemiHardLoss

import numpy as np
from datetime import datetime

from dataloader import DataGenerator,fetch_image_label,create_dataset,create_dataset_large
import baseconf as bcf


#############################dataset base config#############################
val_dir = "data\\vals"
targets_dir = "data\\targets"
noise_dir = "data\\noise"

#############################tensorboard / callbacks#################################
run_logdir = "data\\tb_logs"
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,write_images=True,histogram_freq=5)
logdir = os.path.join(run_logdir,"scalars" , datetime.now().strftime("%Y%m%d-%H%M%S") )

def scheduler(epoch):
  learning_rate = bcf.BASE_LR
  if epoch > 10:
    learning_rate = bcf.BASE_LR/2
  if epoch > 20:
    learning_rate = bcf.BASE_LR/10
  if epoch > 30:
    learning_rate = bcf.BASE_LR * tf.math.exp(0.1 * (10 - epoch))
  #for tensorboard
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

lr_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

def write_images_tb(images):
    # Sets up a timestamped log directory.
    logdir = os.path.join(run_logdir,"train_data",datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
      tf.summary.image("batch of training data examples", images, max_outputs=100,step=0)

##############################################################################
def mean_dist(y_true,y_pred):
    #用几张固定图ds,tf.print,返回 acc (y_true.shape[0],)
    pass

def train(weights_filename=None,lr=None,
                        initial_epoch=1,
                        batch_size = 16,
                        steps_per_epoch = 100,
                        epoch_num = 10):
    
    # load data
    images,labels = fetch_image_label(targets_dir)
    target_ds = create_dataset(images,labels)
    
    images_n,labels_n = fetch_image_label(noise_dir)
    noise_ds = create_dataset_large(images_n,labels_n)

    X_train = DataGenerator(target_ds,noise_ds,batch_size=batch_size,steps_per_epoch=steps_per_epoch)

    # 这种方法不行
    # X_train = tf.data.experimental\
    #             .sample_from_datasets([target_ds, noise_ds], [0.9, 0.1]).repeat()\
    #                             .shuffle(buffer_size=batch_size*10).batch(batch_size)
    
    # write image ds example to tb
    images_to_view,_ = X_train[0]
    write_images_tb(images_to_view)

    #load base model from model file
    loss = TripletSemiHardLoss()
    model = tf.keras.models.load_model(bcf.BASE_MODEL_FILENAME)

    #for checkpoint callbacks
    chkfilename = weights_filename.split(".")[0] \
            +"."+datetime.now().strftime("%Y%m%d_%H%M") \
            +".{epoch:02d}.chkpt.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(chkfilename,save_weights_only=True)

    #for first run 
    if not os.path.exists(weights_filename):
        print("Creating model for first run ...")

        optimizer_first_run = tf.keras.optimizers.Adam(learning_rate=bcf.BASE_LR/1000)
        model.compile(optimizer_first_run,loss)

        model.summary()

        history = model.fit(X_train,epochs=1, verbose=1,steps_per_epoch=steps_per_epoch)
        model.save_weights(weights_filename)

        return history
    
    # set learning rate / lr callbacks
    cbs = [checkpoint_cb,tensorboard_cb]
    if lr is None:
        cbs = cbs+[lr_cb]
        lr = bcf.BASE_LR

    #load and compile
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.8)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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

    #history = train(weights_filename="__weights__\\yys_model_weights.20200201_0955.50.chkpt.h5",epoch_num=100,initial_epoch=50,lr=0.001/10.,batch_size=32)
    history = train(weights_filename="__weights__\\yys_model_weights.20200201_0955.50.chkpt.h5",epoch_num=100,initial_epoch=50,lr=0.001/10.,batch_size=32)

    

    

