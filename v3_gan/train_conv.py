import os

import tensorflow as tf

import numpy as np
from datetime import datetime

from dataloader import fetch_image_label,create_dataset
import baseconf as bcf


#############################dataset base config#############################
val_dir = "data\\vals"
targets_dir = "data\\targets"
noise_dir = "data\\noise"

#############################tensorboard / callbacks#################################

#tb callback
run_logdir = "data\\tb_logs"
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,histogram_freq=5)
logdir = os.path.join(run_logdir,"scalars" , datetime.now().strftime("%Y%m%d-%H%M%S") )

# lr callback
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

#for checkpoint callbacks
chkfilename = "data\\weights\\model_checkpoit_weights_only" \
            +"."+datetime.now().strftime("%Y%m%d_%H%M") \
            +".{epoch:02d}.ckpt"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(chkfilename,save_weights_only=True,period=10)

# viz for images
def write_images_tb(images):
    # Sets up a timestamped log directory.
    img_logdir = os.path.join(run_logdir,"train_data",datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(img_logdir)
    with file_writer.as_default():
      tf.summary.image("batch of training data examples", images, max_outputs=100,step=0)

##############################################################################
callbacks_list = [checkpoint_cb,tensorboard_cb]

def load_model(weights_filename=None,lr=None):
    #load base model from model file
    model = tf.keras.models.load_model(bcf.BASE_MODEL_FILENAME)

    # set learning rate / lr callbacks
    if lr is None:
        callbacks_list.append(lr_cb)
        lr = bcf.BASE_LR

    #load and compile
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.8)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if os.path.exists(weights_filename):
       model.load_weights(weights_filename)
    model.compile(optimizer,loss,metrics=['accuracy'])

    return model

def train(model,initial_epoch=1,batch_size=16,
                        steps_per_epoch=100,
                        epoch_num=10):
    
    # load data
    images,labels = fetch_image_label(targets_dir)
    target_ds = create_dataset(images,labels)
    #X_train = DataGenerator(target_ds,batch_size=batch_size,steps_per_epoch=steps_per_epoch)
    X_train = target_ds.batch(batch_size)

    val_ds = create_dataset(images,labels)
    #X_val = DataGenerator(val_ds,batch_size=batch_size,steps_per_epoch=10)
    X_val = val_ds.batch(batch_size)

    # write image ds example to tb
    # for i in range(10):
    #     img_for_view,_  = X_train[i]
    #     write_images_tb(img_for_view)

    #fire...
    history = model.fit(X_train,
                  validation_data=X_val,validation_steps=10,
                  epochs=epoch_num, verbose=1,
                  steps_per_epoch=steps_per_epoch,
                  initial_epoch=initial_epoch,
                  callbacks=callbacks_list)

    return history


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU') 

    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True) 
      assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
      pass

    model = load_model("data\\weights\\model_disc_only.ckpt")
    history = train(model,initial_epoch=130,epoch_num=150)
