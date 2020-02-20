# -*- coding: utf-8 -*-
"""v4_softmax.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NqVkxJMu5pgsMvwHck0Zi8tDFhtZgPX4
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
import time

from dataloader import fetch_image_label,create_dataset,create_dataset_in_mem,create_ds_temp
from tools import write_images_ds
import baseconf as bcf

tf.config.list_physical_devices()

initializer = tf.keras.initializers.he_uniform()
def downsample(filters, size, strides=2,apply_batchnorm=False,apply_pool=True):
    #initializer = tf.random_normal_initializer(0., 0.02)
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
    
    inp = layers.Input(shape=[128, 128, 3], name='input_image')
    
    x = downsample(64, 4, 2,True)(inp) 
    x = downsample(128, 2,1)(x) 
    x = downsample(128, 2,1)(x)
    x = downsample(256, 4,2)(x)
    x = downsample(256, 4,2,False,False)(x)
    x = downsample(512, 1,1,False,False)(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024,activation='relu',kernel_initializer=initializer)(x)
    x = layers.Dense(512,activation='relu',kernel_initializer=initializer)(x)
    #x = layers.BatchNormalization()(x)
    last = layers.Dense(20,activation='softmax')(x)
  
    return tf.keras.Model(inputs=inp, outputs=last)


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


discriminator = make_discriminator_model()
#discriminator = model_base_on_mobilenet()
discriminator.summary()

###################HParams##############################
BATCH_SIZE=8
noise_dim=512
base_learning_rate=0.001
save_interval=5
# Optimizers
# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.SGD(base_learning_rate,0.5)
# lr callback
def lr_scheduler(epoch):
  learning_rate = base_learning_rate
  if epoch > 10:
    learning_rate =base_learning_rate/2
  if epoch > 20:
    learning_rate = base_learning_rate/10
  if epoch > 30:
    learning_rate = base_learning_rate * tf.math.exp(0.1 * (10 - epoch))
  #for tensorboard
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  print("current lr:",learning_rate)
  return learning_rate

LAMBDA = 10

#tb callback
run_logdir = "/content/drive/My Drive/ugb"


##############################################################################



# softmax loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def discriminator_loss(y_true,y_pred):
  return loss_object(y_true,y_pred)

# checkpoint and fitlog
checkpoint = tf.train.Checkpoint(discriminator=discriminator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, run_logdir+"/v4", max_to_keep=3)

fit_log = os.path.join(run_logdir,"fit_v4",datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(fit_log)

# single train step
@tf.function
def train_step(target,y_true, epoch):

  with tf.GradientTape() as disc_tape:
 
    y_pred = discriminator(target, training=True)
    disc_loss = discriminator_loss(y_true, y_pred)

  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
  return disc_loss
# def generate_and_save_images(model, epoch, test_input):
#     # 注意 training` 设定为 False
#     # 因此，所有层都在推理模式下运行（batchnorm）。
#     #predictions = model(test_input, training=False)
#     #write_images_tb(predictions)

val_metric = tf.keras.metrics.SparseCategoricalAccuracy()
def eveluator(x,y_true):
  """
  这里只测试，目标图片（随机变换）和真实随机噪声（即不包含目标的图片）
  """
  result = discriminator(x, training=False)
  ev_loss = loss_object(y_true,result)
  val_metric.update_state(y_true,result)
  return ev_loss,val_metric.result().numpy()

def train(X_train,X_noise,X_val,epochs,init_epoch=1,steps_per_epoch=100,val_steps=10):
  """
  X_train - ds: 只包含目标图片
  X_val - ds: 只包含noise
  """
  checkpoint.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  
  print("Fire to Train....")
  for epoch in range(init_epoch,epochs):
    
    start = time.time()
    #discriminator_optimizer = tf.keras.optimizers.Adam(lr_scheduler(epoch))
    for step in range(1,steps_per_epoch):
      for x,y in X_train.take(1):
        disc_loss = train_step(x,y,epoch)
        if step % 50 == 0:
          print( "EPOCH-[%s/%s]- DISC LOSS:%.4f (%s/%s) " % \
                    (epoch,epochs,disc_loss,step,steps_per_epoch))
      # for x,y in X_noise.take(1):
      #   disc_loss = train_step(x,y,epoch)
      #   if step % 50 == 0:
      #     print( "noise::>> EPOCH-[%s/%s]- DISC LOSS:%.4f (%s/%s) " % \
      #               (epoch,epochs,disc_loss,step,steps_per_epoch))
    # eva
    for x,y in X_val.take(1):
      val_loss,val_acc = eveluator(x,y)
      print( "EPOCH-[%s/%s]-  DISC LOSS:%.4f ,val_loss:%.4f,acc:%.4f , (%s/%s) " % \
                    (epoch,epochs,disc_loss,val_loss,val_acc,step,steps_per_epoch))
    with summary_writer.as_default():
      tf.summary.scalar('val_loss', val_loss, step=epoch)
      tf.summary.scalar('val_acc', val_acc, step=epoch)

    # 保存一次模型
    if epoch % save_interval == 0:
      save_path = ckpt_manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
      # chkfilename = "gan_checkpoints" \
      #       +"."+datetime.now().strftime("%Y%m%d_%H%M") \
      #       +( ".epoch-%s.ckpt" % epoch)
      # checkpoint_prefix = os.path.join(run_logdir, chkfilename)
      # checkpoint.save(file_prefix = checkpoint_prefix)

    # epoch end
    print ('Time for epoch {} is {:.2f} sec'.format(epoch, time.time()-start))


# %%

val_dir = run_logdir+"/noise"
targets_dir = run_logdir+"/targets"
tg_imgs,tg_labels = fetch_image_label(targets_dir)
noise_imgs,noise_labels = fetch_image_label(val_dir,0)
X_train = create_dataset_in_mem(tg_imgs,tg_labels)#.batch(BATCH_SIZE)
X_val = create_ds_temp(tg_imgs,tg_labels).batch(BATCH_SIZE)
X_noise = create_dataset(noise_imgs,noise_labels)#.batch(BATCH_SIZE)

# new_train = X_train.concatenate(X_noise) - 不起作用
new_train = tf.data.experimental.sample_from_datasets([X_train,X_noise],[0.9,0.1]).batch(BATCH_SIZE)


from vizer import plot_multiple_images

for x,y in new_train.take(2):
  print(y)
  plot_multiple_images(x,4)

train(new_train,X_noise,X_val,100,1)

#checkpoint.restore(ckpt_manager.latest_checkpoint)

X_val = tf.data.Dataset.zip((X_train,X_noise))
for t,n in X_val.take(10):
  tx,ty = t
  nx,ny = n
  x = tf.concat([tx,nx],0)
  y_true = tf.concat([ty,ny],0)
  y_pred = discriminator(x,training=False)
  val_metric.update_state(y_true,y_pred)
  print(val_metric.result().numpy())

for x,y_true in X_train.take(10):
  y_pred = discriminator(x,training=False)
  val_metric.update_state(y_true,y_pred)
  print(val_metric.result().numpy())