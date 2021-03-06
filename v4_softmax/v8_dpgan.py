# -*- coding: utf-8 -*-
"""V8_DPGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oeG7_KGvdbKlXIDIiNr-qD8UBc3lfntO
"""

# Commented out IPython magic to ensure Python compatibility.
# %%
# %tensorflow_version 2.x

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
import time
import baseconf as bcf
import matplotlib.pyplot as plt

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
  
    x = downsample(64, 4, 1,True)(x) 
    x = downsample(128, 4,2)(x) 
    x = downsample(256, 4,2)(x)
    x = downsample(512, 4,1,False,False)(x)
    x = downsample(512, 4,2,False,False)(x)
    x = layers.GlobalAvgPool2D()(x)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Flatten()(x)
    x = layers.Dense(1024,activation='relu',kernel_initializer=initializer)(x)
    #x = layers.Dense(512,activation='relu',kernel_initializer=initializer)(x)
    #x = layers.BatchNormalization()(x)
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

generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()

###################HParams##############################
BATCH_SIZE=4
noise_dim=512
base_learning_rate=0.01
save_interval=5
tf.random.set_seed(0)
#tb callback
run_logdir = "/content/drive/My Drive/ugb"
version = "/v8"
generator_optimizer = tf.keras.optimizers.SGD(base_learning_rate,0.3)
discriminator_optimizer = tf.keras.optimizers.SGD(base_learning_rate,0.3)
# lr callback
def lr_scheduler(epoch):
  learning_rate = base_learning_rate
  if epoch > 10:
    learning_rate =base_learning_rate/2
  if epoch > 20:
    learning_rate = base_learning_rate/10
  if epoch > 40:
    learning_rate = base_learning_rate * tf.math.exp(0.1 * (10 - epoch))
  #for tensorboard
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  tf.print("**Current Learning_rate: ",learning_rate)
  return learning_rate

LAMBDA = 100

# 该方法返回计算交叉熵损失的辅助函数
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss,real_loss

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# checkpoint and fitlog
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, run_logdir+version, max_to_keep=3)

fit_log = os.path.join(run_logdir+version+"_fit",datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(fit_log)

# single train step
@tf.function
def train_step(orirg_img,target, epoch):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(noise, training=True)

    disc_real_output = discriminator([target, orirg_img], training=True)
    disc_generated_output = discriminator([target, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss,real_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
  
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
  return gen_total_loss,disc_loss,real_loss



mse = tf.keras.losses.MeanSquaredError()
def eveluator(orig_img,target):
  """
  这里只测试，原图和变换过的图片
  """
  result = discriminator([target,orig_img], training=False)
  ev_loss = mse(tf.ones_like(result),result)
  return ev_loss

gimg_list = []
def train(X_train,epochs,init_epoch=1,steps_per_epoch=100,val_steps=10):
  """
  X_train - ds: 包含目标图片，变换过的图片，标签
  """
  checkpoint.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  
  print("Fire to Train....")
  for epoch in range(init_epoch,epochs+1):
    
    start = time.time()

    generator_optimizer = tf.keras.optimizers.SGD(lr_scheduler(epoch))
    discriminator_optimizer = tf.keras.optimizers.SGD(lr_scheduler(epoch))

    for step in range(1,steps_per_epoch+1):
      for orig_img,target,_ in X_train.take(1):
        gen_loss,disc_loss,real_loss = train_step(orig_img,target,epoch)
        if step % 50 == 0:
          print( "EPOCH-[%s/%s]- GEN LOSS:%.4f , DISC LOSS:%.4f / %.4f  (%s/%s) " % \
                    (epoch,epochs,gen_loss,disc_loss,real_loss,step,steps_per_epoch))

    print("Testing....")
    # 每个EPOCH测试一次
    val_losses = []
    for _ in range(val_steps):
      for orig_img,target,_ in X_train.take(1):
        val_losses.append(eveluator(orig_img,target)) 

    tnoise = tf.random.normal([1, noise_dim])
    gimg_list.append(generator(tnoise,training=False)[0]*0.5+0.5)

    with summary_writer.as_default():
      val_loss = tf.reduce_mean(val_losses)
      print( "EPOCH-[%s/%s]- GEN LOSS:%.4f , DISC LOSS:%.4f ,val_loss:%.4f" % \
                    (epoch,epochs,gen_loss,disc_loss,val_loss))
      tf.summary.scalar('val_loss', val_loss, step=epoch)


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
    print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time()-start))


# %%

from dataloader import load_image,preprocess_image,resize_with_pad,im_rescale,load_and_preprocess_image
def create_dataset_in_mem(images,lables,gan=False,buffer_size=128):
    ratio = bcf.RESCAL_RATIO
    m_size=bcf.MID_SIZE
    
    copys_len = len(ratio)
    
    img_list = []
    for im in images:
        im = load_image(im)
        tmp_list=im_rescale(im,ratio=ratio,t_size=m_size)
        img_list+= [tf.image.resize_with_pad(img,m_size,m_size) for img in tmp_list]
        
    new_lables = []
    for lb in lables:
        new_lables+=[lb]*copys_len

    assert len(img_list) == len(new_lables),"img_list,new_lables is not match!"

    image_ds = tf.data.Dataset.from_tensor_slices(img_list)
    label_ds = tf.data.Dataset.from_tensor_slices(new_lables)

    image_ds = image_ds.map(preprocess_image)


    if gan:
        new_images = []
        for imgp in images:
            new_images+=[imgp]*copys_len
        assert len(img_list) == len(new_images),"img_list,new_images is not match!"
        path_ds = tf.data.Dataset.from_tensor_slices(new_images)
        img_orig_ds = path_ds.map(load_image)
        img_orig_ds = img_orig_ds.map(resize_with_pad)
        img_lab_ds = tf.data.Dataset.zip((img_orig_ds,image_ds,label_ds))
    else:
        img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat()

from dataloader import fetch_image_label,create_dataset
run_logdir = "/content/drive/My Drive/ugb"
val_dir = run_logdir+"/noise"
targets_dir = run_logdir+"/targets"
tg_imgs,tg_labels = fetch_image_label(targets_dir)
val_imgs,val_labels = fetch_image_label(val_dir,0)
X_train = create_dataset_in_mem(tg_imgs,tg_labels,True).batch(BATCH_SIZE)
X_val = create_dataset(val_imgs,val_labels).batch(BATCH_SIZE)

from vizer import plot_multiple_images
for img,timg,lb in X_train.take(10):
  print(lb)
  plot_multiple_images(tf.stack([img[0],timg[0]]),2)

train(X_train,100,11)

def plot_multiple_images2(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    # if images.shape[-1] == 1:
    #     images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
plot_multiple_images2(gimg_list,4)

for org,t,_ in X_train.take(1):
  org_img = org[0]
  timg = t[0]
plt.imshow(org_img)

discriminator([tf.expand_dims(org_img,0),tf.expand_dims(timg,0)],training=False)

discriminator([tf.expand_dims(org_img,0),tf.expand_dims(org_img,0)],training=False)

nimgs = []
for o,nim,_ in X_train.take(10):
  nimgs.append(nim[0])
plot_multiple_images2(nimgs,8)

test_loss =[]
for im in nimgs:
  test_loss.append(discriminator([tf.expand_dims(org_img,0),tf.expand_dims(im,0)],training=False))
print([t.numpy()[0,0] for t in test_loss],mse(tf.zeros_like(test_loss),test_loss))

test_loss[tf.argmax(test_loss).numpy()[0,0]],tf.argmax(test_loss)

nimgs2 = []
for nim,_ in X_val.take(10):
  nimgs2.append(nim[0])
plot_multiple_images2(nimgs2,8)

test_loss =[]
for im in nimgs2:
  test_loss.append(discriminator([tf.expand_dims(org_img,0),tf.expand_dims(im,0)],training=False))
print([t.numpy()[0,0] for t in test_loss],mse(tf.zeros_like(test_loss),test_loss))
test_loss[tf.argmax(test_loss).numpy()[0,0]],tf.argmax(test_loss)

"""# eveluat"""

# %load_ext tensorboard
# %tensorboard --bind_all  --logdir=/content/drive/My\ Drive/ugb/v8_fit --port=6007