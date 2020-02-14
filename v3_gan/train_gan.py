import os

import tensorflow as tf

import numpy as np
from datetime import datetime
import time

from dataloader import fetch_image_label,create_dataset
import baseconf as bcf

###################HParams##############################
BATCH_SIZE=16
noise_dim=512
learning_rate=1e-4

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

LAMBDA = 100

#tb callback
run_logdir = "data\\tb_logs"


##############################################################################
from models import make_generator_model,make_discriminator_model

generator = make_generator_model()
discriminator = make_discriminator_model()

# 该方法返回计算交叉熵损失的辅助函数
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# checkpoint and fitlog
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

fit_log = os.path.join(run_logdir,"fit",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(fit_log)

# single train step
@tf.function
def train_step(target, epoch):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(noise, training=True)

    disc_real_output = discriminator([target, target], training=True)
    disc_generated_output = discriminator([target, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

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
  return gen_total_loss,disc_loss
# def generate_and_save_images(model, epoch, test_input):
#     # 注意 training` 设定为 False
#     # 因此，所有层都在推理模式下运行（batchnorm）。
#     #predictions = model(test_input, training=False)
#     #write_images_tb(predictions)


def eveluator(target_ds,true_noise_ds):
  """
  这里只测试，目标图片（随机变换）和固定噪声（即不包含目标的图片）
  """
  result = discriminator([target_ds,true_noise_ds], training=False)
  ev_loss = loss_object(tf.zeros_like(result),result)
  return ev_loss

def train(X_train,X_val,epochs,steps_per_epoch=100,val_steps=10):
  """
  X_train - ds: 只包含目标图片
  X_val - generator: target,noise (包含目标图片和noise)
  """
  
  for epoch in range(epochs):
    start = time.time()

    for i in range(steps_per_epoch):
      image_batch = X_train.take(1)
      gen_loss,disc_loss = train_step(image_batch,epoch)
      if (i+1) % 10 == 0:
        print( "EPOCH-[%s/%s]- GEN LOSS:%.4f , DISC LOSS:%.4f (%s/%s)" % \
                    (epoch,epochs,gen_loss,disc_loss,i,steps_per_epoch))
    
    # TODO: 每个EPOCH测试一次
    # 
    # 

    # 保存一次模型
    if (epoch + 1) % 10 == 0:
      #tf.summary(loss)
      #
      chkfilename = "gan_checkpoints" \
            +"."+datetime.now().strftime("%Y%m%d_%H%M") \
            +( ".epoch-%s.ckpt" % epoch)
      checkpoint_prefix = os.path.join(run_logdir, chkfilename)
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def load_checkpoint():
  pass

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU') 

    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True) 
      assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
      pass

    val_dir = "data\\vals"
    targets_dir = "data\\targets"







    
