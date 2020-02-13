import os

import tensorflow as tf

import numpy as np
from datetime import datetime
import time

from dataloader import fetch_image_label,create_dataset
import baseconf as bcf

###################HParams##############################
BATCH_SIZE=16
noise_dim=256
learning_rate=1e-4

#############################dataset base config#############################
val_dir = "data\\vals"
targets_dir = "data\\targets"
noise_dir = "data\\noise"

#############################tensorboard / callbacks#################################
#tb callback
run_logdir = "data\\tb_logs"

# viz for images
def write_images_tb(images):
    # Sets up a timestamped log directory.
    logdir = os.path.join(run_logdir,"train_data",datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
      tf.summary.image("batch of training data examples", images, max_outputs=100,step=0)

##############################################################################
from models import make_generator_model,make_discriminator_model

generator = make_generator_model()
discriminator = make_discriminator_model()

# 该方法返回计算交叉熵损失的辅助函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss,disc_loss

# def generate_and_save_images(model, epoch, test_input):
#     # 注意 training` 设定为 False
#     # 因此，所有层都在推理模式下运行（batchnorm）。
#     #predictions = model(test_input, training=False)
#     #write_images_tb(predictions)

def train(dataset, epochs):
  gen_loss_list = []
  disc_loss_list = []
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss,disc_loss = train_step(image_batch)
      gen_loss_list.append(gen_loss)
      disc_loss_list.append(disc_loss)

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

    
