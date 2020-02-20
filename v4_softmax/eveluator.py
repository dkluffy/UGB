# %%
import os

import tensorflow as tf

import numpy as np
from datetime import datetime
import time

from dataloader import fetch_image_label,create_dataset,create_dataset_in_mem
from tools import write_images_ds
from utils import pprint
import baseconf as bcf

# %%
log_dir="data\\colab_log"

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

from models import make_generator_model,make_discriminator_model

generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=3)

# %%

if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
# %%
BATCH_SIZE=8
noise_dim=512
from vizer import plot_multiple_images
# %%
#plot generator

noise = tf.random.normal([BATCH_SIZE, noise_dim])
fake_imgs = generator(noise,training=False)

plot_multiple_images(fake_imgs,4)


# %%

targets_dir = "data\\targets"
noise_dir = "data\\noise_crops"
from dataloader import load_test_image

tg_imgs,tg_labels = fetch_image_label(targets_dir)
noise_imgs,noise_labels = fetch_image_label(noise_dir)


# %%
# test tg vs noise

ind_noise = np.random.choice(list(range(len(noise_labels))),len(tg_labels))

val_noise = [noise_imgs[i] for i in ind_noise]
val_noise_labels = [noise_labels[i] for i in ind_noise]

targets = tf.stack([load_test_image(im) for im in tg_imgs])
val_noise = tf.stack([load_test_image(im) for im in val_noise])

start = time.time()
y_pred = discriminator([targets,val_noise],training=False)
print("total images: %s ,pred time: %.2f" % (len(targets),time.time()-start))
y_true = tf.zeros_like(y_pred)

from sklearn.metrics import mean_squared_error
print(y_pred)
print(mean_squared_error(y_true,y_pred))
# %%


