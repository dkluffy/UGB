# %%

import tensorflow as tf 


# %%
from functools import partial
resizer = partial(tf.image.resize_with_pad,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
import os






# %%
from dataloader import fetch_image_label
img_path = "targets"
images,labels = fetch_image_label(img_path)
[ im_pad_and_save(img_path,im,lb,128) for im,lb in zip(images,labels) ]

# %%
