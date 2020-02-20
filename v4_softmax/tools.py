import tensorflow as tf

# viz for images
def write_images_ds(ds,name,summary_writer):
    with summary_writer.as_default():
        for images,_ in ds.take(2):
            tf.summary.image(name, images, max_outputs=100,step=0)
