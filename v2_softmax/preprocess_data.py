import tensorflow as tf
import os

from dataloader import fetch_image_label,resizer

def im_pad_and_save(save_path,im_dir,label,t_size):
    """
    辅助用，扩大图像并保存,
    """
    image = tf.io.read_file(im_dir)
    image = tf.image.decode_image(image,channels=3,dtype=tf.uint8)
    h,w = (image.shape[0]+t_size,image.shape[1]+t_size)
    image = resizer(image,h,w)
    c = tf.io.encode_jpeg(image)
    filename = "".join([str(label),".jpeg"])
    img_path = os.path.join(save_path,filename)
    tf.io.write_file(img_path,c)

def im_random_crop_save(save_path,im_dir,label,n=100,t_size=192):
    """
    主要用于生成NOISE，固定noise标签
    """
    image = tf.io.read_file(im_dir)
    image = tf.image.decode_image(image,channels=3,dtype=tf.uint8)
    for i in range(1,n):
        im = tf.image.random_crop(image,[t_size,t_size,3])
        im = tf.io.encode_jpeg(im)
        new_label = label +i
        img_path = os.path.join(save_path,"".join([str(new_label),".jpeg"]))
        tf.io.write_file(img_path,im)
if __name__ == "__main__":
    print(__name__)

    # 读取截图获得noise，目的是固定一定的noise，标签
    # img_p,labels = fetch_image_label("noise")
    # sp = "noise_crops"
    # for i,b in zip(img_p,labels):
    #     im_random_crop_save(sp,i,b,1000)

    # 预处理 目标图片
    # imgs,labs = fetch_image_label("org_targets")
    # [ im_pad_and_save("targets",im,b,128) for im,b in zip(imgs,labs) ]