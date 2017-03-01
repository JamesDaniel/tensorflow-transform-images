# REFERENCE http://stackoverflow.com/questions/34783030/saving-image-files-in-tensorflow

import tensorflow as tf
import numpy as np
import os
from PIL import Image

cur_dir = os.getcwd()
print("resizing images")
print("current directory:", cur_dir)


def modify_image(image):
    resized = tf.image.resize_images(image, [180, 180], 1)
    resized.set_shape([180, 180, 3])
    flipped_images = tf.image.flip_up_down(resized)
    return flipped_images


def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return image


def inputs():
    filenames = ['images/05_00.jpg']
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return reshaped_image

with tf.Graph().as_default():
    image = inputs()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for i in xrange(1):
        img = sess.run(image)
        img = Image.fromarray(img, "RGB")
        img.save(os.path.join(cur_dir, "foo"+str(i)+".jpg"))
