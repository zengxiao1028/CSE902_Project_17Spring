import itertools
import os
import numpy as np
import tensorflow as tf
import random
import project_config
def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_png(file_contents, channels=1)
    return img, label


def preprocess(img):
    image = tf.cast(img, tf.float32)
    image = tf.image.resize_images(image, [250, 250])
    image = tf.random_crop(image, [project_config.IMG_SIZE,project_config.IMG_SIZE,1])

    distorted_image = tf.image.random_brightness(image,max_delta=63)

    image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

    #Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_whitening(distorted_image)

    return image


def get_batch(imgs_folder, batch_size):
    files = os.listdir(imgs_folder)
    random.seed(1024)
    random.shuffle(files)
    split_point = int(len(files)/10)

    X_train = files[:split_point]
    X_test = files[split_point:]

    y_train = [int(each.split('_')[3][0]) for each in X_train]
    y_test = [int(each.split('_')[3][0]) for each in X_test]

    # Reads pfathes of images together with their labels
    X_train_list = [os.path.join(imgs_folder, each) for each in X_train]
    X_test_list  = [os.path.join(imgs_folder, each) for each in X_test]


    #training sampels
    images_train = tf.convert_to_tensor(X_train_list, dtype=tf.string)
    labels_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    # Makes an input queue
    input_queue_train = tf.train.slice_input_producer([images_train, labels_train], shuffle=True)
    image_train, label_train = read_images_from_disk(input_queue_train)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image_train = preprocess(image_train)

    # Optional Image and Label Batching
    image_batch_train, label_batch_train = tf.train.batch([image_train, label_train],
                                              batch_size=batch_size,capacity=512)

    #testing sampels
    images_test = tf.convert_to_tensor(X_test_list, dtype=tf.string)
    labels_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    # Makes an input queue
    input_queue_test = tf.train.slice_input_producer([images_test, labels_test], shuffle=True)
    image_test, label_test = read_images_from_disk(input_queue_test)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image_test = preprocess(image_test)

    # Optional Image and Label Batching
    image_batch_test, label_batch_test = tf.train.batch([image_test, label_test],
                                              batch_size=batch_size,capacity=256)

    #test samples
    return image_batch_train, label_batch_train, image_batch_test, label_batch_test



if __name__ == '__main__':

    get_batch(project_config.DES_FOLDER,64)