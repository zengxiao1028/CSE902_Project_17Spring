import tensorflow as tf
from network import network
import numpy as np
from network.gen_sp_data import SPDataGenerator
import project_config
import time
import os
import fingerprint_data.convert_sd4
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train():

    shape = (None, project_config.IMG_SIZE, project_config.IMG_SIZE, 1)
    num_classes = 5

    net = network.FingerNet(shape, num_classes, network='localization')
    dg = SPDataGenerator(project_config.SP_DATA_FOLDER, project_config.SP_LABEL_FOLDER)
    x_train, y_train, x_test, y_test = dg.get_batch(64)

    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(tf.global_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    fig = plt.figure(figsize=(8, 8))
    plt.ion()

    try:
        while not coord.should_stop():
            start_time = time.time()

            # Run one step of the model.
            x_train_batch, y_train_batch = sess.run([x_train, y_train])
            _, step,loss, = sess.run([net.train_op, net.global_step, net.loss],
                                       feed_dict={net.x_ph:x_train_batch,net.y_ph:y_train_batch,net.is_training:True})

            duration = time.time() - start_time
            # test procedure
            if step % 50 == 0:
                # Print status to stdout.
                x_test_batch, y_test_batch = sess.run([x_test, y_test])
                loss_test, y_predict = sess.run( [net.loss,net.conv6],
                                             feed_dict={net.x_ph: x_test_batch, net.y_ph: y_test_batch,
                                                        net.is_training: False})
                print('Testing Step %d: loss = %.2f (%.3f sec)' % (step, loss_test, duration))

                plt.subplot(221)
                plt.axis('off')
                plt.imshow(x_test_batch[0].reshape((224,224)))
                plt.subplot(222)
                plt.axis('off')
                plt.imshow(y_predict[0].reshape((224,224))*255)
                plt.pause(0.001)
                plt.subplot(223)
                plt.axis('off')
                plt.imshow(x_test_batch[1].reshape((224,224)))
                plt.subplot(224)
                plt.axis('off')
                plt.imshow(y_predict[1].reshape((224,224))*255)
                plt.pause(0.001)



    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()



def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # if raw data have not been processed
    if not os.path.exists(project_config.DATA_FOLDER):
        os.mkdir(project_config.DATA_FOLDER)
        print("Converting fingerprint data")
        for i in range(8):
            scr_folder = os.path.join(project_config.RAWDATA_FOLDER, 'figs_' + str(i))
            # resize the images to the size we want
            fingerprint_data.downsample_sd4.convert(scr_folder, project_config.DATA_FOLDER)

    train()

if __name__ == '__main__':

    tf.app.run()