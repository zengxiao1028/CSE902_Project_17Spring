import tensorflow as tf
from network import network
import numpy as np
from network.gen_data import get_batch
import project_config
import time
import os
import fingerprint_data.downsample_sd4
def train():

    shape = (None,project_config.IMG_SIZE,project_config.IMG_SIZE,1)
    num_classes = 5

    net = network.FingerNet(shape,num_classes)

    x_train, y_train, x_test, y_test = get_batch(project_config.DES_FOLDER,64)

    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(tf.global_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            start_time = time.time()

            # Run one step of the model.
            x_train_batch, y_train_batch = sess.run([x_train, y_train])
            _, step,loss, acc, x_entropy = sess.run([net.train_op, net.global_step, net.loss, net.acc,net.x_entropy],
                                       feed_dict={net.x_ph:x_train_batch,net.y_ph:y_train_batch,net.is_training:True})

            duration = time.time() - start_time

            # test procedure
            if step % 50 == 0:
                # Print status to stdout.
                x_test_batch, y_test_batch = sess.run([x_test, y_test])
                loss_test, acc_test, x_entropy_test = sess.run([ net.loss, net.acc, net.x_entropy ],
                                             feed_dict={net.x_ph: x_test_batch, net.y_ph: y_test_batch,
                                                        net.is_training: False})
                print('Step %d: loss(xentropy) = %.2f (%.2f)  acc = %.2f (%.3f sec)' % (step, loss_test, x_entropy_test,acc_test, duration))

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

    #if raw data have not been processed
    if not os.path.exists(project_config.DES_FOLDER):
        os.mkdir(project_config.DES_FOLDER)
        for i in range(8):
            scr_folder = os.path.join(project_config.RAWDATA_FOLDER, 'figs_' + str(i))
            # resize the images to the size we want
            fingerprint_data.downsample_sd4.convert(scr_folder, project_config.DES_FOLDER)

    train()

if __name__ == '__main__':
    tf.app.run()