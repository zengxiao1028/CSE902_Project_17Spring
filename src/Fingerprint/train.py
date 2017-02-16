import tensorflow as tf
from network import network
import numpy as np
from network.gen_data import get_batch
import project_config
import time
def train():

    X_train = np.zeros((128,224,224,1))

    num_classes = 5

    net = network.FingerNet(X_train.shape,num_classes)

    x_batch, y_batch = get_batch(project_config.DES_FOLDER,64)

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
            x, y = sess.run([x_batch, y_batch])
            _,loss,acc,step = sess.run([net.train_op, net.loss, net.acc, net.global_step],
                                       feed_dict={net.x_ph:x,net.y_ph:y,net.is_training:True})

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 1 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss,
                                                           duration))

    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()



def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()