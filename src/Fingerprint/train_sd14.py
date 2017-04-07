import os
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import fingerprint_data.sd4.convert_sd4
import project_config
from data_provider.gen_sd14_data import SD14DataGenerator
from network import SD14FingerNet

from util.plot_util import collect_failure_cases,plot



def train():

    input_shape = (None, project_config.SD14_INPUT_IMG_SIZE, project_config.SD14_INPUT_IMG_SIZE, 1)
    transformed_shape = (None, project_config.SD14_TRANSFORMED_IMG_SIZE, project_config.SD14_TRANSFORMED_IMG_SIZE, 1)

    num_classes = 5

    net = SD14FingerNet.FingerNet(input_shape, transformed_shape, num_classes, 'classification')
    dg = SD14DataGenerator(project_config.SD14_DATA_FOLDER)
    x_train, y_train, x_test, y_test = dg.get_batch(32)

    saver = tf.train.Saver(max_to_keep=1)

    # trainable_vars = tf.trainable_variables()
    # restore_var_list = [each for each in trainable_vars if each.name.find('transformer')<0 ]
    saver_for_restore = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()

    #b_tensor = sess.graph.get_tensor_by_name('transformer/fully_connected/biases/read:0')
    #fc7_biases_tanh = tf.nn.tanh(b_tensor)

    # Run the Op to initialize the variables.
    sess.run(tf.global_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

        ckpt = tf.train.get_checkpoint_state('sd14_out/models/')
        if ckpt and ckpt.model_checkpoint_path:
            #Restores from checkpoint
           saver_for_restore.restore(sess, ckpt.model_checkpoint_path)

        while not coord.should_stop():
            start_time = time.time()

            # Run one step of the model.
            x_train_batch, y_train_batch = sess.run([x_train, y_train])
            _, step,loss, acc, x_entropy = sess.run([net.train_op, net.global_step, net.loss, net.acc, net.x_entropy],
                                       feed_dict={net.x_ph:x_train_batch,net.y_ph:y_train_batch,net.is_training:True})

            duration = time.time() - start_time
            # test procedure
            if step % 200 == 0:
                # Print status to stdout.
                x_test_batch, y_test_batch = sess.run([x_test, y_test])
                loss_test, acc_test, x_entropy_test, y_red = sess.run([ net.loss, net.acc, net.x_entropy, net.predction ],
                                             feed_dict={net.x_ph: x_test_batch, net.y_ph: y_test_batch,
                                                        net.is_training: False})

                print('Testing Step %d: loss(xentropy) = %.2f (%.2f)  acc = %.2f (%.3f sec)' % (step, loss_test, x_entropy_test,acc_test, duration))
                print(confusion_matrix(y_test_batch, y_red))
                collect_failure_cases(y_test_batch,y_red,x_test_batch)


            #save transformed images
            # if step % 100 == 0:
            #         #samples,theta,theta_b = sess.run([net.x_transformed, net.theta, fc7_biases_tanh], feed_dict={net.x_ph:x_test_batch, net.is_training:False})
            #
            #         #print(theta[0])
            #         #print(theta_b)
            #         #fig = plot(samples[:16])
            #         #plt.savefig('sd14_out/pics/{}_transformed.png'
            #         #            .format(str(step).zfill(3)), bbox_inches='tight')
            #         #plt.close(fig)
            #
            #         fig = plot(x_test_batch[:16])
            #         plt.savefig('sd14_out/pics/{}_raw.png'
            #                     .format(str(step).zfill(3)), bbox_inches='tight')
            #         plt.close(fig)



            if step%500 == 0:
                save_path = saver.save(sess, 'sd14_out/models/model.ckpt', global_step=step)

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
    if not os.path.exists(project_config.SD4_DATA_FOLDER):

        if os.path.exists(project_config.RAWDATA_FOLDER):
            os.mkdir(project_config.SD4_DATA_FOLDER)
            print("Converting fingerprint data")
            for i in range(8):
                scr_folder = os.path.join(project_config.RAWDATA_FOLDER, 'figs_' + str(i))

                # resize the images to the size we want
                fingerprint_data.downsample_sd4.convert(scr_folder, project_config.SD4_DATA_FOLDER)

    train()

if __name__ == '__main__':
    tf.app.run()