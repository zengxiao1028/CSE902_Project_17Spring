import numpy as np
import tensorflow as tf
import os
from network import SD14FingerNet
import project_config
from data_provider.gen_sd14_data import SD14DataGenerator
from sklearn.externals import joblib
from data_provider.gen_sd4_data import SD4DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def convert_label(x):
    x = x[0]
    return {0:'Arch',1:'Left Loop',2:'Right Loop', 3:'Tented Arch',4:'Whorl'}.get(x,'error')

def save_case(y_test_batch,y_pred,x_test_batch,i,folder):
    fig = plt.figure(figsize=(4, 4))
    title = 'truth:'+convert_label(y_test_batch)+',pred:'+convert_label(y_pred)
    plt.title(s=title)
    plt.imshow(x_test_batch.reshape(x_test_batch.shape[1], x_test_batch.shape[1]), cmap='Greys_r')
    save_file_name = os.path.join(folder,'fail_'+str(i)+'.jpg')
    plt.savefig(save_file_name, bbox_inches='tight')
    plt.close(fig)




def evaluate_sd4():
    input_shape = (None, project_config.SD14_INPUT_IMG_SIZE, project_config.SD14_INPUT_IMG_SIZE, 1)
    transformed_shape = (None, project_config.SD14_TRANSFORMED_IMG_SIZE, project_config.SD14_TRANSFORMED_IMG_SIZE, 1)

    num_classes = 5

    net = SD14FingerNet.FingerNet(input_shape, transformed_shape, num_classes, 'classification')
    dg = SD4DataGenerator(project_config.SD4_DATA_ORIGIN_FOLDER)

    x_test, y_test = dg.get_batch_for_test()

    saver_for_restore = tf.train.Saver()
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

        ckpt = tf.train.get_checkpoint_state('sd4_out/restore_models/')

        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver_for_restore.restore(sess, ckpt.model_checkpoint_path)
        print('Restored')
        x_features = []
        y_labels = []
        mean_acc = []
        fail_cnt = 0
        y_preds = []
        while not coord.should_stop():

            ## Run one step of the model.
            # x_train_batch, y_train_batch = sess.run([x_train, y_train])
            #
            # acc, feas = sess.run([net.acc, net.fea],
            #                            feed_dict={net.x_ph:x_train_batch, net.y_ph:y_train_batch, net.is_training:False})
            #
            # x_features.append(feas)
            # y_labels.append(y_train_batch)



            # # Run one step of the model.
            x_test_batch, y_test_batch = sess.run([x_test, y_test])

            acc, feas, y_pred = sess.run([net.acc, net.fea, net.predction],
                                         feed_dict={net.x_ph: x_test_batch, net.y_ph: y_test_batch,
                                                    net.is_training: False})

            x_features.append(feas)
            y_labels.append(y_test_batch)
            y_preds.append(y_pred)
            if y_test_batch != y_pred:
                fail_cnt += 1
                save_case(y_test_batch, y_pred, x_test_batch, fail_cnt,'sd4_out/fails')

            mean_acc.append(acc)



    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    x_features = np.vstack(x_features).reshape((-1, 1024))
    y_labels = np.vstack(y_labels).reshape((-1, ))
    y_preds = np.vstack(y_preds).reshape((-1,))
    print(x_features.shape)
    print(np.mean(mean_acc))

    # joblib.dump((x_features,y_labels),'sd14_train_features.pkl')

    joblib.dump((x_features, y_labels), 'tmp.pkl')

    print(classification_report(y_labels, y_preds,
                                target_names=['arch', 'left-loop', 'right-loop', 'tented-arch', 'whorl']))


    print(accuracy_score(y_labels, y_preds))

    print(confusion_matrix(y_labels, y_preds))



def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    evaluate_sd4()

    x, y =joblib.load( os.path.join(project_config.ROOT_DIR ,'sd14_out/extracted_features/sd14_test_features.pkl'))


    print(x.shape,y.shape)

if __name__ == '__main__':
    tf.app.run()