import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_data_batch(data_folder, batch_size):
    myDataset.py

class FingerNet:

    def __init__(self,input_shape,num_classes):

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.build_network()


    def build_network(self):

        print('Building network...')

        #inputs
        x_ph = tf.placeholder(dtype=tf.float32, shape= (None,) + self.input_shape[1:])
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        #global step indicator
        global_step = tf.Variable(0,trainable=False)
        #train or test
        is_training = tf.placeholder(dtype=tf.bool)

        with slim.arg_scope([slim.convolution2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0001)):

            conv_1 = slim.convolution2d(x_ph, 96, kernel_size=11, stride=4, padding='VALID', scope='conv1')
            conv_1_pool = slim.max_pool2d(conv_1, kernel_size=3, stride=2, scope='pool1')

            conv_2_1 = slim.convolution2d(conv_1_pool, 256, kernel_size=3,  scope='conv2_1')
            conv_2_2 = slim.convolution2d(conv_2_1, 256, kernel_size=3,  scope='conv2_2')

            conv_2_pool = slim.max_pool2d(conv_2_2, kernel_size=3, scope='pool2')

            conv_3_1 = slim.convolution2d(conv_2_pool, 384, kernel_size=3, scope='conv3_1')
            conv_3_2 = slim.convolution2d(conv_3_1, 384, kernel_size=3, scope='conv3_2')

            conv_4_1 = slim.convolution2d(conv_3_2, 256, kernel_size=3, scope='conv4_1')
            conv_4_2 = slim.convolution2d(conv_4_1, 256, kernel_size=3, scope='conv4_2')

            conv_5 = slim.convolution2d(conv_4_2, 256, kernel_size=3, scope='conv5')

            conv_5_pool = slim.max_pool2d(conv_5, kernel_size=3, scope='pool5')
            conv_5_pool_flat = slim.flatten(conv_5_pool, scope='flat5')

            fc6 = slim.fully_connected(conv_5_pool_flat, 1024, scope='fc6')
            fc6_dropout = slim.dropout(fc6, is_training=is_training, scope='fc6_dropout')

            fc7 = slim.fully_connected(fc6_dropout, self.num_classes, activation_fn=None, scope='fc7')
            prediction = tf.argmax(fc7,axis=1,name = 'prediction')

        #prediction
        acc = slim.metrics.accuracy(predictions=tf.cast(prediction, dtype=tf.int32), labels=tf.cast(y_ph, dtype=tf.int32))

        # classification loss
        x_entropy = tf.losses.sparse_softmax_cross_entropy( tf.cast(y_ph, dtype=tf.int32),fc7)
        #add regularization loss
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        #
        self.optimizor = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = self.optimizor.minimize(total_loss,global_step=global_step)

        self.x_ph = x_ph
        self.y_ph = y_ph
        self.is_training = is_training
        self.acc = acc
        self.loss = total_loss
        self.x_entropy = x_entropy
        self.optimizor = train_op
        self.global_step = global_step
        self.train_op = train_op
        self.predction = prediction