import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

import project_config
from network.spatial_transformer import transformer
from util.tf_utils import *


class FingerNet:

    def __init__(self,input_shape,transformed_shape,num_classes,network):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transformed_shape = transformed_shape
        if network == 'classification':
            #self.build_inception_v3()
            self.build_classification_network()
        elif network == 'localization':
            self.build_sp_localization_network()

    def build_inception_v3(self):
        print('Building network...')

        # inputs
        x_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape[1:])
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        # global step indicator
        global_step = tf.Variable(0, trainable=False)
        # train or test
        is_training = tf.placeholder(dtype=tf.bool)


        x_out, theta = self._build_transformed_network(x_ph, (self.transformed_shape[1],
                                                              self.transformed_shape[1]),
                                                       is_training)

        self.x_transformed = x_out
        self.theta = theta


        logits, end_points = nets.inception.inception_v3(x_out, self.num_classes, is_training=is_training)

        # prediction
        prediction = tf.argmax(logits, axis=1, name='prediction')
        acc = slim.metrics.accuracy(predictions=tf.cast(prediction, dtype=tf.int32),
                                    labels=tf.cast(y_ph, dtype=tf.int32))

        # classification loss
        x_entropy = tf.losses.sparse_softmax_cross_entropy(tf.cast(y_ph, dtype=tf.int32), logits)
        # add regularization loss
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        #
        self.optimizor = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = self.optimizor.minimize(total_loss, global_step=global_step)

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

    def _build_transformed_network(self, x_in, out_size, is_training):

        with tf.variable_scope('transformer'):

            with slim.arg_scope([slim.convolution2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0001) ):
                with slim.arg_scope([slim.convolution2d] ):

                    #regression_in = tf.image.resize_images(x_in,(224,224))
                    conv_1 = slim.convolution2d(x_in, 96, kernel_size=11, stride=4, padding='VALID', scope='conv1')
                    conv_1_pool = slim.max_pool2d(conv_1, kernel_size=3,  scope='pool1')

                    conv_2_1 = slim.convolution2d(conv_1_pool, 128, kernel_size=3, stride=2, scope='conv2_1')
                    conv_2_2 = slim.convolution2d(conv_2_1, 128, kernel_size=3,stride=2,  scope='conv2_2')


                    conv_3_2 = slim.convolution2d(conv_2_2, 128, kernel_size=3, scope='conv3_2')

                    conv_4_1 = slim.convolution2d(conv_3_2, 128, kernel_size=3, scope='conv4_1')


                    conv_5 = slim.convolution2d(conv_4_1, 128, kernel_size=3, scope='conv5')

                    conv_5_pool = slim.max_pool2d(conv_5, kernel_size=3, scope='pool5')
                    conv_5_pool_flat = slim.flatten(conv_5_pool, scope='flat5')


                    conv_5_pool_flat = slim.flatten(conv_5_pool_flat, scope='flat5')

                    fc6 = slim.fully_connected(conv_5_pool_flat, 512, scope='fc6')
                    fc6_dropout = slim.dropout(fc6, is_training=is_training, scope='fc6_dropout')

                    fc_7 = slim.fully_connected(fc6_dropout,6, activation_fn=tf.nn.tanh,
                                                weights_initializer=tf.constant_initializer(np.zeros([512, 6])),
                                                biases_initializer=tf.constant_initializer(np.array([[1, 0, 0], [0, 1, 0]])))


            out = transformer(x_in, fc_7, out_size=out_size)

        return tf.reshape(out,(-1,)+ out_size + (1,)), fc_7


    def build_classification_network(self):
        print('Building network...')

        #inputs
        x_ph = tf.placeholder(dtype=tf.float32, shape= (None,) + self.input_shape[1:])
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        #global step indicator
        global_step = tf.Variable(0,trainable=False)
        #train or test
        is_training = tf.placeholder(dtype=tf.bool)

        # x_out,theta = self._build_transformed_network(x_ph, (project_config.SD14_TRANSFORMED_IMG_SIZE,
        #                                                      project_config.SD14_TRANSFORMED_IMG_SIZE), is_training)
        #
        # self.x_transformed = x_out
        # self.theta = theta

        with slim.arg_scope([slim.convolution2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0001),
                            variables_collections=['CLS_VARS']):
            conv_1 = slim.convolution2d(x_ph, 96, kernel_size=11, stride=4,  scope='conv1')
            conv_1_pool = slim.max_pool2d(conv_1, kernel_size=3, stride=2, scope='pool1')

            conv_2_1 = slim.convolution2d(conv_1_pool, 256, kernel_size=3, stride=2,  scope='conv2_1')
            conv_2_2 = slim.convolution2d(conv_2_1, 256, kernel_size=3,  scope='conv2_2')

            conv_2_pool = slim.max_pool2d(conv_2_2, kernel_size=3, scope='pool2')

            conv_3_1 = slim.convolution2d(conv_2_pool, 384, kernel_size=3, scope='conv3_1')
            conv_3_2 = slim.convolution2d(conv_3_1, 384, kernel_size=3, scope='conv3_2')

            conv_4_1 = slim.convolution2d(conv_3_2, 256, kernel_size=3, scope='conv4_1')
            conv_4_2 = slim.convolution2d(conv_4_1, 256, kernel_size=3, scope='conv4_2')

            conv_5 = slim.convolution2d(conv_4_2, 256, kernel_size=3, scope='conv5')

            conv_5_pool = slim.max_pool2d(conv_5, kernel_size=3, scope='pool5')
            conv_5_pool_flat = slim.flatten(conv_5_pool, scope='flat5')

            fc6_fea = slim.fully_connected(conv_5_pool_flat, 1024, scope='fc6',activation_fn=None)
            fc6 = tf.nn.relu(fc6_fea)
            fc6_dropout = slim.dropout(fc6, is_training=is_training, scope='fc6_dropout')

            fc7 = slim.fully_connected(fc6_dropout, self.num_classes, activation_fn=None, scope='fc7')
            prediction = tf.argmax(fc7, axis=1, name = 'prediction')

        #prediction
        acc = slim.metrics.accuracy(predictions=tf.cast(prediction, dtype=tf.int32), labels=tf.cast(y_ph, dtype=tf.int32))

        # classification loss
        x_entropy = tf.losses.sparse_softmax_cross_entropy( tf.cast(y_ph, dtype=tf.int32),fc7)
        #add regularization loss
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        #
        self.optimizor = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = self.optimizor.minimize(total_loss,global_step=global_step)

        self.fea = fc6_fea
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

    def build_sp_localization_network(self):
        print('Building network...')

        # inputs
        x_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape[1:])
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape[1:])

        # global step indicator
        global_step = tf.Variable(0, trainable=False)
        # train or test
        is_training = tf.placeholder(dtype=tf.bool)

        with slim.arg_scope([slim.convolution2d],
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            conv_1 = slim.convolution2d(x_ph, 96, kernel_size=9, stride=4, scope='conv1')

            conv_2_1 = slim.convolution2d(conv_1, 64, kernel_size=3, stride=2, scope='conv2_1')
            conv_2_2 = slim.convolution2d(conv_2_1, 64, kernel_size=3, stride=2, scope='conv2_2')
            conv_3_2 = slim.convolution2d(conv_2_2, 64, kernel_size=3, scope='conv3_2')

            conv_3_2_upsample = tf.image.resize_images(conv_3_2, np.array(conv_3_2.get_shape().as_list()[1:3]) * 2)

            conv_4_1 = slim.convolution2d(conv_3_2_upsample, 64, kernel_size=3, scope='conv4_1')
            conv_4_1_upsample = tf.image.resize_images(conv_4_1, np.array(conv_4_1.get_shape().as_list()[1:3]) * 2)

            conv_4_2 = slim.convolution2d(conv_4_1_upsample, 64, kernel_size=3, scope='conv4_2')
            conv_4_2_upsample = tf.image.resize_images(conv_4_2, np.array(conv_4_2.get_shape().as_list()[1:3]) * 2)

            conv_4_3 = slim.convolution2d(conv_4_2_upsample, 64, kernel_size=3, scope='conv4_3')
            conv_4_3_upsample = tf.image.resize_images(conv_4_3, np.array(conv_4_3.get_shape().as_list()[1:3]) * 2)

            conv_5 = slim.convolution2d(conv_4_3_upsample, 64, kernel_size=3, scope='conv_5')

            conv_6 = slim.convolution2d(conv_5, 1, kernel_size=3, scope='conv6', activation_fn=None)

            y_ph = y_ph / 255.

        sq_dif = tf.squared_difference(conv_6, y_ph)
        mean_square_erro = tf.reduce_mean(sq_dif)
        tf.losses.add_loss(mean_square_erro)

        # add regularization loss
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        #
        self.optimizor = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = self.optimizor.minimize(total_loss, global_step=global_step)

        self.x_ph = x_ph
        self.y_ph = y_ph
        self.is_training = is_training
        self.loss = total_loss
        self.x_entropy = sq_dif
        self.optimizor = train_op
        self.global_step = global_step
        self.train_op = train_op
        self.conv6 = conv_6