import tensorflow as tf
import tensorflow.contrib.slim as slim

class depress_net:

    def __init__(self,input_shape):

        self.input_shape = input_shape
        self.build_network()

    def build_network(self):

        print('Building network...')
        x_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape[1:])
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        global_step = tf.Variable(0,trainable=False)
        is_training = tf.placeholder(dtype=tf.bool)

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.001)):
            conv_1 = slim.conv2d(x_ph, 128, [1, 5], scope='conv1')
            pool_1 = slim.max_pool2d(conv_1, kernel_size=(1, 3), stride=1)
            pool_1 = slim.dropout(pool_1, is_training=is_training)

            conv_2 = slim.conv2d(pool_1, 128, [2, 3], scope='conv2')
            pool_2 = slim.max_pool2d(conv_2, kernel_size=(1, 3), stride=1)
            pool_2 = slim.dropout(pool_2, is_training=is_training)

            fc_1 = slim.flatten(pool_2)
            fc_1 = slim.fully_connected(fc_1, 128)
            fc_1 = slim.dropout(fc_1, is_training=is_training)

            fc_2 = slim.fully_connected(fc_1, 1, activation_fn=None)


            predict = tf.cast(tf.nn.sigmoid(fc_2)+0.5, dtype=tf.int32, name='prediction')
            acc = slim.metrics.accuracy(predictions=predict, labels=tf.cast(y_ph, dtype=tf.int32, name='prediction'))

            # loss
            slim.losses.sigmoid_cross_entropy(fc_2, y_ph)
            total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

            optimizor = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(total_loss,global_step=global_step)

            self.x_ph = x_ph
            self.y_ph = y_ph
            self.is_training = is_training
            self.conv_1 = conv_1
            self.pool_1 = pool_1
            self.conv_2 = conv_2
            self.pool_2 = pool_2
            self.fc_1 = fc_1
            self.fc_2 = fc_2
            self.predict = predict
            self.acc = acc
            self.loss = total_loss
            self.optimizor = optimizor
            self.global_step = global_step


