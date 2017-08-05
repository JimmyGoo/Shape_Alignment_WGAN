from tensorflow import layers as ly
import tensorflow as tf
import tensorflow.contrib.layers as cly

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def init_weights(filter_num_d, output_shape, Z_SIZE):

    global weights
    weights = {}
    xavier_init = tf.contrib.layers.xavier_initializer()

    g1_s = output_shape['g1']

    # filter for deconv3d: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]
    weights['wg1'] = tf.get_variable("wg1", shape=[Z_SIZE, g1_s[1]*g1_s[2]*g1_s[3]*filter_num_d['4']], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, filter_num_d['3'], filter_num_d['4']], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, filter_num_d['2'], filter_num_d['3']], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, filter_num_d['1'], filter_num_d['2']], initializer=xavier_init)
    
def init_biases(filter_num_d, output_shape):
    
    global biases
    biases = {}
    zero_init = tf.zeros_initializer()

    g1_s = output_shape['g1']

    biases['bg1'] = tf.get_variable("bg1", shape=[g1_s[1]*g1_s[2]*g1_s[3]*filter_num_d['4']], initializer=zero_init)
    biases['bg2'] = tf.get_variable("bg2", shape=[filter_num_d['3']], initializer=zero_init)
    biases['bg3'] = tf.get_variable("bg3", shape=[filter_num_d['2']], initializer=zero_init)
    biases['bg4'] = tf.get_variable("bg4", shape=[filter_num_d['1']], initializer=zero_init)


def Generator(n_samples, output_shape, Z_SIZE, phase_train=True, noise=None, reuse=False):
    if noise is None:
        noise = tf.random_normal([n_samples, Z_SIZE])

    strides = [1,2,2,2,1]
    out_s = output_shape['g4']
    OUTPUT_DIM = out_s[1] * out_s[2] * out_s[3] * out_s[4]

    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        print "noise shape: ", noise.shape
        shape = output_shape['g1']
        number_outputs = shape[1] * shape[2] * shape[3] * shape[4]
        g1 = cly.fully_connected(inputs=noise, num_outputs=number_outputs, activation_fn=tf.nn.relu, 
            normalizer_fn=cly.batch_norm)
        g1 = tf.reshape(g1, output_shape['g1'])

        print "g1 shape: ", g1.shape

        g2 = tf.nn.conv3d_transpose(g1, weights['wg2'], output_shape=output_shape['g2'], strides=strides, padding="SAME")
        g2 = tf.nn.bias_add(g2, biases['bg2'])
        g2 = tf.contrib.layers.batch_norm(g2, is_training=phase_train)
        g2 = tf.nn.relu(g2)

        print "g2 shape: ", g2.shape

        g3 = tf.nn.conv3d_transpose(g2, weights['wg3'], output_shape=output_shape['g3'], strides=strides, padding="SAME")
        g3 = tf.nn.bias_add(g3, biases['bg3'])
        g3 = tf.contrib.layers.batch_norm(g3, is_training=phase_train)
        g3 = tf.nn.relu(g3)

        print "g3 shape: ", g3.shape

        g4 = tf.nn.conv3d_transpose(g3, weights['wg4'], output_shape=output_shape['g4'], strides=strides, padding="SAME")
        g4 = tf.nn.bias_add(g4, biases['bg4'])                                   
        g4 = tf.tanh(g4)

        print "g4 shape: ", g4.shape

        output = tf.reshape(g4, [-1, OUTPUT_DIM])
    return output

def Discriminator(inputs, output_shape, phase_train=True, reuse=False):

    stride_d = [2,2,2]
    kernel_d = [4,4,4]
   
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        inputs = tf.reshape(inputs, output_shape['g4'])
        print "inputs shape: ", inputs.shape

        d1 = ly.conv3d(inputs=inputs, filters=filter_num_d['2'], kernel_size=kernel_d,
            strides=stride_d, padding="SAME",activation=LeakyReLU, activity_regularizer=cly.batch_norm,
            kernel_initializer=xavier_init)

        print "d1 shape: ", d1.shape

        d2 = ly.conv3d(inputs=d1, filters=filter_num_d['3'], kernel_size=kernel_d,
            strides=stride_d, padding="SAME",activation=LeakyReLU, activity_regularizer=cly.batch_norm,
            kernel_initializer=xavier_init)

        print "d2 shape: ", d2.shape

        ##one layer deeper

        d3 = ly.conv3d(inputs=d2, filters=filter_num_d['4'], kernel_size=kernel_d,
            strides=stride_d, padding="SAME",activation=LeakyReLU, activity_regularizer=cly.batch_norm,
            kernel_initializer=xavier_init)

        print "d3 shape: ", d3.shape

        d4 = cly.fully_connected(tf.reshape(
            d3, [BATCH_SIZE, -1]), 1, activation_fn=None)
        d4 = tf.reshape(d4, [-1])

        print "d4 shape: ", d4.shape

    return d4