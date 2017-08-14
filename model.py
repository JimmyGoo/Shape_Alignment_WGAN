from tensorflow import layers as ly
import tensorflow as tf
import tensorflow.contrib.layers as cly

delving_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()
fc_init = tf.truncated_normal_initializer(stddev=0.005)

init_mode = 1
if init_mode == 0:
	current_init = xavier_init
elif init_mode == 1:
	current_init = delving_init
else:
	print "init mode error"

DROPOUT = True

def LeakyReLU(x, alpha=0.2):
	return tf.maximum(alpha*x, x)

def init_weights(filter_num_d, output_shape, Z_SIZE):

	global weights
	weights = {}

	g1_s = output_shape['g1']

	# filter for deconv3d: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]
	weights['wg1'] = tf.get_variable("wg1", shape=[Z_SIZE, g1_s[1]*g1_s[2]*g1_s[3]*filter_num_d['4']], initializer=current_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, filter_num_d['3'], filter_num_d['4']], initializer=current_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, filter_num_d['2'], filter_num_d['3']], initializer=current_init)
	weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, filter_num_d['1'], filter_num_d['2']], initializer=current_init)

	weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, filter_num_d['1'], filter_num_d['2']], initializer=current_init)
	weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, filter_num_d['2'], filter_num_d['3']], initializer=current_init)
	weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, filter_num_d['3'], filter_num_d['4']], initializer=current_init)
	
def init_biases(filter_num_d, output_shape):
	
	global biases
	biases = {}

	g1_s = output_shape['g1']

	biases['bg1'] = tf.get_variable("bg1", shape=[g1_s[1]*g1_s[2]*g1_s[3]*filter_num_d['4']], initializer=zero_init)
	biases['bg2'] = tf.get_variable("bg2", shape=[filter_num_d['3']], initializer=zero_init)
	biases['bg3'] = tf.get_variable("bg3", shape=[filter_num_d['2']], initializer=zero_init)
	biases['bg4'] = tf.get_variable("bg4", shape=[filter_num_d['1']], initializer=zero_init)

	biases['bd1'] = tf.get_variable("bd1", shape=[filter_num_d['2']], initializer=zero_init)
	biases['bd2'] = tf.get_variable("bd2", shape=[filter_num_d['3']], initializer=zero_init)
	biases['bd3'] = tf.get_variable("bd3", shape=[filter_num_d['4']], initializer=zero_init)

def generator(n_samples, output_shape, Z_SIZE, phase_train=True, noise=None, reuse=False):
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
		g1 = cly.fully_connected(inputs=noise, num_outputs=number_outputs, activation_fn=tf.nn.relu, weights_initializer=fc_init)
		g1 = ly.batch_normalization(g1, training=phase_train)
		g1 = tf.reshape(g1, output_shape['g1'])

		print "g1 shape: ", g1.shape

		g2 = tf.nn.conv3d_transpose(g1, weights['wg2'], output_shape=output_shape['g2'], strides=strides, padding="SAME")
		g2 = tf.nn.bias_add(g2, biases['bg2'])
		g2 = ly.batch_normalization(g2, training=phase_train)
		g2 = LeakyReLU(g2)

		print "g2 shape: ", g2.shape

		g3 = tf.nn.conv3d_transpose(g2, weights['wg3'], output_shape=output_shape['g3'], strides=strides, padding="SAME")
		g3 = tf.nn.bias_add(g3, biases['bg3'])
		g3 = ly.batch_normalization(g3, training=phase_train)
		g3 = LeakyReLU(g3)

		print "g3 shape: ", g3.shape

		g4 = tf.nn.conv3d_transpose(g3, weights['wg4'], output_shape=output_shape['g4'], strides=strides, padding="SAME")
		g4 = tf.nn.bias_add(g4, biases['bg4'])                                   
		g4 = tf.tanh(g4)

		print "g4 shape: ", g4.shape

		output = tf.reshape(g4, [n_samples, OUTPUT_DIM])

		print "output shape: ", output.shape

	return output

def discriminator(inputs, filter_num_d, output_shape, batch_size, phase_train=True, reuse=False, GP=False):

	stride_d = [1,2,2,2,1]
	kernel_d = [4,4,4]
	
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

		inputs = tf.reshape(inputs, output_shape['g4'])
		print "inputs shape: ", inputs.shape

		d1 = tf.nn.conv3d(inputs, weights['wd1'], strides=stride_d, padding="SAME")
		d1 = tf.nn.bias_add(d1, biases['bd1'])
		if GP == False:
			d1 = ly.batch_normalization(d1, training=phase_train)                              
		d1 = LeakyReLU(d1)

		print "d1 shape: ", d1.shape

		d2 = tf.nn.conv3d(d1, weights['wd2'], strides=stride_d, padding="SAME") 
		d2 = tf.nn.bias_add(d2, biases['bd2'])
		if GP == False:                
			d2 = ly.batch_normalization(d2, training=phase_train)
		d2 = LeakyReLU(d2)

		print "d2 shape: ", d2.shape

		d3 = tf.nn.conv3d(d2, weights['wd3'], strides=stride_d, padding="SAME") 
		d3 = tf.nn.bias_add(d3, biases['bd3'])
		if GP == False:                            
			d3 = ly.batch_normalization(d3, training=phase_train)
		d3 = LeakyReLU(d3)

		print "d3 shape: ", d3.shape

		d4 = cly.fully_connected(tf.reshape(
			d3, [batch_size, -1]), 1, activation_fn=None, weights_initializer=fc_init)
		d4 = tf.reshape(d4, [-1])

		print "d4 shape: ", d4.shape

	return d4