import tensorflow as tf
from tensorflow import layers as ly
import tensorflow.contrib.layers as cly
from util import *
from functools import partial
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

xavier_init = cly.xavier_initializer()

batch_size = 64
z_size = 10

learning_rate_gen = 5e-5
learning_rate_dis = 5e-5
shape_size = [9,9,9,3]

max_iter_step = 2000

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

Citers = 5

log_path = './log/chair/'
record_path = './data/tfrecord/'
model_path = './model/chair/'
model_file_name = 'model_wgan_' + str(shape_size[0])

n_epoch = 5000

clamp_lower = -0.01
clamp_upper = 0.01

resume = False

filter_num_d = {
	'1':3,
	'2':32, 
	'3':64,
}

def init_weights():

	global weights
	weights = {}
	xavier_init = tf.contrib.layers.xavier_initializer()

	# filter for deconv3d: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]
	weights['wg1'] = tf.get_variable("wg1", shape=[z_size,3*3*3*filter_num_d['3']], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, filter_num_d['2'], filter_num_d['3']], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, filter_num_d['1'], filter_num_d['2']], initializer=xavier_init)
	
def init_biases():
	
	global biases
	biases = {}
	zero_init = tf.zeros_initializer()

	biases['bg1'] = tf.get_variable("bg1", shape=[3*3*3*filter_num_d['3']], initializer=zero_init)
	biases['bg2'] = tf.get_variable("bg2", shape=[filter_num_d['2']], initializer=zero_init)
	biases['bg3'] = tf.get_variable("bg3", shape=[filter_num_d['1']], initializer=zero_init)
   
def generator(z, batch_size=batch_size,phase_train=True, ):
	strides = [1,2,2,2,1]

	output_shape = {
		'g1':[batch_size,3,3,3,filter_num_d['3']],
		'g2':[batch_size,5,5,5,filter_num_d['2']],
		'g3':[batch_size,9,9,9,filter_num_d['1']],
	}

	with tf.variable_scope("generator"):
		print "z shape: ", z.shape
		g1 = tf.add(tf.matmul(z, weights['wg1']), biases['bg1'])
		g1 = tf.reshape(g1, output_shape['g1'])
		g1 = tf.contrib.layers.batch_norm(g1, is_training=phase_train)

		print "g1 shape: ", g1.shape

		g2 = tf.nn.conv3d_transpose(g1, weights['wg2'], output_shape=output_shape['g2'], strides=strides, padding="SAME")
		g2 = tf.nn.bias_add(g2, biases['bg2'])
		g2 = tf.contrib.layers.batch_norm(g2, is_training=phase_train)
		g2 = tf.nn.relu(g2)

		print "g2 shape: ", g2.shape
		
		g3 = tf.nn.conv3d_transpose(g2, weights['wg3'], output_shape=output_shape['g3'], strides=strides, padding="SAME")
		g3 = tf.nn.bias_add(g3, biases['bg3'])                                   
		g3 = tf.nn.sigmoid(g3)

		print "g3 shape: ", g3.shape

	return g3

def discriminator(inputs, phase_train=True, reuse=False):
	stride_d = [2,2,2]
	kernel_d = [4,4,4]

	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

		print "inputs shape: ", inputs.shape

		d1 = ly.conv3d(inputs=inputs, filters=filter_num_d['2'], kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d1 shape: ", d1.shape

		d2 = ly.conv3d(inputs=d1, filters=filter_num_d['3'], kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d2 shape: ", d2.shape

		d3 = cly.fully_connected(tf.reshape(
			d2, [batch_size, -1]), 1, activation_fn=None)

		print "d3 shape: ", d3.shape

	return d3

def build_graph(real_cp):

	noise_dist = tf.contrib.distributions.Normal(0., 1.)
	z = noise_dist.sample((batch_size, z_size))
	gen = generator
	dis = discriminator

	with tf.variable_scope('generator'):
		fake_cp = generator(z)

	true_logit = dis(real_cp)
	fake_logit = dis(fake_cp, reuse=True)

	d_loss = tf.reduce_mean(fake_logit - true_logit)
	g_loss = tf.reduce_mean(-fake_logit)

	print "true logit shape: ", true_logit.shape
	print "fake logit shape: ", fake_logit.shape

	d_real_conf = tf.divide(tf.reduce_sum(tf.maximum(tf.minimum(true_logit, 0.99), 0.01)), batch_size)
	d_fake_conf = tf.divide(tf.reduce_sum(tf.maximum(tf.minimum(fake_logit, 0.99), 0.01)), batch_size)

	summary_real_conf = tf.summary.scalar("real_conf", d_real_conf)
	summary_fake_conf = tf.summary.scalar("fake_conf", d_fake_conf)

	d_output_x = true_logit
	d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
	summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

	d_output_z = -fake_logit
	d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
	summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

	g_loss_sum = tf.summary.scalar("g_loss", g_loss)
	d_loss_sum = tf.summary.scalar("d_loss", d_loss)

	g_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
	d_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
	opt_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate_gen).minimize(g_loss, var_list=g_params)
    opt_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate_dis).minimize(d_loss, var_list=d_params)

	# counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	# opt_g = cly.optimize_loss(loss=g_loss, learning_rate=learning_rate_gen,
	# 				optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer, 
	# 				variables=theta_g, global_step=counter_g,
	# 				summaries = ['gradient_norm'])
	# counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	# opt_d = cly.optimize_loss(loss=d_loss, learning_rate=learning_rate_dis,
	# 				optimizer=
	# 				variables=theta_d, global_step=counter_d,
	# 				summaries = ['gradient_norm'])

	
	clipped_var_d = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in d_params]
	# merge the clip operations on critic variables
	with tf.control_dependencies([opt_d]):
		opt_d = tf.tuple(clipped_var_d)


	return opt_g, opt_d, g_loss, d_loss, d_real_conf, d_fake_conf

def main():
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8

	clear_log(log_path)
	print "Clear Log"

	with tf.device(device_gpu):
		weights = init_weights()
		biases = init_biases()
		cp_batch = load_data(record_path, n_epoch, batch_size, tuple(shape_size))
		opt_g, opt_d, g_loss, d_loss, real_conf, fake_conf = build_graph(cp_batch)
		merged_all = tf.summary.merge_all()		
		print "Finish Building Graph"

	with tf.Session(config=config) as sess:
		init_op = tf.local_variables_initializer()
		sess.run(init_op)
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		summary_writer = tf.summary.FileWriter(log_path, sess.graph)

		print "Finish Init and Start Training Step"

		if resume:
			print("Load existing model " + "!"*10)
			saver = tf.train.Saver()
			saver.restore(sess, model_file_name)

		saver = tf.train.Saver(max_to_keep=None)

		for i in range(max_iter_step):
			if i < 25 or i % 500 == 0:
				citers = 15
			else:
				citers = Citers

			if i % 50 == 49 and i != 0:
				print "step: %r of total step %r, fake_conf %r g_loss %r, real_conf %r, d_loss %r" % (i+1, max_iter_step,
					sess.run(fake_conf), sess.run(g_loss), sess.run(real_conf), sess.run(d_loss))

			for j in range(citers):
				if j % 50 == 49 and j != 0 :
					print "citers %r of %r during step %r d_citer_loss: %r" % (j+1, citers, i+1, sess.run(d_loss))
				else:
					sess.run([opt_d])
			else:
				sess.run([opt_g])

			merged = sess.run(merged_all)
			summary_writer.add_summary(merged, i)

			if i % 1000 == 999 or i == 0:
				if not os.path.exists(model_path):
					os.mkdir(model_path)
				saver.save(sess, model_path+model_file_name+".ckpt")
				print("saving model " + "!"*10)

if __name__ == '__main__':
	main()
