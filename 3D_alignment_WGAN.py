import tensorflow as tf
from tensorflow import layers as ly
import tensorflow.contrib.layers as cly
from util import *
from functools import partial
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]
###Problem

##8 * 8 * 8 in voxel space?
##giving output S = 64 64 64, how to back in 8 * 8 * 8(S value is not int, S maybe larger then 8 * 8 * 8, value in S dont have
##corresponding index)

xavier_init = cly.xavier_initializer()
#input 10(pending) dim Z latent space
#Pending
batch_size = 50
z_size = 10

learning_rate_gen = 5e-5
learning_rate_dis = 5e-5
shape_size = [9,9,9,3]

max_iter_step = 2000

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

is_adam = True

Citers = 5

log_path = './log/chair/'
record_path = './data/tfrecord/'

n_epoch = 500

clamp_lower = -0.01
clamp_upper = 0.01

def init_weights():

	global weights
	weights = {}
	xavier_init = tf.contrib.layers.xavier_initializer()

	# filter for deconv3d: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]
	weights['wg1'] = tf.get_variable("wg1", shape=[z_size,3*3*3*384], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 192, 384], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 3, 192], initializer=xavier_init)
	
def init_biases():
	
	global biases
	biases = {}
	zero_init = tf.zeros_initializer()

	biases['bg1'] = tf.get_variable("bg1", shape=[3*3*3*384], initializer=zero_init)
	biases['bg2'] = tf.get_variable("bg2", shape=[192], initializer=zero_init)
	biases['bg3'] = tf.get_variable("bg3", shape=[3], initializer=zero_init)
   
def generator(z, batch_size=batch_size,phase_train=True, ):
	strides = [1,2,2,2,1]

	output_shape = {
		'g1':[batch_size,3,3,3,384],
		'g2':[batch_size,5,5,5,192],
		'g3':[batch_size,9,9,9,3],
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

		d1 = ly.conv3d(inputs=inputs, filters=192, kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d1 shape: ", d1.shape

		d2 = ly.conv3d(inputs=d1, filters=384, kernel_size=kernel_d,
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

	g_loss_sum = tf.summary.scalar("g_loss", g_loss)
	d_loss_sum = tf.summary.scalar("c_loss", d_loss)

	theta_g = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
	theta_d = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

	counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	opt_g = cly.optimize_loss(loss=g_loss, learning_rate=learning_rate_gen,
					optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_g, global_step=counter_g,
					summaries = ['gradient_norm'])
	counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	opt_d = cly.optimize_loss(loss=d_loss, learning_rate=learning_rate_dis,
					optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_d, global_step=counter_d,
					summaries = ['gradient_norm'])

	
	clipped_var_d = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_d]
	# merge the clip operations on critic variables
	with tf.control_dependencies([opt_d]):
		opt_d = tf.tuple(clipped_var_d)


	return opt_g, opt_d

def main():
	merged_all = tf.summary.merge_all()
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8

	with tf.device(device_gpu):
		weights = init_weights()
		biases = init_biases()
		cp_batch = load_data(record_path, n_epoch, batch_size, tuple(shape_size))
		opt_g, opt_d = build_graph(cp_batch)

	with tf.Session(config=config) as sess:
		summary_writer = tf.summary.FileWriter(log_path, sess.graph)
		init_op = tf.local_variables_initializer()
		sess.run(init_op)
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		print "Finish Building Graph"
		print "Finish Init and Start Training Step"

		saver = tf.train.Saver()

		for i in range(max_iter_step):
			if i < 25 or i % 500 == 0:
				citers = 100
			else:
				citers = Citers

			print "step: %r of total step %r" % (i, max_iter_step)

			for j in range(citers):
				print "citers %r of %r during step %r" % (j, citers, i)
				if i % 100 == 99 and j == 0:
				
					run_options = tf.RunOptions(
						trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					_, merged = sess.run([opt_d, merged_all],
										 options=run_options, run_metadata=run_metadata)

					summary_writer.add_summary(merged, i)
					summary_writer.add_run_metadata(
						run_metadata, 'critic_metadata {}'.format(i), i)
				else:
		
					sess.run([opt_d])

			if i % 100 == 99:
				_, merged = sess.run([opt_g, merged_all],
					 options=run_options, run_metadata=run_metadata)
				summary_writer.add_summary(merged, i)
				summary_writer.add_run_metadata(
					run_metadata, 'generator_metadata {}'.format(i), i)
			else:
				sess.run([opt_g])                
			if i % 1000 == 999:
				saver.save(sess, os.path.join(
					ckpt_dir, "model.ckpt"), global_step=i)

if __name__ == '__main__':
	main()
