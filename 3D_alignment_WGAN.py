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
batch_size = 5
z_size = 10

learning_rate_gen = 5e-5
learning_rate_dis = 5e-5
shape_size = [13,13,13,3]

max_iter_step = 2

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

is_adam = True

Citers = 5

log_path = './log/chair/'
record_path = './data/tfrecord/'

n_epoch = 50

clamp_lower = -0.01
clamp_upper = 0.01

def generator(z, batch_size=batch_size,phase_train=True, ):
	#input 10(pending) dim Z latent space
	#Pending
	stride_g = [2,2,2]
	kernel_g1 = [4,4,4]
	kernel_g2 = [8,8,8] 
	kernel_g3 = [16,16,16]
	kernel_g4 = [32,32,32]
	kernel_g5 = [64,64,64]

	with tf.variable_scope("generator"):
		print "z shape: ", z.shape

		net = tf.reshape(z, (batch_size,1,1,1,z_size))
		g1 = ly.conv3d_transpose(inputs=net, filters=384, kernel_size=kernel_g1, strides=stride_g
			, padding="VALID", activation=tf.nn.relu, activity_regularizer=cly.batch_norm, 
			kernel_initializer=xavier_init)

		print "g1 shape: ", g1.shape

		g2 = ly.conv3d_transpose(inputs=g1, filters=192, kernel_size=kernel_g2, strides=stride_g
			, padding="SAME", activation=tf.nn.relu, activity_regularizer=cly.batch_norm, 
			kernel_initializer=xavier_init)

		print "g2 shape: ", g2.shape

		g3 = ly.conv3d_transpose(inputs=g2, filters=3, kernel_size=kernel_g3, strides=stride_g
			, padding="SAME", activation=tf.nn.tanh, kernel_initializer=xavier_init)

		print "g3 shape: ", g3.shape

		# g4 = ly.conv3d_transpose(inputs=g3, filters=64, kernel_size=kernel_g4, strides=stride_g
		# 	, padding="SAME", activation=tf.nn.relu, activity_regularizer=cly.batch_norm, 
		# 	kernel_initializer=xavier_init)

		# print "g4 shape: ", g4.shape

		# g5 = ly.conv3d_transpose(inputs=g4, filters=3, kernel_size=kernel_g5, strides=stride_g
		# 	, padding="SAME", activation=tf.nn.tanh, kernel_initializer=xavier_init)

		# print "g5 shape: ", g5.shape

	return g3

def discriminator(inputs, phase_train=True, reuse=False):
	stride_d = [2,2,2]
	stride_d5 = [1,1,1]
	kernel_d = [4,4,4]

	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()

		print "inputs shape: ", inputs.shape

		d1 = ly.conv3d(inputs=inputs, filters=3, kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d1 shape: ", d1.shape

		d2 = ly.conv3d(inputs=d1, filters=192, kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d2 shape: ", d2.shape

		d3 = ly.conv3d(inputs=d2, filters=384, kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d3 shape: ", d3.shape

		# d4 = ly.conv3d(inputs=d3, filters=512, kernel_size=kernel_d,
		# 	strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
		# 	kernel_initializer=xavier_init)

		# print "d4 shape: ", d4.shape

		d4 = cly.fully_connected(tf.reshape(
			d3, [batch_size, -1]), 1, activation_fn=None)

		print "d4 shape: ", d4.shape

	return d4

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
	
	with tf.Session(config=config) as sess:
		summary_writer = tf.summary.FileWriter(log_path, sess.graph)
	
		cp_batch = load_data(record_path, n_epoch, batch_size, tuple(shape_size))
		print cp_batch
		with tf.device(device_cpu):
			opt_g, opt_d = build_graph(cp_batch)

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
