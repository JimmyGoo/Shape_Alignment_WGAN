import tensorflow as tf
from tensorflow import layers as ly
import tensorflow.contrib.layers as cly
from util import *
from functools import partial
from sys import argv

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

xavier_init = cly.xavier_initializer()

batch_size = 64
z_size = 32

learning_rate_gen = 2e-4

learning_rate_dis = 2e-4
shape_size = [9,9,9,3]

max_iter_step = 200000

g_extra_step = 100
d_extra_step = 5

VIS_SAVE = 50

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

log_path = './log/chair/dcgan/'
record_path = './data/tfrecord/'
model_path = './model/chair/'
model_path = './model/chair/'
model_file_name = 'model_dcgan_' + str(shape_size[0])
bs_path = './data/bsCoeff/1_bsCoeff.mat'
vis_path = './vis/'

n_epoch = 5000

resume = False

filter_num_d = {
	'1':3,
	'2':8, 
	'3':16,
	'4':32,
}

def init_weights():

	global weights
	weights = {}
	xavier_init = tf.contrib.layers.xavier_initializer()

	# filter for deconv3d: A 5-D Tensor with the same type as value and shape [depth, height, width, output_channels, in_channels]
	weights['wg1'] = tf.get_variable("wg1", shape=[z_size,2*2*2*filter_num_d['4']], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, filter_num_d['3'], filter_num_d['4']], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, filter_num_d['2'], filter_num_d['3']], initializer=xavier_init)
	weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, filter_num_d['1'], filter_num_d['2']], initializer=xavier_init)
	
def init_biases():
	
	global biases
	biases = {}
	zero_init = tf.zeros_initializer()

	biases['bg1'] = tf.get_variable("bg1", shape=[2*2*2*filter_num_d['4']], initializer=zero_init)
	biases['bg2'] = tf.get_variable("bg2", shape=[filter_num_d['3']], initializer=zero_init)
	biases['bg3'] = tf.get_variable("bg3", shape=[filter_num_d['2']], initializer=zero_init)
	biases['bg4'] = tf.get_variable("bg4", shape=[filter_num_d['1']], initializer=zero_init)
   
def generator(z, batch_size=batch_size,phase_train=True, reuse=False):
	strides = [1,2,2,2,1]

	output_shape = {
		'g1':[batch_size,2,2,2,filter_num_d['4']],
		'g2':[batch_size,3,3,3,filter_num_d['3']],
		'g3':[batch_size,5,5,5,filter_num_d['2']],
		'g4':[batch_size,9,9,9,filter_num_d['1']],
	}

	with tf.variable_scope("generator") as scope:
		if reuse:
			scope.reuse_variables()

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

		## one layer deeper

		g3 = tf.nn.conv3d_transpose(g2, weights['wg3'], output_shape=output_shape['g3'], strides=strides, padding="SAME")
		g3 = tf.nn.bias_add(g3, biases['bg3'])
		g3 = tf.contrib.layers.batch_norm(g3, is_training=phase_train)
		g3 = tf.nn.relu(g3)

		print "g3 shape: ", g3.shape
		
		g4 = tf.nn.conv3d_transpose(g3, weights['wg4'], output_shape=output_shape['g4'], strides=strides, padding="SAME")
		g4 = tf.nn.bias_add(g4, biases['bg4'])                                   
		g4 = tf.tanh(g4)

		print "g4 shape: ", g4.shape

	return g4



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

		##one layer deeper

		d3 = ly.conv3d(inputs=d2, filters=filter_num_d['4'], kernel_size=kernel_d,
			strides=stride_d, padding="SAME",activation=leaky_relu, activity_regularizer=cly.batch_norm,
			kernel_initializer=xavier_init)

		print "d3 shape: ", d3.shape

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

	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit),logits=fake_logit))
	d_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))
	d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logit), logits=true_logit))
	d_loss /= 2.

	print "true logit shape: ", true_logit.shape
	print "fake logit shape: ", fake_logit.shape

	d_real_conf = tf.divide(tf.reduce_sum(tf.maximum(tf.minimum(true_logit, 0.99), 0.01)), batch_size)
	d_fake_conf = tf.divide(tf.reduce_sum(tf.maximum(tf.minimum(fake_logit, 0.99), 0.01)), batch_size)

	summary_real_conf = tf.summary.scalar("real_conf", d_real_conf)
	summary_fake_conf = tf.summary.scalar("fake_conf", d_fake_conf)

	# rimg = tf.placeholder(tf.float32)
	fimg = tf.placeholder(tf.float32)
	# real_img_summary = tf.summary.image('real', rimg, max_outputs=batch_size)
	fake_img_summary = tf.summary.image('fake', fimg, max_outputs=10)

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

	opt_g = tf.train.AdamOptimizer(learning_rate=learning_rate_gen, beta1=0.5).minimize(g_loss, var_list=g_params)
	opt_d = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=0.5).minimize(d_loss, var_list=d_params)

	return opt_g, opt_d, g_loss, d_loss, d_real_conf, d_fake_conf, fake_cp, fimg

def main():
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True
	# config.gpu_options.per_process_gpu_memory_fraction = 0.8

	clear_file(log_path, '.g01')
	clear_file(vis_path, '.png')
	print "Clear File"
	bsCoeff = load_bsCoeff(bs_path)

	with tf.device(device_gpu):
		weights = init_weights()
		biases = init_biases()
		cp_batch = load_data(record_path, n_epoch, batch_size, tuple(shape_size))
		opt_g, opt_d, g_loss, d_loss, real_conf, fake_conf, fake_cp, fimg = build_graph(cp_batch)
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
				d_extra_step = 20

			rc = sess.run(real_conf)
			gl = sess.run(g_loss)
			if rc < 0.9:
				if rc < 0.7 and gl < 0.65:
					for j in range(d_extra_step):
						sess.run([opt_d])
						if j % 5 == 4:
							print "d_extra: %r of total %r during step %r real_conf %r, d_loss %r" % (j+1, 
								d_extra_step, i+1, sess(real_conf), sess.run(d_loss))
				else:
					sess.run([opt_d])


			fc = sess.run(fake_conf)
			if fc < 0.2:
				g_extra_step = 300

			if fc < 0.5 and sess.run(real_conf) > 0.8:
				for j in range(g_extra_step):
					sess.run([opt_g])
					if j % 50 == 49:
						print "g_extra: %r of total %r during step %r, fake_conf %r g_loss %r" % (j+1, 
							g_extra_step, i+1, sess.run(fake_conf), sess.run(g_loss))
			else:
				sess.run([opt_g])

			
			print "step: %r of total step %r" % (i+1, max_iter_step)
			print "fake_conf %r g_loss %r, real_conf %r d_loss %r" % (sess.run(fake_conf), sess.run(g_loss), sess.run(real_conf), sess.run(d_loss))

			if i % VIS_SAVE == VIS_SAVE-1:
				with tf.device(device_cpu):
					fcp = sess.run(fake_cp)
					fcp = np.reshape(fcp, (batch_size,-1,3))
					fvimg = vis_image(bsCoeff, fcp, i+1, 10, vis_path)
				merged = sess.run(merged_all, feed_dict={fimg:fvimg})
				summary_writer.add_summary(merged, i)

			if i % VIS_SAVE == VIS_SAVE-1:
				if not os.path.exists(model_path):
					os.mkdir(model_path)
				saver.save(sess, model_path+model_file_name+".ckpt", global_step=i)
				print "saving model of step " + str(i) + "!"*10 

if __name__ == '__main__':
	main()
