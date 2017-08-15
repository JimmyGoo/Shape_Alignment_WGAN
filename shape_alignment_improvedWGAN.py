import os, sys
import time
import numpy as np
import tensorflow as tf
from sys import argv
from util import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DIS = 1e-4
BETA1 = 0.5
BETA2 = 0.9
SHAPE_SIZE = [9,9,9,3]

LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many critic iterations per generator iteration

BATCH_SIZE = 64 # Batch size = length1 * length2
LENGTH1 = 8
LENGTH2 = 8

C_LOWER = -0.1
C_UPPER = -C_LOWER

ITERS = 50000 # How many generator iterations to train for
OUTPUT_DIM = SHAPE_SIZE[0] * SHAPE_SIZE[1] * SHAPE_SIZE[2] * SHAPE_SIZE[3] # Number of pixels in  (3*9*9*9)
Z_SIZE = 5
MERGE = 200
PRINT = 50

VIS_SHOW = 2000
VIS_SAVE = 10000

ADAM = False

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

CONFIGURATION = [
	{
		'config_name': 'skull_wgan',
		'log_path': './log/skull_wgan_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/skull/',
		'model_path': './model/skull_wgan_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/skull_bsCoeff.mat',
		'vis_path': './vis/skull_wgan_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 100,
		'MODE': 1,
		'REGULARIZE': False,
		'GP': False,
	},

	{
		'config_name': 'chair_wgan',
		'log_path': './log/chair_wgan_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair/',
		'model_path': './model/chair_wgan_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_bsCoeff.mat',
		'vis_path': './vis/chair_wgan_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': False,
		'GP': False,
	},

	{
		'config_name': 'chair_wgan_reg',
		'log_path': './log/chair_wgan_reg_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair/',
		'model_path': './model/chair_wgan_reg_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_bsCoeff.mat',
		'vis_path': './vis/chair_wgan_reg_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': True,
		'GP': False,
	},

	{
		'config_name': 'chair_lz_wgan_reg',
		'log_path': './log/chair_lz_wgan_reg_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_wgan_reg_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_wgan_reg_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': True,
		'GP': False,
	},

	{
		'config_name': 'chair_lz_wgan',
		'log_path': './log/chair_lz_wgan_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_wgan_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_wgan_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': False,
		'GP': False,
	},

	{
		'config_name': 'chair_lz_iwgan_reg',
		'log_path': './log/chair_lz_iwgan_reg_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_iwgan_reg_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_iwgan_reg_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': True,
		'GP': True,
	},

	{
		'config_name': 'chair_lz_iwgan',
		'log_path': './log/chair_lz_iwgan_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_iwgan_' + str(Z_SIZE) + '/',
		'model_file_name': 'model',
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_iwgan_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': False,
		'GP': True,
	},
]

current_config = CONFIGURATION[int(argv[2])]
SAMPLE_RATE = current_config['SAMPLE_RATE']
log_path = current_config['log_path']
record_path = current_config['record_path']
model_path = current_config['model_path']
model_file_name = current_config['model_file_name']
bs_path = current_config['bs_path']
vis_path = current_config['vis_path']
MODE = current_config['MODE']
GP = current_config['GP']
RESUME = False
REGULARIZE = current_config['REGULARIZE']
REG_LAMDA = 10
REG_LIMIT = 500

filter_num_d = {
	'1':3,
	'2':8, 
	'3':16,
	'4':32,
}

output_shape = {
	'g1':[BATCH_SIZE,2,2,2,filter_num_d['4']],
	'g2':[BATCH_SIZE,3,3,3,filter_num_d['3']],
	'g3':[BATCH_SIZE,5,5,5,filter_num_d['2']],
	'g4':[BATCH_SIZE,9,9,9,filter_num_d['1']]
}

n_epoch = 5000


def build_graph(real_cp):
	real_cp = tf.reshape(real_cp, [BATCH_SIZE, OUTPUT_DIM])
	fake_cp = generator(BATCH_SIZE, output_shape, Z_SIZE)

	true_logit = discriminator(real_cp, filter_num_d, output_shape, BATCH_SIZE)
	fake_logit = discriminator(fake_cp, filter_num_d, output_shape, BATCH_SIZE, reuse=True)

	g_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
	d_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

	# Standard WGAN loss
	gen_cost = -tf.reduce_mean(fake_logit)
	disc_cost = tf.reduce_mean(fake_logit) - tf.reduce_mean(true_logit)

	reg_loss = tf.constant(0)
	if REGULARIZE:
		real_mean = tf.reduce_mean(tf.reshape(real_cp, [-1,3]), axis=0)
		fack_mean = tf.reduce_mean(tf.reshape(fake_cp, [-1,3]), axis=0)

		reg_loss = tf.nn.l2_loss(real_mean - fack_mean)
		reg_loss = tf.minimum(REG_LAMDA * reg_loss, REG_LIMIT)
		gen_cost += reg_loss
		# reg_loss = tf.nn.l2_loss(fake_cp) / ftf.cast(tf.size(fake_cp), tf.float32)
		# reg_loss = REG_LAMDA * reg_loss
		# reg_loss = tf.minimum(reg_loss, REG_LIMIT)
		# gen_cost -= reg_loss
		reg_loss_sum = tf.summary.scalar("g_loss_reg", reg_loss)

	print "true logit shape: ", true_logit.shape
	print "fake logit shape: ", fake_logit.shape

	true_logit = tf.maximum(tf.minimum(true_logit, 0.99), 0.01)
	fake_logit = tf.maximum(tf.minimum(fake_logit, 0.99), 0.01)
	summary_true_hist = tf.summary.histogram("d_prob_true", true_logit)
	summary_fake_hist = tf.summary.histogram("d_prob_fack", fake_logit)

	d_real_conf = tf.reduce_mean(true_logit)
	d_fake_conf = tf.reduce_mean(fake_logit)
	summary_real_conf = tf.summary.scalar("real_conf", d_real_conf)
	summary_fake_conf = tf.summary.scalar("fake_conf", d_fake_conf)

	fimg = tf.placeholder(tf.float32)
	fake_img_summary = tf.summary.image('fake', fimg, max_outputs=10)

	g_loss_sum = tf.summary.scalar("g_loss", gen_cost)
	d_loss_sum = tf.summary.scalar("d_loss", disc_cost)

	# Gradient penalty
	if GP == True:
		alpha = tf.random_uniform(
			shape=[BATCH_SIZE,1], 
			minval=0.,
			maxval=1.
		)

		differences = fake_cp - real_cp
		interpolates = real_cp + (alpha*differences)
		gradients = tf.gradients(discriminator(interpolates, filter_num_d, output_shape, BATCH_SIZE,  reuse=True), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1.)**2)
		disc_cost += LAMBDA*gradient_penalty

	if ADAM == True:
		gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_GEN, beta1=BETA1, beta2=BETA2).minimize(gen_cost, var_list=g_params)
		disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_DIS, beta1=BETA1, beta2=BETA2).minimize(disc_cost, var_list=d_params)
	else:
		gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=g_params)
		disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=d_params)

	merge_no_img = tf.summary.merge([summary_real_conf, summary_fake_conf, summary_fake_hist,summary_true_hist, g_loss_sum, d_loss_sum])

	if GP == False:
		clipped_var_d = [tf.assign(var, tf.clip_by_value(var, C_LOWER, C_UPPER)) for var in d_params]
        with tf.control_dependencies([disc_train_op]):
        	if GP == False:
           		disc_train_op = tf.tuple(clipped_var_d)

	return gen_train_op, disc_train_op, gen_cost, reg_loss, disc_cost, d_real_conf, d_fake_conf, fimg, merge_no_img, fake_img_summary

# Train loop
def main():
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	# config.gpu_options.allow_growth = True
	# config.gpu_options.per_process_gpu_memory_fraction = 0.8
	print "Current Config: ", current_config['config_name']
	print "Loading bsCoeff: %r, Sample Rate %r" % (bs_path, SAMPLE_RATE)

	clear_file(log_path, '.g01')
	clear_file(vis_path, '.png')
	print "Clear File"

	if MODE == 0:
		bsCoeff = load_bsCoeff(bs_path)
	elif MODE == 1:
		bsCoeff, ocp = load_bsCoeff_cp(bs_path)
		bsCoeff = sample_skull_points(bsCoeff, SAMPLE_RATE)

	weights = init_weights(filter_num_d, output_shape, Z_SIZE)
	biases = init_biases(filter_num_d, output_shape)
	#displacement field
	cp_batch = load_data(record_path, n_epoch, BATCH_SIZE, tuple(SHAPE_SIZE), MODE)
	gen_train_op, disc_train_op, g_loss, reg_loss, d_loss, real_conf, fake_conf, fimg, merge_no_img, fake_img_summary = build_graph(cp_batch)
	fake_cp = generator(BATCH_SIZE, output_shape, Z_SIZE, reuse=True, phase_train=False)
	merged_all = tf.summary.merge_all()
	rimg = tf.placeholder(tf.float32)
	real_img_summary = tf.summary.image('real', rimg, max_outputs=5)
	print "Finish Building Graph"

	with tf.Session(config=config) as sess:
		init_op = tf.local_variables_initializer()
		sess.run(init_op)
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		summary_writer = tf.summary.FileWriter(log_path, sess.graph)

		print "Finish Init and Start Training Step"

		if RESUME:
			print("Load existing model " + "!"*10)
			saver = tf.train.Saver()
			saver.restore(sess, model_file_name)

		saver = tf.train.Saver(max_to_keep=None)

		##Vis real img
		#"Test Real img"
		rcp = sess.run(cp_batch)
		rcp = np.reshape(rcp, (BATCH_SIZE,-1,3))
		test_batch(rcp,'rcp')
		
		if MODE == 0:
			rvimg = vis_image(bsCoeff, rcp, 0, True)
		elif MODE == 1:
			rvimg = vis_image_displacement(bsCoeff, ocp, rcp, 0, True)
			# save_vis_image(rvimg, 0, vis_path, LENGTH1, LENGTH2)

		merged = sess.run(real_img_summary, feed_dict={rimg:rvimg[:10]})
		summary_writer.add_summary(merged, 1)

		for iteration in xrange(ITERS):
			# Train generator
			if iteration > 0:
				sess.run(gen_train_op)
		   
			if iteration < 50 or iteration % 500 == 0:
				DISC_ITERS = 25

			for i in xrange(DISC_ITERS):
				sess.run(disc_train_op)
				

			if iteration % PRINT == PRINT - 1:
				print "step: %r of total step %r" % (iteration+1, ITERS)

				fc, rc = sess.run([fake_conf, real_conf])
				if REGULARIZE:
					gl, rl, dl = sess.run([g_loss, reg_loss, d_loss])
					print "fake_conf %r g_loss %r reg_loss %r, real_conf %r d_loss %r" % (fc, gl, rl, rc, dl)
				else:
					gl, dl = sess.run([g_loss, d_loss])
					print "fake_conf %r g_loss %r, real_conf %r d_loss %r" % (fc, gl, rc, dl)
			
			if (iteration % MERGE == MERGE-1) and (iteration % VIS_SAVE != VIS_SAVE-1):
				merged_no_img = sess.run(merge_no_img)
				summary_writer.add_summary(merged_no_img, iteration+1)

			if iteration % VIS_SHOW == VIS_SHOW-1:

				fcp = sess.run(fake_cp)
				print fcp.shape
				fcp = np.reshape(fcp, (BATCH_SIZE,-1,3))
				test_batch(fcp,'fcp')

				if MODE == 0:
					fvimg = vis_image(bsCoeff, fcp, iteration+1)
				elif MODE == 1:
					fvimg = vis_image_displacement(bsCoeff, ocp, fcp, iteration+1)

				if iteration % VIS_SAVE == VIS_SAVE-1:
					save_vis_image(fvimg, iteration+1, vis_path, LENGTH1, LENGTH2)

				merged = sess.run(fake_img_summary, feed_dict={fimg:fvimg[:10]})
				summary_writer.add_summary(merged, iteration+1)

				#real
				rcp = sess.run(cp_batch)
				rcp = np.reshape(rcp, (BATCH_SIZE,-1,3))
				test_batch(rcp,'rcp')

				if MODE == 0:
					rvimg = vis_image(bsCoeff, rcp, iteration+2, True)
				elif MODE == 1:
					rvimg = vis_image_displacement(bsCoeff, ocp, rcp, iteration+2, True)

				merged = sess.run(real_img_summary, feed_dict={rimg:rvimg[:10]})
				summary_writer.add_summary(merged, iteration+1)

			if iteration % VIS_SAVE == VIS_SAVE-1:
				if not os.path.exists(model_path):
					os.mkdir(model_path)
				saver.save(sess, model_path+model_file_name+".ckpt", global_step=iteration + 1)
				print "finish saving model of step " + str(iteration + 1) + "!"*10

		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	main()
