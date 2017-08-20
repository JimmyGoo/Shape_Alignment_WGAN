import tensorflow as tf
from tensorflow import layers as ly
import tensorflow.contrib.layers as cly
from util import *
from functools import partial
from sys import argv
from model import *

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

LEARNING_RATE_GEN = 0.0025
LEARNING_RATE_DIS = 1e-5
BETA1 = 0.5
BETA2 = 0.9
SHAPE_SIZE = [9,9,9,3]

BATCH_SIZE = 64 # Batch size
LENGTH1 = 8
LENGTH2 = 8

G_EXTRA_STEP = 500
D_EXTRA_STEP = 20

ITERS = 50000 # How many generator iterations to train for
OUTPUT_DIM = SHAPE_SIZE[0] * SHAPE_SIZE[1] * SHAPE_SIZE[2] * SHAPE_SIZE[3] # Number of pixels in  (3*9*9*9)
Z_SIZE = 10
MERGE = 200
PRINT = 50

VIS_SHOW = 2000
VIS_SAVE = 10000

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

CONFIGURATION = [
	{
		'config_name': 'chair_lz_gan4_reg',
		'log_path': './log/chair_lz_gan4_reg_' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_gan4_reg/',
		'model_file_name': 'model_' + str(Z_SIZE),
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_gan4_reg_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': True
	},

	{
		'config_name': 'chair_lz_gan4',
		'log_path': './log/chair_lz_gan4' + str(Z_SIZE) + '/',
		'record_path': './data/tfrecord/chair_lz/',
		'model_path': './model/chair_lz_gan4/',
		'model_file_name': 'model_' + str(Z_SIZE),
		'bs_path': './data/bsCoeff/chair_lz_bsCoeff.mat',
		'vis_path': './vis/chair_lz_gan4_' + str(Z_SIZE) + '/',
		'SAMPLE_RATE': 1,
		'MODE': 1,
		'REGULARIZE': False
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

n_epoch = 2000

RESUME = False
REGULARIZE = current_config['REGULARIZE']
REG_LAMDA = 10
REG_LIMIT = 500

def build_graph(real_cp):
	real_cp = tf.reshape(real_cp, [BATCH_SIZE, OUTPUT_DIM])
	fake_cp = generator(BATCH_SIZE, output_shape, Z_SIZE)

	true_logit = discriminator(real_cp, filter_num_d, output_shape, BATCH_SIZE)
	fake_logit = discriminator(fake_cp, filter_num_d, output_shape, BATCH_SIZE, reuse=True)

	g_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
	print "d_p"
	d_params = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

	true_logit = tf.maximum(tf.minimum(true_logit, 0.99), 0.01)
	fake_logit = tf.maximum(tf.minimum(fake_logit, 0.99), 0.01)
	summary_true_hist = tf.summary.histogram("d_prob_true", true_logit)
	summary_fake_hist = tf.summary.histogram("d_prob_fack", fake_logit)

	g_loss = -tf.reduce_mean(tf.log(fake_logit))
	d_loss = -tf.reduce_mean(tf.log(true_logit) + tf.log(1-fake_logit)) # loss for discriminator

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

	d_real_conf = tf.reduce_mean(true_logit)
	d_fake_conf = tf.reduce_mean(fake_logit)

	summary_real_conf = tf.summary.scalar("real_conf", d_real_conf)
	summary_fake_conf = tf.summary.scalar("fake_conf", d_fake_conf)

	fimg = tf.placeholder(tf.float32)
	fake_img_summary = tf.summary.image('fake', fimg, max_outputs=10)

	g_loss_sum = tf.summary.scalar("g_loss", g_loss)
	d_loss_sum = tf.summary.scalar("d_loss", d_loss)

	gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_GEN, beta1=BETA1, beta2=BETA2).minimize(g_loss, var_list=g_params)
	disc_train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE_DIS).minimize(d_loss, var_list=d_params)

	merge_no_img = tf.summary.merge([summary_real_conf,summary_fake_conf,summary_fake_hist,summary_true_hist, g_loss_sum, d_loss_sum])

	return gen_train_op, disc_train_op,	g_loss, reg_loss, d_loss, d_real_conf, d_fake_conf, fimg, merge_no_img, fake_img_summary

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

		merged = sess.run(real_img_summary, feed_dict={rimg:rvimg[:6]})
		summary_writer.add_summary(merged, 1)

		for iteration in xrange(ITERS):
			# sess.run(disc_train_op)
			# sess.run(gen_train_op)
			if sess.run(real_conf) < 0.5:
				for j in range(D_EXTRA_STEP):
					sess.run(disc_train_op)
			elif sess.run(real_conf) < 0.8:
				sess.run(disc_train_op)

			if sess.run(fake_conf) < 0.4 and sess.run(real_conf) > 0.9:
				for j in range(G_EXTRA_STEP):
					sess.run(gen_train_op)
			else:
				sess.run(gen_train_op)
   
			if iteration % PRINT == PRINT - 1:
				print "step: %r of total step %r" % (iteration+1, ITERS)

				fc, rc = sess.run([fake_conf, real_conf])
				gl, dl = sess.run([g_loss, d_loss])
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
				##fake
				fcp = sess.run(fake_cp)
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
