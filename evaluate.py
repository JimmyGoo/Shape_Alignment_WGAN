import tensorflow as tf
import numpy as np
from model import *
from sys import argv
from util import *

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

TEST_PATH = './data/tfrecord/skull/test/'
MODEL_PATH = './model/skull/'
SHAPE_SIZE = [9,9,9,3]
MODE = 1
BATCH_SIZE = 64
Z_SIZE = 40

step = 20000

MODEL_FILE_NAME = 'model_iwgan_skull_' + str(Z_SIZE) + '_9.ckpt-' + str(step)

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

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

def evaluation_space_angle(tcp, fcp):
	angles = []
	for y in fcp:
		base = np.reshape(tcp, [-1, 9*9*9*3])
		base = np.transpose(base)
		y = np.reshape(y, [9*9*9*3])
		x,_,_,_ = np.linalg.lstsq(base, y)
		xv = y - np.dot(base,x)
		xv_u = xv / np.linalg.norm(xv)
		y_u = y / np.linalg.norm(y)
		angles.append(np.arccos(np.clip(np.dot(xv_u, y_u), -1.0, 1.0)))

	angles = np.degrees(angles)
	angles_mean = np.mean(angles)
	return  angles, angles_mean

	
def evaluation_L2norm(tcp, fcp):
	print "fcp shape: ", fcp.shape
	print "tcp shape: ", tcp.shape
	L2norm = []
	tcp_num = tcp.shape[0]

	for f in fcp:
		diff = f - tcp
		L2norm.append(diff)

	L2norm = np.array(L2norm)

	L2norm = np.mean(L2norm,axis=0)
	L2norm = np.reshape(L2norm, (tcp_num, -1, 3))

	L2norm = L2norm ** 2
	L2norm = np.sum(L2norm, axis=2)
	L2norm = np.sqrt(L2norm)
	L2norm = np.mean(L2norm, axis = 1)

	return L2norm, np.mean(L2norm)

def main():
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True

	with tf.device(device_gpu):
		weights = init_weights(filter_num_d, output_shape, Z_SIZE)
		biases = init_biases(filter_num_d, output_shape)
		fack_cp = Generator(BATCH_SIZE, output_shape, Z_SIZE, phase_train=False)

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		print("Load existing model " + "!"*10)
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_PATH + MODEL_FILE_NAME)
		fcp = sess.run(fack_cp)
		tcp = load_test_data(TEST_PATH, SHAPE_SIZE, MODE, sess)

	
	with tf.device(device_cpu):
		fcp = np.reshape(fcp, [BATCH_SIZE] + SHAPE_SIZE)
		L2norm, L2norm_mean = evaluation_L2norm(tcp, fcp)
		print "L2norm: ", L2norm
		print "L2norm_mean: ", L2norm_mean

		angles, angles_mean = evaluation_space_angle(tcp, fcp)
		print "angles: ", angles
		print "angles_mean: ", angles_mean

if __name__ == '__main__':
	main()