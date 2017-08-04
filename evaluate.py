import tensorflow as tf
import numpy as np
import shape_alignment_improvedWGAN
from sys import argv
from util import *

test_path = ''
model_path = 
model_file_name = 
shape_size = [9,9,9,3]
mode = 1

def evaluation_space_angle(tcp, fcp):
	angles = []
	for y in fcp:
		base = np.reshape(tcp, shape=[-1, 9*9*9*3])
		x = np.linalg.solve(base, y)
		xv = y - np.dot(base,x)

	    xv_u = xv / np.linalg.norm(xv)
	    y_u = y / np.linalg.norm(y)
	    angles.append(np.arccos(np.clip(np.dot(xv_u, y_u), -1.0, 1.0)))

	angels = np.degrees(angels)
	angels_m = np.mean(angels)
	return  

	
def evaluation_L2norm(tcp, fcp):
	print "fcp shape: ", fcp.shape
	print "tcp shape: ", tcp.shape
	L2norm = []
	L2norm_mean = []

	for f in fcp:
		L2norm.append(f - tcp)
		L2norm_mean.append(np.mean(f - tcp))

	return np.mean(L2norm), np.mean(L2norm_mean)

def main()
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    
    with tf.device(device_gpu):
        fack_cp = Generator(64, phase_train=False, noise=None, reuse=True)

    with tf.Session(config=config) as sess:
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_path + model_file_name)
        
        fcp = sess.run(fack_cp)

        tcp = load_test_data(test_path, shape_size, mode, sess):

        evaluation_L2norm(tcp, fcp)
        evaluation_space_angle(tcp, fcp)

if __name__ == '__main__':
    main()