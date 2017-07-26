import tensorflow as tf
import scipy.io as sio
import numpy as np
import os

def leaky_relu(x, leak=0.3, name="lrelu"):
	with tf.variable_scope(name):
		return tf.maximum(x, x * leak)

def deform(BSCoeff_path, dstCP):
	deform_shape = tf.matmul(BSCoeff_path, dstCP)
	return deform_shape

def loadBscoeff(BSCoeff_path):
	#to DO
	BSCoeff = sio.loadmat(BSCoeff_path)
	BSCoeff = BSCoeff['BSCoeff']
	shape = None
	return tf.constant(BSCoeff_path)

def read_and_decode(queue, D, W, H):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(queue)
	# get feature from serialized example
	# decode
	features = tf.parse_single_example(
		serialized_example,
		features={
			'dstCP': tf.FixedLenFeature([], tf.string)
		}
	)
	shape = features['dstCP']
	shape = tf.decode_raw(shape, tf.uint8)
	shape = tf.reshape(shape, [D, W, H, 1])
	shape = tf.cast(shape, tf.float32)
	return shape

def load_data(record_path, n_epoch, batch_size):
	filenames = [record_path + name for name in os.listdir(record_path) if name != '.DS_Store']
	print "loading tfrecord: ", filenames
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=n_epoch)
	shape = read_and_decode(filename_queue, 64, 64, 64)
	shape_batch = tf.train.batch(
	  [shape], batch_size=batch_size)
	return shape_batch