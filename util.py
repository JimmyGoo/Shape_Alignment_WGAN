import tensorflow as tf
import scipy.io as sio
import numpy as np
import os

def leaky_relu(x, leak=0.3, name="lrelu"):
	with tf.variable_scope(name):
		return tf.maximum(x, x * leak)

def read_and_decode(queue, size):
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
	shape = tf.decode_raw(shape, tf.float32)
	shape = tf.reshape(shape, size)

	return shape

def load_data(record_path, n_epoch, batch_size, shape_size):
	filenames = [record_path + name for name in os.listdir(record_path) if name != '.DS_Store']
	print "loading tfrecord: ", filenames
	filename_queue = tf.train.string_input_producer(filenames)
	shape = read_and_decode(filename_queue, shape_size)
	
	shape_batch = tf.train.batch(
	  [shape], batch_size=batch_size)
	return shape_batch