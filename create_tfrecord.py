import os
import tensorflow as tf
from sys import argv
import scipy.io
import numpy as np

size = [9,9,9,3]
valid = True

def generate_chair(file_path, record_path):
	path = file_path
	if path[len(path) - 1] != '/':
		path += '/'

	counter = 0
	name = path.split('/')[-2]

	record = record_path + name + '.tfrecords'
	writer = tf.python_io.TFRecordWriter(record)

	for name in os.listdir(path):
		if name == '.DS_Store':
			continue

		mat = scipy.io.loadmat(path + name)
		
		curCP = np.array(mat['disCP'])
		curCP = np.transpose(curCP)
		print curCP.shape
		curCP = np.reshape(curCP, size)
		curCP_raw = curCP.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			'disCP': tf.train.Feature(bytes_list=tf.train.BytesList(value=[curCP_raw])),
		}))

		writer.write(example.SerializeToString())
		counter += 1

	writer.close()
	print "finish writing tfrecord file: %r, len %r" % (record, counter)


def generate_skull(filename, record_path):
	name = 'skull'
	record = record_path + name + '.tfrecords'
	writer = tf.python_io.TFRecordWriter(record)
	record_t = record_path + name + '_test.tfrecords'
	writer_t = tf.python_io.TFRecordWriter(record_t)

	mat = scipy.io.loadmat(filename)
	guidedSpace = np.array(mat['guidedSpace'])
	counter = 0
	counter_t = 0
	length = len(guidedSpace)

	for (i,dis) in enumerate(guidedSpace):
		#self
		if i == 0:
			continue

		dis_field = np.reshape(dis, size)
		dis_raw = dis_field.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			'disCP': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dis_raw])),
		}))
		if i < length - 10:
			writer.write(example.SerializeToString())
			counter += 1
		else:
			writer_t.write(example.SerializeToString())
			counter_t += 1

	writer.close()
	writer_t.close()
	print "finish writing tfrecord file: %r, train len %r, test len %r" % (record, counter, counter_t)

if __name__ == "__main__":
	if argv[3] == str(0):
		generate_chair(argv[1], argv[2])

	elif argv[3] == str(1):
		generate_skull(argv[1], argv[2])

	else:
		print 'invalid argu3'