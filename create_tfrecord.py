import os
import tensorflow as tf
from sys import argv
import scipy.io
import numpy as np

def generate(file_path, record_path):
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
		#3*n
		curCP = np.array(mat['dstCP'])
		offset = np.nanmin(curCP, axis=1)
		curCP -= np.reshape(offset, (3,1))
		maxVal = np.max(curCP)
		curCP *= int(64 / maxVal)
		curCP = curCP.astype('int')
		backGround = np.zeros([64,64,64])
		backGround[curCP[0,:], curCP[1,:], curCP[2,:]] += 1

		curCP_raw = curCP.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			'dstCP': tf.train.Feature(bytes_list=tf.train.BytesList(value=[curCP_raw])),
		}))

		writer.write(example.SerializeToString())
		counter += 1

	writer.close()
	print "finish writing tfrecord file: %r, len %r" % (record, counter)

if __name__ == "__main__":
	generate(argv[1], argv[2])