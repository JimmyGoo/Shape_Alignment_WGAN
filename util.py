import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import PIL
from PIL import Image

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
	shape = tf.decode_raw(shape, tf.float64)
	shape = tf.reshape(shape, size)
	shape = tf.cast(shape, tf.float32)
	return shape

def read_and_decode_dis(queue, size):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(queue)
	# get feature from serialized example
	# decode
	features = tf.parse_single_example(
		serialized_example,
		features={
			'disCP': tf.FixedLenFeature([], tf.string)
		}
	)
	shape = features['disCP']
	shape = tf.decode_raw(shape, tf.float64)
	shape = tf.reshape(shape, size)
	shape = tf.cast(shape, tf.float32)
	return shape

def load_data(record_path, n_epoch, batch_size, shape_size, mode):
	filenames = [record_path + name for name in os.listdir(record_path) if name != '.DS_Store']
	print "loading tfrecord: ", filenames
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
	if mode == 0:
		shape = read_and_decode(filename_queue, shape_size)
	elif mode == 1:
		shape = read_and_decode_dis(filename_queue, shape_size)
	else:
		print 'invalid mode in load_data'
	shape_batch = tf.train.batch([shape], batch_size=batch_size)
	return shape_batch


def load_test_data(test_path, shape_size, mode, sess):
	filename = [test_path + name for name in os.listdir(test_path) if name != '.DS_Store']
	print "loading test: ", filename
	filename_queue = tf.train.string_input_producer(filename)
	if mode == 0:
		shape = read_and_decode(filename_queue, shape_size)
	elif mode == 1:
		shape = read_and_decode_dis(filename_queue, shape_size)
	else:
		print 'invalid mode in load_data'
	init_op = tf.initialize_local_variables()
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	count = 0
	test_shape = []
	record_num = np.sum([1 for s in tf.python_io.tf_record_iterator(filename[0])])

	for i in range(record_num):
		example = sess.run(shape)
		test_shape.append(example)
		count += 1

	coord.request_stop()
	coord.join(threads)
	print "test size: ", count
	return np.array(test_shape)

def clear_file(path, suffix):
	if not os.path.exists(path):
		os.mkdir(path)
		return
		
	for f in os.listdir(path):
		if f[-4:] == suffix:
			name = path + f
			os.remove(name)
			print "file deleted: ",name

def load_bsCoeff(path):
	mat = sio.loadmat(path)
		
	bs = np.array(mat['bsCoeff'])
	bs = np.array([np.reshape(np.array(b[0]), len(b[0])) for b in bs])

	return bs

def load_bsCoeff_cp(path):
	mat = sio.loadmat(path)
		
	bs = np.array(mat['bsCoeff'])
	bs = np.array([np.reshape(np.array(b[0]), len(b[0])) for b in bs])
	cp = np.array(mat['orgCP'])
	cp = np.transpose(cp)
	return bs, cp

def vis_image(bsCoeff, CP_batch, step, real=False):
	print "visualizing of step:", step

	images_raw = np.array([np.matmul(bsCoeff, cp) for cp in CP_batch])

	images = [plot_to_image(i, real) for i in images_raw]

	return images


def vis_image_displacement(bsCoeff, ocp, dis_batch, step, real=False):
	print "visualizing of step:", step

	images_raw = np.array([np.matmul(bsCoeff, cp + ocp) for cp in dis_batch])

	images = [plot_to_image(i, real) for i in images_raw]

	return images

def save_vis_image(images, step, save_path, length, length2, thumb_size=(256,256)):
	###create combine thumbnail
	new_im = Image.new('RGB', (int(thumb_size[0] * length), int(thumb_size[1] * length2)))
	for i in range(length):
		for j in range(length2):
			img = Image.fromarray(images[8*i+j])
			img.thumbnail(thumb_size)
			new_im.paste(img, (i*thumb_size[0],j*thumb_size[1])) 
	new_im.save(save_path + str(step) + '.png')

def plot_to_image(image, real):
	fig = Figure()
	canvas = FigureCanvas(fig)
	ax = Axes3D(fig)
	if real:
		ax.scatter(image[:,0],image[:,1],image[:,2],c='g')
	else:
		ax.scatter(image[:,0],image[:,1],image[:,2],c='r')

	canvas.draw()
	img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
	width, height = (fig.get_size_inches() * fig.get_dpi())
	img = np.reshape(img, (int(height),int(width),3))
	return img

def sample_skull_points(bsCoeff,sample_rate):
	if sample_rate == 1:
		return bsCoeff
		
	length = len(bsCoeff)
	sample_idx = [int(i*sample_rate) for i in range(int(length / sample_rate))]
	return bsCoeff[sample_idx]

def test_batch(cp, name):
	cp0 = cp[0]
	count = 0
	for c in cp:
		if np.all(c == cp0):
			count += 1

	mean = np.mean(np.reshape(cp,[-1,3]), axis=0)

	print name + ' offset: ', mean
	if count > 1:
		print name + " data duplicated! count: ", count
	else:
		print name + " data seems good! count: ", count
