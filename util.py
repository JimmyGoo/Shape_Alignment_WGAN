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

def load_data(record_path, n_epoch, batch_size, shape_size):
	filenames = [record_path + name for name in os.listdir(record_path) if name != '.DS_Store']
	print "loading tfrecord: ", filenames
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
	shape = read_and_decode(filename_queue, shape_size)
	shape_batch = tf.train.batch([shape], batch_size=batch_size)
	return shape_batch

def clear_file(path, suffix):
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

def vis_image(bsCoeff, CP_batch, step, vis_path=None):
	images_raw = np.array([np.matmul(bsCoeff, cp) for cp in CP_batch])

	images = [plot_to_image(i) for i in images_raw]

	if vis_path != None:
		###create combine thumbnail
		length = 8
		thumb_size = (256,256)
		new_im = Image.new('RGB', (thumb_size[0] * length, thumb_size[1] * length))
		for i in range(length):
			for j in range(length):
				img = Image.fromarray(images[8*i+j])
				img.thumbnail(thumb_size)
				new_im.paste(img, (i*thumb_size[0],j*thumb_size[1])) 
		new_im.save(vis_path + str(step) + '.png')
	return images

def plot_to_image(image):
	fig = Figure()
	canvas = FigureCanvas(fig)
	ax = Axes3D(fig)
	ax.scatter(image[0,:],image[1,:],image[2,:],c='r')
	canvas.draw()
	img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
	width, height = (fig.get_size_inches() * fig.get_dpi())
	img = np.reshape(img, (int(height),int(width),3))
	return img



