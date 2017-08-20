import tensorflow as tf
import tensorlayer as tl
import tensorlayer.layers as ly

delving_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()
fc_init = tf.truncated_normal_initializer(stddev=0.005)

init_mode = 1
if init_mode == 0:
	current_init = xavier_init
elif init_mode == 1:
	current_init = delving_init
else:
	print "init mode error"

TANH = True

def generator(n_samples, output_shape, gen_filter_shape, Z_SIZE, is_training=True, noise=None, reuse=False):
	if noise is None:
		noise = tf.random_normal([n_samples, Z_SIZE])

	strides_g = [1,2,2,2,1]
	shape = output_shape['g1']
	with tf.variable_scope("generator", reuse=reuse) as scope:
		tl.layers.set_name_reuse(reuse)

		print "noise shape: ", noise.shape
		number_outputs = shape[1] * shape[2] * shape[3] * shape[4]
		inputs = ly.InputLayer(noise, name='gen_input')
		g1 = ly.DenseLayer(inputs, n_units=number_outputs, name='gen1_dense')
		g1 = ly.BatchNormLayer(g1, name='gen1_bn', is_train=is_training, act=tf.nn.relu)
		g1 = ly.ReshapeLayer(g1,shape=output_shape['g1'],name='gen1_reshape')

		g2 = ly.DeConv3dLayer(g1,shape=gen_filter_shape['g2'], output_shape=output_shape['g2'] ,strides=strides_g,padding="SAME", name='gen2_deconv3d')
		g2 = ly.BatchNormLayer(g2, name='gen2_bn', is_train=is_training, act=tf.nn.relu)

		g3 = ly.DeConv3dLayer(g2,shape=gen_filter_shape['g3'], output_shape=output_shape['g3'] ,strides=strides_g,padding="SAME", name='gen3_deconv3d')
		g3 = ly.BatchNormLayer(g3, name='gen3_bn', is_train=is_training, act=tf.nn.relu)

		g4 = ly.DeConv3dLayer(g3,shape=gen_filter_shape['g4'], output_shape=output_shape['g4'] ,strides=strides_g,padding="SAME", name='gen4_deconv3d')
		g4 = ly.BatchNormLayer(g4, name='gen4_bn', is_train=is_training)

		train_params = g4.all_params
		ly.print_all_variables(True)
		g4.print_layers()
		if TANH:
			outputs = tf.tanh(g4.outputs)
		else:
			outputs = g4.outputs

	return outputs, train_params

def discriminator(inputs, batch_size, dis_filter_shape, output_shape, is_training=True, reuse=False, GP=False):

	stride_d = [1,2,2,2,1]
	
	with tf.variable_scope("discriminator", reuse=reuse) as scope:
		tl.layers.set_name_reuse(reuse)

		inputs = tf.reshape(inputs, output_shape)
		inputs = ly.InputLayer(inputs, name='dis_input')

		d1 = ly.Conv3dLayer(inputs, shape=dis_filter_shape['d1'], strides=stride_d, padding="SAME", name='d1_conv3d')
		d1 = ly.BatchNormLayer(d1, name='dis1_bn', is_train=is_training, act= lambda x : tl.act.lrelu(x, 0.2))                       

		d2 = ly.Conv3dLayer(d1, shape=dis_filter_shape['d2'], strides=stride_d, padding="SAME", name='d2_conv3d')               
		d2 = ly.BatchNormLayer(d2, name='dis2_bn', is_train=is_training, act= lambda x : tl.act.lrelu(x, 0.2)) 


		d3 = ly.Conv3dLayer(d2, shape=dis_filter_shape['d3'], strides=stride_d, padding="SAME", name='d3_conv3d')              
		d3 = ly.BatchNormLayer(d3, name='dis3_bn', is_train=is_training, act= lambda x : tl.act.lrelu(x, 0.2)) 
		d3 = ly.ReshapeLayer(d3, shape=[batch_size, -1], name='dis3_reshape')

		d4 = ly.DenseLayer(d3, n_units=1, name='dis4_dense', act= lambda x : tl.act.lrelu(x, 0.2))

		train_params = d4.all_params
		ly.print_all_variables(True)
		d4.print_layers()
		outputs = d4.outputs

	return outputs, train_params
