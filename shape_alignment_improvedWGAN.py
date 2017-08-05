import os, sys
import time
import numpy as np
import tensorflow as tf
from sys import argv
from util import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"]=argv[1]

LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DIS = 2e-4
SHAPE_SIZE = [9,9,9,3]

LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = SHAPE_SIZE[0] * SHAPE_SIZE[1] * SHAPE_SIZE[2] * SHAPE_SIZE[3] # Number of pixels in  (3*9*9*9)
Z_SIZE = 40
MERGE = 50
PRINT = 50

VIS_SAVE = 2000

device_gpu = '/gpu:0'
device_cpu = '/cpu:0'

CONFIGURATION = [
    {
        'config_name': 'skull_config',
        'log_path': './log/skull_' + str(Z_SIZE) + '/',
        'record_path': './data/tfrecord/skull/',
        'model_path': './model/skull/',
        'model_file_name': 'model_iwgan_skull_' + str(Z_SIZE),
        'bs_path': './data/bsCoeff/skull_bsCoeff.mat',
        'vis_path': './vis/skull_' + str(Z_SIZE) + '/',
        'SAMPLE_RATE': 100,
        'MODE': 1
    },

    {
        'config_name': 'chair_config',
        'log_path': './log/chair_' + str(Z_SIZE) + '/',
        'record_path': './data/tfrecord/chair/',
        'model_path': './model/chair/',
        'model_file_name': 'model_iwgan_chair_' + str(Z_SIZE),
        'bs_path': './data/bsCoeff/chair_bsCoeff.mat',
        'vis_path': './vis/chair_' + str(Z_SIZE) + '/',
        'SAMPLE_RATE': 1,
        'MODE': 1
    }
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

n_epoch = 100000

resume = False

def build_graph(real_cp):
    real_cp = tf.reshape(real_cp, [BATCH_SIZE, OUTPUT_DIM])
    fake_cp = Generator(BATCH_SIZE, output_shape, Z_SIZE)

    disc_real = Discriminator(real_cp, filter_num_d, output_shape, BATCH_SIZE)
    disc_fake = Discriminator(fake_cp, filter_num_d, output_shape, BATCH_SIZE, reuse=True)

    g_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    d_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    print "true logit shape: ", disc_real.shape
    print "fake logit shape: ", disc_fake.shape


    d_real_conf = tf.reduce_mean(tf.sigmoid(disc_real))
    d_fake_conf = tf.reduce_mean(tf.sigmoid(disc_fake))

    summary_real_conf = tf.summary.scalar("real_conf", d_real_conf)
    summary_fake_conf = tf.summary.scalar("fake_conf", d_fake_conf)

    fimg = tf.placeholder(tf.float32)
    fake_img_summary = tf.summary.image('fake', fimg, max_outputs=10)

    d_output_x = disc_real
    d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_z = -disc_fake
    d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    g_loss_sum = tf.summary.scalar("g_loss", gen_cost)
    d_loss_sum = tf.summary.scalar("d_loss", disc_cost)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )

    differences = fake_cp - real_cp
    interpolates = real_cp + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, filter_num_d, output_shape, BATCH_SIZE,  reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_GEN, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=g_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_DIS, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=d_params)

    merge_no_img = tf.summary.merge([summary_real_conf,summary_fake_conf,summary_d_z_hist,summary_d_x_hist, g_loss_sum, d_loss_sum])

    return gen_train_op, disc_train_op, gen_cost, disc_cost, d_real_conf, d_fake_conf, real_cp, fake_cp, fimg, merge_no_img

# Train loop
def main():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    print "Current Config: ", current_config['config_name']
    print "Loading bsCoeff: %r, Sample Rate %r" % (bs_path, SAMPLE_RATE)

    with tf.device(device_cpu):
        clear_file(log_path, '.g01')
        clear_file(vis_path, '.png')
        print "Clear File"

        if MODE == 0:
            bsCoeff = load_bsCoeff(bs_path)
        elif MODE == 1:
            bsCoeff, ocp = load_bsCoeff_cp(bs_path)
            bsCoeff = sample_skull_points(bsCoeff, SAMPLE_RATE)

    with tf.device(device_gpu):
        weights = init_weights(filter_num_d, output_shape, Z_SIZE)
        biases = init_biases(filter_num_d, output_shape)
        #displacement field
        cp_batch = load_data(record_path, n_epoch, BATCH_SIZE, tuple(SHAPE_SIZE), MODE)
        gen_train_op, disc_train_op, g_loss, d_loss, real_conf, fake_conf, real_cp, fake_cp, fimg, merge_no_img = build_graph(cp_batch)
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

        if resume:
            print("Load existing model " + "!"*10)
            saver = tf.train.Saver()
            saver.restore(sess, model_file_name)

        saver = tf.train.Saver(max_to_keep=None)

        ##Vis real img
        #"Test Real img"
        rcp = sess.run(real_cp)

        with tf.device(device_cpu):
            rcp = np.reshape(rcp, (BATCH_SIZE,-1,3))

            rcp0 = rcp[0]
            count = 0
            for r in rcp:
                if np.all(r == rcp0):
                    count += 1

            if count > 1:
                print "data duplicated! count: ", count
            else:
                print "data seems good! count: ", count

            if MODE == 0:
                rvimg = vis_image(bsCoeff, rcp, 0, 5, vis_path, True)
            elif MODE == 1:
                rvimg = vis_image_displacement(bsCoeff, ocp, rcp, 0, 5, vis_path, True)

        merged = sess.run(real_img_summary, feed_dict={rimg:rvimg})
        summary_writer.add_summary(merged, 1)

        for iteration in xrange(ITERS):
            # Train generator
            if iteration > 0:
                sess.run(gen_train_op)
                # print "training g, fake_conf %r g_loss %r" % (sess.run(fake_conf), sess.run(g_loss))
            # Train critic
            if iteration < 500:
                disc_iters = 25
            else:
                disc_iters = CRITIC_ITERS
            for i in xrange(disc_iters):
                sess.run(disc_train_op)
                # print "d_critic: %r of total %r real_conf %r, d_loss %r" % (i+1, 
                #     disc_iters, sess.run(real_conf), sess.run(d_loss))
            if iteration % PRINT == PRINT - 1:
                print "step: %r of total step %r" % (iteration+1, ITERS)

                fc, rc = sess.run([fake_conf, real_conf])
                gl, dl = sess.run([g_loss, d_loss])
                if fc > 0.99 and rc > 0.99:
                    print "g_loss %r, d_loss %r" % (gl, dl)
                else:
                    print "fake_conf %r g_loss %r, real_conf %r d_loss %r" % (fc, gl, rc, dl)
            
            if (iteration % MERGE == MERGE-1) and (iteration % VIS_SAVE != VIS_SAVE-1):
                merged_no_img = sess.run(merge_no_img)
                summary_writer.add_summary(merged_no_img, iteration+1)

            if iteration % VIS_SAVE == VIS_SAVE-1:
                with tf.device(device_cpu):
                    fcp = sess.run(fake_cp)
                    fcp = np.reshape(fcp, (BATCH_SIZE,-1,3))
                    if MODE == 0:
                        fvimg = vis_image(bsCoeff, fcp, iteration+1, 10, vis_path)
                    elif MODE == 1:
                        fvimg = vis_image_displacement(bsCoeff, ocp, fcp, iteration+1, 10, vis_path)

                merged = sess.run(merged_all, feed_dict={fimg:fvimg})
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
