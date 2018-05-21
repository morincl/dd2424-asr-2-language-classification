import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os

dataset = np.transpose(np.load('/media/karl/Elements/DeepLearningProject/VoxForge/speechT-output/English/batch0.npy'), [1, 0, 2])
mean_and_std = np.load('mean_and_std_no_mean.npy')

#print(mean_and_std)

for i in range(dataset.shape[2]):
    dataset[:,:,i] = (dataset[:,:,i] - mean_and_std[i,0]) / mean_and_std[i,1]

# Network Parameters
num_input = 250  # Input size
num_hidden_1 = 1000  # 1st layer num features
num_hidden_2 = 1000  # 2st layer num features
num_hidden_3 = 50  # latent dimension

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input], name="X")


def build_encoder(inputs):
    encoder1 = tf.layers.dense(inputs=inputs,
                               units=num_hidden_1,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    encoder2 = tf.layers.dense(inputs=encoder1,
                               units=num_hidden_2,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    encoder3 = tf.layers.dense(inputs=encoder2,
                               units=num_hidden_3,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False, name="encoder_op")

    return encoder3


def build_decoder(inputs):
    decoder1 = tf.layers.dense(inputs=inputs,
                               units=num_hidden_2,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    decoder2 = tf.layers.dense(inputs=decoder1,
                               units=num_hidden_1,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    decoder3 = tf.layers.dense(inputs=decoder2,
                               units=num_input,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False, name="decoder_op")
    return decoder3


encoder = build_encoder(X)
decoder = build_decoder(encoder)

saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('/home/karl/Documents/deep/project/dd2424-asr-2-language-classification/auto-encoder/trained-model/'))
    print("Model restored.")

    input_array = np.reshape(dataset[62, 104, :], (1, 250))
    decoded = sess.run(decoder, feed_dict={X: input_array})

    print("Input array:")
    print(input_array[0, 0:15])
    print("Decoded:")
    print(decoded[0, 0:15])
    print("Diff:")
    print(decoded[0, 0:15] - input_array[0, 0:15])
