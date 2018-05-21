import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
import math

speechT_output_folder_path = "/media/karl/Elements/DeepLearningProject/VoxForge/speechT-output/Spanish/"
output_folder = "/media/karl/Elements/DeepLearningProject/VoxForge/auto-enc-output/Spanish/"

# def get_batch_no_nan(languages=None):
#     """
#     Generator that yields data batches that does not have any NaN:s in them, alternating between the the languages
#     :param languages: list of strings for languages to be loaded
#     :yield: one batch
#     """
#
#     if languages is None:
#         languages = ["English"]
#     i = 0
#     while True:
#         for language in languages:
#             try:
#                 batch = np.transpose(np.load(speechT_output_folder_path + "{}/batch{}.npy".format(language, i)),
#                                      [1, 0, 2])
#                 if not math.isnan((np.max(batch))):
#                     yield batch
#             except FileNotFoundError:
#                 return
#         i += 1
#
#
# # Load data
# data_generator = get_batch_no_nan()



mean_and_std = np.load('mean_and_std_no_mean.npy')

#print(mean_and_std)

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



with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('/home/karl/Documents/deep/project/dd2424-asr-2-language-classification/auto-encoder/trained-model/'))
    print("Model restored.")

    files = os.listdir(speechT_output_folder_path)

    print("Found " + str(len(files)) + " files.")
    file_index = 0
    for input_file in files:
        print("Converting file " + str(file_index+1))
        dataset = np.transpose(np.load(speechT_output_folder_path + input_file), [1, 0, 2])
        if math.isnan((np.max(dataset))):
            print("File " + str(file_index+1) + " contains nan and is therefore skipped.")
            continue
        output = np.zeros([dataset.shape[0], dataset.shape[1], 50])
        for i in range(dataset.shape[2]):
            dataset[:,:,i] = (dataset[:,:,i] - mean_and_std[i,0]) / mean_and_std[i,1]

        for sample_ind in range(dataset.shape[0]):
            sample = dataset[sample_ind, :, :]
            output[sample_ind, :, :] = sess.run(encoder, feed_dict={X: sample})

        np.save(output_folder + "batch" + str(file_index) + ".npy", output)
        file_index += 1
