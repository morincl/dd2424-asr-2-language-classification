""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf
import numpy as np
import os
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import math

power_cutoff = 200  # How many frames to use per sample
random_seed = None  # Set this to any number to make random the same all the time (good for parameter optimization)
# tf.set_random_seed(1)  # Uncomment to make tf use same random seed

# Parameters
learning_rate = 0.0001
learning_rate_place_holder = tf.placeholder(tf.float32, shape=[], name="learning_rate_place_holder")
num_steps = 20000
minibatch_size = 128
display_step = 100
num_validation_set_minibatches = 10

# Network Parameters
num_hidden_1 = 2560  # 1st layer number of neurons
num_hidden_2 = 2560  # 2nd layer number of neurons
num_input = power_cutoff * 128  # frame_cutoff frames * 50 elements per frame
num_classes = 2  # English or French

training_data_folder_path = "/media/karl/Elements/DeepLearningProject/VoxForge/power_labelled/training/"
validation_data_folder_path = "/media/karl/Elements/DeepLearningProject/VoxForge/power_labelled/validation/"


def load_data(input_path):
    files = os.listdir(input_path)
    data = None
    file_counter = 0
    for file in files:
        print("Loading file: " + file)
        data_batch = np.load(input_path + file)

        indicies_to_keep = []
        for sample_ind in range(data_batch.shape[0]):
            if data_batch[sample_ind, 0].shape[0] == 501 and not math.isnan(np.max(data_batch[sample_ind, 0])):
                indicies_to_keep.append(sample_ind)
        data_batch = data_batch[indicies_to_keep, :]

        if data is None:
            data = data_batch
        else:
            data = np.concatenate([data, data_batch])
        file_counter += 1

    # Rebalance dataset
    num_samples_per_langauge = np.sum(data[:,1])
    num_samples_to_remove = 0
    language_to_remove = -1
    if num_samples_per_langauge[0] > num_samples_per_langauge[1]:
        num_samples_to_remove = num_samples_per_langauge[0] - num_samples_per_langauge[1]
        language_to_remove = 0
    elif num_samples_per_langauge[1] > num_samples_per_langauge[0]:
        num_samples_to_remove = num_samples_per_langauge[1] - num_samples_per_langauge[0]
        language_to_remove = 1

    if num_samples_to_remove > 0:
        indicies_to_keep = []
        num_excluded_samples = 0
        for sample_ind in range(data.shape[0]):
            if data[sample_ind, 1][language_to_remove] == 0 or num_excluded_samples >= num_samples_to_remove:
                indicies_to_keep.append(sample_ind)
            else:
                num_excluded_samples += 1
        data = data[indicies_to_keep, :]
    return data


def next_mini_batch(data_list, epoch_callback=None):
    while True:
        for i in range(len(data_list)//minibatch_size):
            minibatch = data_list[i * minibatch_size:(i + 1) * minibatch_size]
            batch_x = None
            batch_y = None
            for sample in minibatch:
                sample_x = sample[0]
                offset = random.randint(0, 298)
                flat_and_cut_sample_x = np.reshape(sample_x[offset:power_cutoff + offset, :],[1, power_cutoff * sample_x.shape[1]])
                if batch_x is None:
                    batch_x = flat_and_cut_sample_x
                else:
                    batch_x = np.concatenate([batch_x, flat_and_cut_sample_x])

                sample_y = np.reshape(sample[1], [1, 2])
                if batch_y is None:
                    batch_y = sample_y
                else:
                    batch_y = np.concatenate([batch_y, sample_y])

            yield batch_x, batch_y
        if epoch_callback is not None:
            epoch_callback()


training_data = load_data(training_data_folder_path)
validation_data = load_data(validation_data_folder_path)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Create model
def neural_net(x):
    hidden_1 = tf.layers.dense(inputs=x,
                               units=num_hidden_1,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    hidden_2 = tf.layers.dense(inputs=hidden_1,
                               units=num_hidden_2,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False)

    logits = tf.layers.dense(inputs=hidden_2,
                               units=num_classes,
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=False, name="logits")
    return logits


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits, name="prediction")

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_place_holder, name="adam_op")
train_op = optimizer.minimize(loss_op, name="optimizer_op")

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Build validation set
# Set seed=1 to always use the same seed for validation
random.seed(1)

validation_set_x = None
validation_set_y = None
validation_generator = next_mini_batch(validation_data)
for i in range(num_validation_set_minibatches):
    batch_x, batch_y = next(validation_generator)
    if validation_set_x is not None:
        validation_set_x = np.concatenate([validation_set_x, batch_x])
        validation_set_y = np.concatenate([validation_set_y, batch_y])
    else:
        validation_set_x = batch_x
        validation_set_y = batch_y
# remove seed (or change to what was used before)
random.seed(random_seed)

# Start training
training_generator = next_mini_batch(training_data)
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
    val_loss, val_acc, val_score = sess.run([loss_op, accuracy, prediction], feed_dict={X: validation_set_x, Y: validation_set_y})
    fpr, tpr, thresholds = roc_curve(validation_set_y[:, 0], val_score[:, 0], pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    print("Initial State, Minibatch Loss= " + \
          "{:.4f}".format(loss) + ", Training Accuracy= " + \
          "{:.3f}".format(acc) + ", Validation Loss= " + \
          "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
          "{:.3f}".format(val_acc) + ", EER= " + \
          "{:.4f}".format(eer))

    for step in range(1, num_steps+1):
        batch_x, batch_y = next(training_generator)

        if step < 15000:
            learning_rate = learning_rate * 0.9997

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, learning_rate_place_holder: learning_rate})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})

            val_loss, val_acc, val_score = sess.run([loss_op, accuracy, prediction], feed_dict={X: validation_set_x,
                                                                         Y: validation_set_y})

            fpr, tpr, thresholds = roc_curve(validation_set_y[:, 0], val_score[:, 0], pos_label=1)

            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", Validation Loss= " + \
                  "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                  "{:.3f}".format(val_acc) + ", EER= " + \
                  "{:.4f}".format(eer))
    print("Optimization Finished!")
