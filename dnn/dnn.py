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

frame_cutoff = 100 # How many frames to use per sample
random_seed = None  # Set this to any number to make random the same all the time (good for parameter optimization)
# tf.set_random_seed(1)  # Uncomment to make tf use same random seed

# Parameters
learning_rate = 0.0001
num_steps = 2000
minibatch_size = 128
display_step = 100
num_validation_set_minibatches = 10

# Network Parameters
num_hidden_1 = 256  # 1st layer number of neurons
num_hidden_2 = 256  # 2nd layer number of neurons
num_input = frame_cutoff * 50  # frame_cutoff frames * 50 elments per frame
num_classes = 2  # English or French

training_data_folder_path = "/media/karl/Elements/DeepLearningProject/VoxForge/dataset/labelled/training/"
validation_data_folder_path = "/media/karl/Elements/DeepLearningProject/VoxForge/dataset/labelled/validation/"


def load_data(input_path):
    files = os.listdir(input_path)
    data = None
    for file in files:
        data_batch = np.load(input_path + file)
        if data is None:
            data = data_batch
        else:
            data = np.concatenate([data, data_batch])
    return data


def next_mini_batch(data_list, epoch_callback=None):
    while True:
        for i in range(len(data_list)//minibatch_size):
            minibatch = data_list[i * minibatch_size:(i + 1) * minibatch_size]
            batch_x = None
            batch_y = None
            for sample in minibatch:
                sample_x = sample[0]
                offset = random.randint(0, 149)
                flat_and_cut_sample_x = np.reshape(sample_x[offset:frame_cutoff + offset, :],
                                                   [1, frame_cutoff * sample_x.shape[1]])
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
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam_op")
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

    for step in range(1, num_steps+1):
        batch_x, batch_y = next(training_generator)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})

            val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X: validation_set_x,
                                                                         Y: validation_set_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", Validation Loss= " + \
                  "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                  "{:.3f}".format(val_acc))
    print("Optimization Finished!")
