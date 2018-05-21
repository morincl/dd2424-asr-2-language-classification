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

minibatch_size = 100
frame_cutoff = 100

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 5000  # MNIST data input (img shape: 28*28)
num_classes = 2  # MNIST total classes (0-9 digits)

training_data_folder_path = "/media/david/Elements/David/deep/final_data/labelled/test/"
validation_data_folder_path = "/media/david/Elements/David/deep/final_data/labelled/validation/"


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


def next_mini_batch(data_list):
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
        print("looped")


training_data = load_data(training_data_folder_path)
validation_data = load_data(validation_data_folder_path)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
training_generator = next_mini_batch(training_data)
validation_generator = next_mini_batch(validation_data)  # TODO: HIGH PRIO: this does not good for the validation data, fix
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next(training_data)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            #batch_x_val, batch_y_val = next(validation_data)
            #loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 #Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

"""
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
Y: mnist.test.labels}))
"""