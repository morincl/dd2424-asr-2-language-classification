import tensorflow as tf
import numpy as np
import math

speechT_output_folder_path = "/media/david/Elements/David/deep/speechT-output/"


def get_batch_no_nan(languages=None):
    """
    Generator that yields data batches that does not have any NaN:s in them, alternating between the the languages
    :param languages: list of strings for languages to be loaded
    :yield: one batch
    """

    if languages is None:
        languages = ["English", "French"]
    i = 0
    while True:
        for language in languages:
            try:
                batch = np.transpose(np.load(speechT_output_folder_path + "{}/batch{}.npy".format(language, i)),
                                     [1, 0, 2])
                if not math.isnan((np.max(batch))):
                    yield batch
            except FileNotFoundError:
                return
        i += 1


# Load data
data_generator = get_batch_no_nan()

dataset = next(data_generator)
for j in range(10):
    dataset = np.concatenate([dataset, next(data_generator)])

validation_set = next(data_generator)[0:125, :, :]
for j in range(1):
    validation_set = np.concatenate([validation_set, next(data_generator)[0:125, :, :]])

# normalize
for i in range(dataset.shape[2]):
    mean = np.mean(dataset[:, :, i])
    std = np.std(dataset[:, :, i])
    dataset[:, :, i] = (dataset[:, :, i] - mean) / std

for i in range(validation_set.shape[2]):
    mean = np.mean(validation_set[:, :, i])
    std = np.std(validation_set[:, :, i])
    validation_set[:, :, i] = (validation_set[:, :, i] - mean) / std

print(dataset.shape)

dataset_mean = np.mean(dataset)
dataset_std = np.std(dataset)

print(dataset_mean)
print(dataset_std)

# dataset -= dataset_mean
# dataset /= dataset_std

print(np.mean(validation_set))
print(np.std(validation_set))


# validation_set -= dataset_mean
# validation_set /= dataset_std

def split_indicies(index):
    return index // dataset.shape[1], index % dataset.shape[1]


current_index = 0
indicies = np.random.permutation(dataset.shape[0] * dataset.shape[1])


def get_next_batch(batch_size):
    global current_index, indicies
    if current_index + batch_size > len(indicies):
        current_index = 0
        indicies = np.random.permutation(dataset.shape[0] * dataset.shape[1])

    batch = np.zeros([batch_size, dataset.shape[2]])
    for i in range(batch_size):
        sample_index, frame_index = split_indicies(indicies[current_index + i])
        batch[i, :] = dataset[sample_index, frame_index, :]
    current_index += batch_size
    return batch


def format_validation_set(validation_set):
    new_val_set = validation_set[0, :, :]
    for i in range(1, validation_set.shape[0]):
        new_val_set = np.concatenate([new_val_set, validation_set[i, :, :]])
    return new_val_set


validation_set = format_validation_set(validation_set)
print(validation_set.shape)

tf.set_random_seed(1000)

# Training Parameters
learning_rate = 0.0001
learning_rate_place_holder = tf.placeholder(tf.float32, shape=[], name="learning_rate_place_holder")
num_steps = 1000000
batch_size = 250

display_step = 1000

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

# Prediction
y_pred = decoder
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), name="loss_op")
optimizer = tf.train.AdamOptimizer(learning_rate, name="adam_op").minimize(loss, name="optimizer_op")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

best_known_loss = 1000

weights = None
# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    current_index = 0
    indicies = np.random.permutation(dataset.shape[0] * dataset.shape[1])

    validation_loss = sess.run([loss], feed_dict={X: validation_set})
    print("Initial validation loss: " + str(np.sqrt(validation_loss)))
    best_known_loss = np.sqrt(validation_loss)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = get_next_batch(batch_size)

        learning_rate = learning_rate * 0.99999

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, learning_rate_place_holder: learning_rate})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, np.sqrt(l)))
            validation_loss = sess.run([loss], feed_dict={X: validation_set})
            print("val loss: " + str(np.sqrt(validation_loss)))
            print("Best known loss: " + str(best_known_loss))
            if np.sqrt(validation_loss) < best_known_loss:
                best_known_loss = np.sqrt(validation_loss)
                model_name = "auto-enc-" + str(i) + "-" + ("{0:.4f}".format(np.sqrt(validation_loss))).split('.')[1]
                save_path = saver.save(sess,
                                       "/media/karl/Elements/DeepLearningProject/auto-encoder/" + model_name + "/auto-enc.ckpt")
                print("Model saved in path: %s" % save_path)

    validation_loss = sess.run([loss], feed_dict={X: validation_set})
    print("Final validation loss: " + str(np.sqrt(validation_loss)))
