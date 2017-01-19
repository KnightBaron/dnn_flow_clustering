import tensorflow as tf
import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

INPUT_FILE = "/home/knightbaron/data/data_500hz_noised_noip/data_500hz_noised_noip.csv"
LOG_PATH = "/tmp/tensorflow/"
TOTAL_PARTS = 77
WINDOW_SIZE = 5000
META_FIELDS = 3  # Src IP, Dst IP, Port
TOTAL_COLUMNS = (WINDOW_SIZE * 2) + META_FIELDS  # meta, noised, cleaned
MAX_PACKETS_SIZE = 805744.0  # Bytes
METADATA_NORMALIZER = [1.0 / 1.0, 1.0 / 65535.0, 1.0 / 65535.0]  # Protocal, SrcPrt, DesPrt
TOTAL_EPOCHS = 200
LEARNING_RATE = 0.2


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def readers():
    # CSV Reader modified from:
    # https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#csv-files

    # Load a single CSV or multi-part CSVs
    if TOTAL_PARTS < 1:
        filename_queue = tf.train.string_input_producer([INPUT_FILE])
    else:
        filename_queue = list()
        for i in range(TOTAL_PARTS):
            filename_queue.append("{}.{:02d}".format(INPUT_FILE, i))
        filename_queue = tf.train.string_input_producer(filename_queue)

    reader = tf.TextLineReader()
    (key, value) = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[]] * TOTAL_COLUMNS
    # record_defaults = [tf.constant([], dtype=tf.float64) for _ in range(TOTAL_COLUMNS)]
    columns = tf.decode_csv(value, record_defaults=record_defaults)
    # features = tf.pack(columns)

    noised_features = tf.pack(
        columns[:(WINDOW_SIZE + META_FIELDS)],
        name="noised_examples")
    cleaned_features = tf.pack(
        columns[:META_FIELDS] + columns[(WINDOW_SIZE + META_FIELDS):],
        name="cleaned_examples")

    return (noised_features, cleaned_features)


def preprocess_input_noexpansion(x, name):
    # Normalize input into range [0, 1]
    normalizer = np.array(
        METADATA_NORMALIZER + ([1.0 / MAX_PACKETS_SIZE] * WINDOW_SIZE))
    with tf.name_scope("preprocess"):
        normalizer = tf.constant(
            normalizer, dtype=tf.float32, shape=[META_FIELDS + WINDOW_SIZE],
            name="normalizer")
    x = tf.multiply(x, normalizer, name="normalized_" + name)

    return x


def new_create(noised_x, cleaned_x):
    """
    Modified from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
    """
    N_INPUT = 5003
    N_HIDDEN_1 = 1000
    N_HIDDEN_2 = 100
    N_HIDDEN_3 = 10

    with tf.name_scope("network"):
        weights = {
            "encoder_h1": tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1]), name="encoder_h1"),
            "encoder_h2": tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2]), name="encoder_h2"),
            "encoder_h3": tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_3]), name="encoder_h3"),
            # "decoder_h1": tf.Variable(tf.random_normal([N_HIDDEN_3, N_HIDDEN_2]), name="decoder_h1"),
            # "decoder_h2": tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_1]), name="decoder_h2"),
            # "decoder_h3": tf.Variable(tf.random_normal([N_HIDDEN_1, N_INPUT]), name="decoder_h3"),
        }

        biases = {
            "encoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="encoder_b1"),
            "encoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="encoder_b2"),
            "encoder_b3": tf.Variable(tf.random_normal([N_HIDDEN_3]), name="encoder_b3"),
            "decoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="decoder_b1"),
            "decoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="decoder_b2"),
            "decoder_b3": tf.Variable(tf.random_normal([N_INPUT]), name="decoder_b3"),
        }

        encoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(noised_x, weights["encoder_h1"]), biases["encoder_b1"]), name="encoder_l1")
        encoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l1, weights["encoder_h2"]), biases["encoder_b2"]), name="encoder_l2")
        encoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l2, weights["encoder_h3"]), biases["encoder_b3"]), name="encoder_l3")
        decoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l3, tf.transpose(weights["encoder_h3"])), biases["decoder_b1"]), name="decoder_l1")
        decoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l1, tf.transpose(weights["encoder_h2"])), biases["decoder_b2"]), name="decoder_l2")
        decoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l2, tf.transpose(weights["encoder_h1"])), biases["decoder_b3"]), name="decoder_l3")

        cost = tf.reduce_mean(tf.square(cleaned_x - decoder_l3))

    # decoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l3, weights["decoder_h1"]), biases["decoder_b1"]))
    # decoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l1, weights["decoder_h2"]), biases["decoder_b2"]))
    # decoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l2, weights["decoder_h3"]), biases["decoder_b3"]))

    # cost = tf.reduce_mean(tf.pow(cleaned_x - decoder_l3, 2))
    # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cost)

    return {
        "encoded": encoder_l3,
        "decoded": decoder_l3,
        "cost": cost,
    }


def create(x, x_clean, layer_sizes):
    """
    Modified from https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5
    """
    # Build the encoding layers
    next_layer_input = x

    encoding_matrices = list()
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(
            tf.random_uniform(
                [input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)),
            name="Weight")
        # with tf.name_scope("weights"):
        #     variable_summaries(W)

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]), name="Bias")
        # with tf.name_scope("biases"):
        #     variable_summaries(b)

        # We are going to use tied-weights so store the W matrix for later
        # reference.
        encoding_matrices.append(W)

        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        # we are using tied weights, so just lookup the encoding matrix for
        # this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x_clean - reconstructed_x)))
    }


if __name__ == "__main__":
    with tf.name_scope("input"):
        noised_features, cleaned_features = readers()

        noised_features = preprocess_input_noexpansion(noised_features, name="noised_examples")
        cleaned_features = preprocess_input_noexpansion(cleaned_features, name="cleaned_examples")

    # Batching
    min_after_dequeue = 12
    batch_size = 4
    capacity = min_after_dequeue + 14 * batch_size
    noised_features_batch, cleaned_features_batch = tf.train.shuffle_batch(
        [noised_features, cleaned_features],
        batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    # autoencoder = create(noised_features, cleaned_features, [1000, 100, 10])
    # autoencoder = new_create(noised_features, cleaned_features)
    autoencoder = new_create(noised_features_batch, cleaned_features_batch)
    # start_dim = 5003
    # x = tf.placeholder("float", shape=(None, start_dim), name="x")
    # x_clean = tf.placeholder("float", shape=(None, start_dim), name="cleaned_x")
    # autoencoder = create(x, x_clean, [1000, 500, 100])

    # train_step = tf.train.GradientDescentOptimizer(
    #     LEARNING_RATE).minimize(autoencoder['cost'])

    with tf.name_scope("training"):
        train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(autoencoder["cost"])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())

        for i in range(TOTAL_EPOCHS):
            sess.run(train_step)
            # sess.run(train_step, feed_dict={x: noised_features, x_clean: cleaned_features})

            if i % 100 == 0:
                print(i, " cost", sess.run(autoencoder['cost']))
                print(i, " original", sess.run(noised_features_batch))
                print(i, " decoded", sess.run(autoencoder['decoded']))
            #     print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch, x_clean: batch}))
            #     print(i, " original", batch[0])
            #     print(i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch, x_clean: batch}))

            # Retrieve a single instance:
            # example, label = sess.run([features, columns[0]])
            # example = sess.run(noised_features)
            # import pdb; pdb.set_trace()
            # example = sess.run(cleaned_features)

        coord.request_stop()
        coord.join(threads)
