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
METADATA_SCALER = [1.0 / 1.0, 1.0 / 65535.0, 1.0 / 65535.0]  # Protocal, SrcPrt, DesPrt
TOTAL_EPOCHS = 20000
LEARNING_RATE = 0.001  # Adam's default learning rate
BATCH_SIZE = 10

# Network parameters
N_INPUT = WINDOW_SIZE + META_FIELDS
N_HIDDEN_1 = 1000
N_HIDDEN_2 = 100
N_HIDDEN_3 = 10


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


def preprocess_input(x, name):
    # Normalize input into range [0, 1]
    scaler = np.array(
        METADATA_SCALER + ([1.0 / MAX_PACKETS_SIZE] * WINDOW_SIZE))
    with tf.name_scope("preprocess"):
        scaler = tf.constant(
            scaler, dtype=tf.float32, shape=[META_FIELDS + WINDOW_SIZE],
            name="scaler")
    x = tf.multiply(x, scaler, name="scaled_" + name)

    return x


def create_network(noised_x, cleaned_x):
    """
    Modified from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
    """

    with tf.name_scope("network"):
        with tf.name_scope("weights"):
            weights = {
                "encoder_h1": tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1]), name="encoder_h1"),
                "encoder_h2": tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2]), name="encoder_h2"),
                "encoder_h3": tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_3]), name="encoder_h3"),
            }

        with tf.name_scope("biases"):
            biases = {
                "encoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="encoder_b1"),
                "encoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="encoder_b2"),
                "encoder_b3": tf.Variable(tf.random_normal([N_HIDDEN_3]), name="encoder_b3"),
                "decoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="decoder_b1"),
                "decoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="decoder_b2"),
                "decoder_b3": tf.Variable(tf.random_normal([N_INPUT]), name="decoder_b3"),
            }

        with tf.name_scope("encoder"):
            encoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(noised_x, weights["encoder_h1"]), biases["encoder_b1"]), name="encoder_l1")
            encoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l1, weights["encoder_h2"]), biases["encoder_b2"]), name="encoder_l2")
            encoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l2, weights["encoder_h3"]), biases["encoder_b3"]), name="encoder_l3")
        with tf.name_scope("decoder"):
            decoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l3, tf.transpose(weights["encoder_h3"])), biases["decoder_b1"]), name="decoder_l1")
            decoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l1, tf.transpose(weights["encoder_h2"])), biases["decoder_b2"]), name="decoder_l2")
            decoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l2, tf.transpose(weights["encoder_h1"])), biases["decoder_b3"]), name="decoder_l3")

        cost = tf.reduce_mean(tf.square(cleaned_x - decoder_l3), name="cost")
        tf.summary.scalar("cost_summary", cost)

    return {
        "encoded": encoder_l3,
        "decoded": decoder_l3,
        "cost": cost,
    }


if __name__ == "__main__":
    with tf.name_scope("input"):
        noised_examples, cleaned_examples = readers()
        noised_examples = preprocess_input(noised_examples, name="noised_examples")
        cleaned_examples = preprocess_input(cleaned_examples, name="cleaned_examples")

        # Batching
        # min_after_dequeue = 12
        # BATCH_SIZE = 4
        # capacity = min_after_dequeue + 14 * batch_size
        # noised_examples_batch, cleaned_examples_batch = tf.train.shuffle_batch(
        #     [noised_examples, cleaned_examples],
        #     batch_size=batch_size, capacity=capacity,
        #     min_after_dequeue=min_after_dequeue)
        noised_examples_batch, cleaned_examples_batch = tf.train.batch(
            (noised_examples, cleaned_examples), batch_size=BATCH_SIZE)

    autoencoder = create_network(noised_examples_batch, cleaned_examples_batch)

    with tf.name_scope("training"):
        # train_step = tf.train.GradientDescentOptimizer(
        #     LEARNING_RATE).minimize(autoencoder['cost'])
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(autoencoder["cost"])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())

        for i in range(TOTAL_EPOCHS):
            _, summary = sess.run([train_step, tf.summary.merge_all()])

            writer.add_summary(summary, i)

            if i % 100 == 0:
                print("{} / {}".format(i, TOTAL_EPOCHS))
                # print(i, " cost", sess.run(autoencoder['cost']))
                # print(i, " original", sess.run(noised_examples_batch))
                # print(i, " decoded", sess.run(autoencoder['decoded']))
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
