import tensorflow as tf
import numpy as np
import logging
import random

# import pdb; pdb.set_trace()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

INPUT_FILE = "/work/pongsakorn-u/data_500hz_noised_noip/data_500hz_noised_noip.csv"
TOTAL_PARTS = 77
WINDOW_SIZE = 5000
META_FIELDS = 3  # Src IP, Dst IP, Port
TOTAL_COLUMNS = (WINDOW_SIZE * 2) + META_FIELDS  # meta, noised, cleaned
MAX_PACKETS_SIZE = 805744.0  # Bytes
METADATA_SCALER = [1.0 / 1.0, 1.0 / 65535.0, 1.0 / 65535.0]  # Protocal, SrcPrt, DesPrt
TOTAL_EPOCHS = 13000
LEARNING_RATE = 0.001  # Adam's default learning rate
BATCH_SIZE = 10
LOAD_META_GRAPH = False

# Network parameters
N_INPUT = WINDOW_SIZE + META_FIELDS
N_HIDDEN_1 = 5500
N_HIDDEN_2 = 500
N_HIDDEN_3 = 8

META_GRAPH_FILE = "/work/pongsakorn-u/tensorflow_model/{}outputs/my-model.meta".format(N_HIDDEN_3)
MODEL_FILE = "/work/pongsakorn-u/tensorflow_model/{}outputs/my-model".format(N_HIDDEN_3)
LOG_PATH = "/work/pongsakorn-u/tensorflow_log/{}outputs".format(N_HIDDEN_3)


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
    columns = tf.decode_csv(value, record_defaults=record_defaults)

    noised_features = tf.stack(
        columns[:(WINDOW_SIZE + META_FIELDS)],
        name="noised_examples")
    cleaned_features = tf.stack(
        columns[:META_FIELDS] + columns[(WINDOW_SIZE + META_FIELDS):],
        name="cleaned_examples")

    return (noised_features, cleaned_features)


def preprocess_input_batch(x_batch, name):
    # Normalize input into range [0, 1]
    scaler = METADATA_SCALER + ([1.0 / MAX_PACKETS_SIZE] * WINDOW_SIZE)
    scaler = [scaler for _ in range(BATCH_SIZE)]
    scaler = np.array(scaler)
    with tf.name_scope("preprocess"):
        scaler = tf.constant(
            scaler, dtype=tf.float32, shape=[BATCH_SIZE, META_FIELDS + WINDOW_SIZE],
            name="scaler")
    x_batch = tf.multiply(x_batch, scaler, name="scaled_" + name)

    return x_batch


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
                "decoder_h1": tf.Variable(tf.random_normal([N_HIDDEN_3, N_HIDDEN_2]), name="decoder_h1"),
                "decoder_h2": tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_1]), name="decoder_h2"),
                "decoder_h3": tf.Variable(tf.random_normal([N_HIDDEN_1, N_INPUT]), name="decoder_h3"),
            }
            # for variable in weights.values():
            #     tf.add_to_collection("variables", variable)

        with tf.name_scope("biases"):
            biases = {
                "encoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="encoder_b1"),
                "encoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="encoder_b2"),
                "encoder_b3": tf.Variable(tf.random_normal([N_HIDDEN_3]), name="encoder_b3"),
                "decoder_b1": tf.Variable(tf.random_normal([N_HIDDEN_2]), name="decoder_b1"),
                "decoder_b2": tf.Variable(tf.random_normal([N_HIDDEN_1]), name="decoder_b2"),
                "decoder_b3": tf.Variable(tf.random_normal([N_INPUT]), name="decoder_b3"),
            }
            # for variable in biases.values():
            #     tf.add_to_collection("variables", variable)

        with tf.name_scope("encoder"):
            encoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(noised_x, weights["encoder_h1"]), biases["encoder_b1"]), name="encoder_l1")
            encoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l1, weights["encoder_h2"]), biases["encoder_b2"]), name="encoder_l2")
            encoder_l3 = tf.nn.softmax(tf.add(tf.matmul(encoder_l2, weights["encoder_h3"]), biases["encoder_b3"]), name="encoder_l3")
        with tf.name_scope("decoder"):
            decoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l3, weights["decoder_h1"]), biases["decoder_b1"]), name="decoder_l1")
            decoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l1, weights["decoder_h2"]), biases["decoder_b2"]), name="decoder_l2")
            decoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l2, weights["decoder_h3"]), biases["decoder_b3"]), name="decoder_l3")
            # decoder_l1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_l3, tf.transpose(weights["encoder_h3"])), biases["decoder_b1"]), name="decoder_l1")
            # decoder_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l1, tf.transpose(weights["encoder_h2"])), biases["decoder_b2"]), name="decoder_l2")
            # decoder_l3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_l2, tf.transpose(weights["encoder_h1"])), biases["decoder_b3"]), name="decoder_l3")

        cost = tf.reduce_mean(tf.square(tf.subtract(cleaned_x, decoder_l3)), name="cost_op")
        tf.summary.scalar("cost", cost)

    return {
        "encoded": encoder_l3,
        "decoded": decoder_l3,
        "cost": cost,
    }


if __name__ == "__main__":
    with tf.name_scope("input_pipeline"):
        noised_examples, cleaned_examples = readers()

        # Batching
        noised_examples_batch, cleaned_examples_batch = tf.train.batch(
            [noised_examples, cleaned_examples], batch_size=BATCH_SIZE)

        preprocessed_noised_examples_batch = preprocess_input_batch(noised_examples_batch, name="noised_examples_batch")
        preprocessed_cleaned_examples_batch = preprocess_input_batch(cleaned_examples_batch, name="cleaned_examples_batch")

    autoencoder = create_network(
        preprocessed_noised_examples_batch, preprocessed_cleaned_examples_batch)

    if LOAD_META_GRAPH:
        saver = tf.train.import_meta_graph(META_GRAPH_FILE)
    else:
        saver = tf.train.Saver()
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(autoencoder["cost"])
        tf.add_to_collection("train_step", train_step)
        tf.add_to_collection("noised_examples_batch", noised_examples_batch)
        tf.add_to_collection("cleaned_examples_batch", cleaned_examples_batch)
        tf.add_to_collection("encoded", autoencoder["encoded"])

    with tf.Session() as sess:
        if LOAD_META_GRAPH:
            logging.info("Loading model...")
            saver.restore(sess, MODEL_FILE)
            train_step = tf.get_collection("train_step")[0]

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        merged = tf.summary.merge_all()

        for i in range(TOTAL_EPOCHS):
            if i % 1000 == 0:
                _, summary = sess.run(
                    [train_step, merged],
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
                # tl = timeline.Timeline(step_stats=run_metadata.step_stats)
                # with open(LOG_PATH + "/timeline_{}.json".format(i), "w") as tl_file:
                #     tl_file.write(tl.generate_chrome_trace_format())
                writer.add_run_metadata(run_metadata, "step:{}".format(i))
                logging.info("EPOCH: {} / {}".format(i + 1, TOTAL_EPOCHS))
            else:
                _, summary = sess.run([train_step, merged])
            writer.add_summary(summary, i)

        logging.info("Saving model...")
        if not LOAD_META_GRAPH:
            saver.save(sess, MODEL_FILE, global_step=(TOTAL_EPOCHS - 1))

        coord.request_stop()
        coord.join(threads)
    logging.info("DONE")
