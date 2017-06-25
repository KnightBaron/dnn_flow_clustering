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

INPUT_FILE = "/work/pongsakorn-u/data_1hz_noip/data_1hz_noip.csv"
TOTAL_PARTS = 3108
WINDOW_SIZE = 10
META_FIELDS = 3  # Src IP, Dst IP, Port
TOTAL_COLUMNS = (WINDOW_SIZE * 2) + META_FIELDS  # meta, data, counter
MAX_PACKETS_SIZE = 9795156.0  # Bytes
MAX_PACKETS_COUNT = 7375.0  # Packets
METADATA_SCALER = [1.0 / 1.0, 1.0 / 65535.0, 1.0 / 65535.0]  # Protocal, SrcPrt, DesPrt
TOTAL_EPOCHS_LAYER1 = 80000
TOTAL_EPOCHS_LAYER2 = 80000
TOTAL_EPOCHS_LAYER3 = 80001
LEARNING_RATE = 0.001  # Adam's default learning rate
BATCH_SIZE = 100
MIN_AFTER_DEQUEUE = 2000
CAPACITY = MIN_AFTER_DEQUEUE + 3 * (BATCH_SIZE)
SHIFTER = 0.2  # Shift from 0.5 to 0.7 then round to get 70% chance of 1
CHECKPOINT_FREQUENCY = 10000  # Every n epochs

# Network parameters
N_INPUT = (WINDOW_SIZE * 2) + META_FIELDS
N_HIDDEN_1 = 100
N_HIDDEN_2 = 50
N_HIDDEN_3 = 5

MODEL_FILE = "/work/pongsakorn-u/tensorflow_model/softmax/{}outputs/model".format(N_HIDDEN_3)
LOG_PATH = "/work/pongsakorn-u/tensorflow_log/softmax/{}outputs".format(N_HIDDEN_3)


def readers():
    # CSV Reader modified from:
    # https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#csv-files

    # Load a single CSV or multi-part CSVs
    if TOTAL_PARTS < 1:
        filename_queue = tf.train.string_input_producer([INPUT_FILE])
    else:
        filename_queue = list()
        for i in range(TOTAL_PARTS):
            filename_queue.append("{}.{:04d}".format(INPUT_FILE, i))
        filename_queue = tf.train.string_input_producer(filename_queue, shuffle=True)

    reader = tf.TextLineReader()
    (key, value) = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[]] * TOTAL_COLUMNS
    columns = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.stack(columns, name="examples")


def preprocess_input_batch(x_batch, name):
    # Normalize input into range [0, 1]
    scaler = METADATA_SCALER + ([1.0 / MAX_PACKETS_SIZE] * WINDOW_SIZE) + ([1.0 / MAX_PACKETS_COUNT] * WINDOW_SIZE)
    scaler = [scaler for _ in range(BATCH_SIZE)]
    scaler = np.array(scaler)
    with tf.name_scope("preprocess"):
        scaler = tf.constant(
            scaler, dtype=tf.float32, shape=[BATCH_SIZE, META_FIELDS + (WINDOW_SIZE * 2)],
            name="scaler")
    x_batch = tf.multiply(x_batch, scaler, name="scaled_" + name)

    return x_batch


def add_noise(x, size, name="layer0", is_first_layer=False):
    if is_first_layer:
        noise = tf.round(tf.add(
            tf.random_uniform([BATCH_SIZE, size]),
            tf.constant(np.array([[SHIFTER] * size] * BATCH_SIZE), dtype=tf.float32)
        ))
        noise = tf.concat_v2(
            [tf.constant(np.array([[1.0] * META_FIELDS] * BATCH_SIZE), dtype=tf.float32),  # Always keep metadata
                noise, noise],
            1, name="{}_noise".format(name))
    else:
        noise = tf.round(tf.add(
            tf.random_uniform([BATCH_SIZE, size]),
            tf.constant(np.array([[SHIFTER] * size] * BATCH_SIZE), dtype=tf.float32)
        ), name="{}_noise".format(name))

    return tf.multiply(x, noise, name="{}_noised".format(name))


def create_network(noised_x, cleaned_x, n_input, n_hidden, name="layer0", is_final_layer=False):
    """
    Modified from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
    """

    with tf.name_scope("{}_network".format(name)):
        with tf.name_scope("{}_weights".format(name)):
            weights = {
                "encoder": tf.Variable(tf.random_normal([n_input, n_hidden]), name="{}_encoder_weight".format(name)),
                "decoder": tf.Variable(tf.random_normal([n_hidden, n_input]), name="{}_decoder_weight".format(name)),
            }
            tf.add_to_collection("{}_encoder_weight".format(name), weights["encoder"])
            tf.add_to_collection("{}_decoder_weight".format(name), weights["decoder"])

        with tf.name_scope("{}_biases".format(name)):
            biases = {
                "encoder": tf.Variable(tf.random_normal([n_hidden]), name="{}_encoder_bias".format(name)),
                "decoder": tf.Variable(tf.random_normal([n_input]), name="{}_decoder_bias".format(name)),
            }
            tf.add_to_collection("{}_encoder_bias".format(name), biases["encoder"])
            tf.add_to_collection("{}_decoder_bias".format(name), biases["decoder"])

        with tf.name_scope("{}_encoder_scope".format(name)):
            if is_final_layer:
                encoder = tf.nn.softmax(tf.add(tf.matmul(noised_x, weights["encoder"]), biases["encoder"]), name="{}_encoder".format(name))
            else:
                encoder = tf.nn.sigmoid(tf.add(tf.matmul(noised_x, weights["encoder"]), biases["encoder"]), name="{}_encoder".format(name))
        with tf.name_scope("{}_decoder_scope".format(name)):
            decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weights["decoder"]), biases["decoder"]), name="{}_decoder".format(name))

        cost = tf.reduce_mean(tf.square(tf.subtract(cleaned_x, decoder)), name="{}_cost_op".format(name))
        tf.summary.scalar("{}_cost".format(name), cost)

    return {
        "weights": weights,
        "biases": biases,
        "encoded": encoder,
        "decoded": decoder,
        "cost": cost,
    }


if __name__ == "__main__":
    with tf.name_scope("input_pipeline"):
        examples = readers()

        # Batching
        examples_batch = tf.train.shuffle_batch(
            [examples], batch_size=BATCH_SIZE, capacity=CAPACITY,
            min_after_dequeue=MIN_AFTER_DEQUEUE)

        preprocessed_examples_batch = preprocess_input_batch(examples_batch, name="examples_batch")

    with tf.name_scope("layer1"):
        layer_1_noised_input = add_noise(preprocessed_examples_batch, WINDOW_SIZE, name="layer1", is_first_layer=True)
        layer_1 = create_network(
            layer_1_noised_input, preprocessed_examples_batch, N_INPUT, N_HIDDEN_1, name="layer1")

    with tf.name_scope("layer2"):
        layer_2_noised_input = add_noise(layer_1["encoded"], N_HIDDEN_1, name="layer2")
        layer_2 = create_network(
            layer_2_noised_input, layer_1["encoded"], N_HIDDEN_1, N_HIDDEN_2, name="layer2")

    with tf.name_scope("layer3"):
        layer_3_noised_input = add_noise(layer_2["encoded"], N_HIDDEN_2, name="layer3")
        layer_3 = create_network(
            layer_3_noised_input, layer_2["encoded"], N_HIDDEN_2, N_HIDDEN_3, name="layer3", is_final_layer=True)

    saver = tf.train.Saver()
    train_layer_1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        layer_1["cost"], var_list=[
            layer_1["weights"]["encoder"], layer_1["weights"]["decoder"],
            layer_1["biases"]["encoder"], layer_1["biases"]["decoder"],
        ])
    train_layer_2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        layer_2["cost"], var_list=[
            layer_2["weights"]["encoder"], layer_2["weights"]["decoder"],
            layer_2["biases"]["encoder"], layer_2["biases"]["decoder"],
        ])
    train_layer_3 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        layer_3["cost"], var_list=[
            layer_3["weights"]["encoder"], layer_3["weights"]["decoder"],
            layer_3["biases"]["encoder"], layer_3["biases"]["decoder"],
        ])

    tf.add_to_collection("train_layer_1", train_layer_1)
    tf.add_to_collection("train_layer_2", train_layer_2)
    tf.add_to_collection("train_layer_3", train_layer_3)
    tf.add_to_collection("examples_batch", examples_batch)
    tf.add_to_collection("encoded", layer_3["encoded"])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        merged = tf.summary.merge_all()

        logging.info("Training layer 1...")
        for i in range(TOTAL_EPOCHS_LAYER1):
            if i % CHECKPOINT_FREQUENCY == 0:
                _, summary = sess.run(
                    [train_layer_1, merged],
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, "layer1:step:{}".format(i))
                logging.info("EPOCH: {} / {}".format(i + 1, TOTAL_EPOCHS_LAYER1))
            else:
                _, summary = sess.run([train_layer_1, merged])
            writer.add_summary(summary, i)

        logging.info("Training layer 2...")
        for i in range(TOTAL_EPOCHS_LAYER2):
            if i % CHECKPOINT_FREQUENCY == 0:
                _, summary = sess.run(
                    [train_layer_2, merged],
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, "layer2:step:{}".format(i))
                logging.info("EPOCH: {} / {}".format(i + 1, TOTAL_EPOCHS_LAYER2))
            else:
                _, summary = sess.run([train_layer_2, merged])
            writer.add_summary(summary, i + TOTAL_EPOCHS_LAYER1)

        logging.info("Training layer 3...")
        for i in range(TOTAL_EPOCHS_LAYER3):
            if i % CHECKPOINT_FREQUENCY == 0:
                _, summary = sess.run(
                    [train_layer_3, merged],
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, "layer3:step:{}".format(i))
                logging.info("EPOCH: {} / {}".format(i + 1, TOTAL_EPOCHS_LAYER3))
                logging.info("Saving model...")
                saver.save(sess, MODEL_FILE, global_step=(TOTAL_EPOCHS_LAYER1 + TOTAL_EPOCHS_LAYER2 + i - 1))
            else:
                _, summary = sess.run([train_layer_3, merged])
            writer.add_summary(summary, i + TOTAL_EPOCHS_LAYER1 + TOTAL_EPOCHS_LAYER2)

        logging.info("Saving final model...")
        saver.save(sess, MODEL_FILE, global_step=(TOTAL_EPOCHS_LAYER1 + TOTAL_EPOCHS_LAYER2 + TOTAL_EPOCHS_LAYER3 - 1))

        coord.request_stop()
        coord.join(threads)
    logging.info("DONE")
