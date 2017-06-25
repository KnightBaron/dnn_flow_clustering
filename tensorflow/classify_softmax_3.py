import tensorflow as tf
import numpy as np
import logging
import random
import gzip
import csv

# import pdb; pdb.set_trace()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

TOTAL_CLASS = 3

MODEL_FILE = "/work/pongsakorn-u/tensorflow_model/softmax/{}outputs/model-240000".format(TOTAL_CLASS)
META_GRAPH_FILE = MODEL_FILE + ".meta"

LOG_PATH = "/work/pongsakorn-u/tensorflow_classified/log/softmax/{}outputs".format(TOTAL_CLASS)
OUTPUT_FILE = "/work/pongsakorn-u/tensorflow_classified/softmax/{}outputs.csv.gz".format(TOTAL_CLASS)
CHECKPOINT_FREQUENCY = 1000
BATCH_SIZE = 161  # Should be divisible to total input count to use all data

# For scaler
META_FIELDS = 3  # Src IP, Dst IP, Port
WINDOW_SIZE = 10
MAX_PACKETS_SIZE = 9795156.0  # Bytes
MAX_PACKETS_COUNT = 7375.0  # Packets
METADATA_SCALER = [1.0 / 1.0, 1.0 / 65535.0, 1.0 / 65535.0]  # Protocal, SrcPrt, DesPrt

# For reader
INPUT_FILE = "/work/pongsakorn-u/data_1hz_noip/data_1hz_noip.csv"
TOTAL_PARTS = 3108
TOTAL_COLUMNS = (WINDOW_SIZE * 2) + META_FIELDS  # meta, data, counter


def readers():
    # CSV Reader modified from:
    # https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#csv-files

    # Load a single CSV or multi-part CSVs
    if TOTAL_PARTS < 1:
        filename_queue = tf.train.string_input_producer([INPUT_FILE], num_epochs=1)
    else:
        filename_queue = list()
        for i in range(TOTAL_PARTS):
            filename_queue.append("{}.{:04d}".format(INPUT_FILE, i))
        filename_queue = tf.train.string_input_producer(filename_queue, num_epochs=1)

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


if __name__ == "__main__":
    logging.info("Loading meta graph...")
    saver = tf.train.import_meta_graph(META_GRAPH_FILE)

    examples = readers()
    example_batch_op = tf.train.batch([examples], batch_size=BATCH_SIZE)

    with tf.Session() as session:
        logging.info("Restoring session...")
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        saver.restore(session, MODEL_FILE)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        logging.info("Getting tensors...")
        # example_batch_op = tf.get_collection("examples_batch")[0]
        weights = {
            "layer1": tf.get_collection("layer1_encoder_weight")[0],
            "layer2": tf.get_collection("layer2_encoder_weight")[0],
            "layer3": tf.get_collection("layer3_encoder_weight")[0],
        }
        biases = {
            "layer1": tf.get_collection("layer1_encoder_bias")[0],
            "layer2": tf.get_collection("layer2_encoder_bias")[0],
            "layer3": tf.get_collection("layer3_encoder_bias")[0],
        }

        logging.info("Creating network...")
        preprocessed_example_batch = preprocess_input_batch(example_batch_op, "example_batch")
        with tf.name_scope("network"):
            layer1 = tf.nn.sigmoid(tf.add(tf.matmul(preprocessed_example_batch, weights["layer1"]), biases["layer1"]), name="layer1")
            layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["layer2"]), biases["layer2"]), name="layer2")
            layer3 = tf.nn.softmax(tf.add(tf.matmul(layer2, weights["layer3"]), biases["layer3"]), name="layer3")
            output = layer3

        writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        summary_op = tf.summary.merge_all()

        with gzip.open(OUTPUT_FILE, "wt") as output_file:
            output_writer = csv.writer(output_file, quoting=csv.QUOTE_NONNUMERIC)

            logging.info("Start clasification loop")
            i = 0
            while not coord.should_stop():
                try:
                    if i % CHECKPOINT_FREQUENCY != 0:
                        example_batch, probabilities_batch = session.run(
                            [example_batch_op, output])
                    else:
                        # Log run metadata every n iterations
                        example_batch, probabilities_batch, summary = session.run(
                            [example_batch_op, output, summary_op],
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata)
                        writer.add_run_metadata(run_metadata, "step:{}".format(i))
                        writer.add_summary(summary, i)
                        logging.info("ITERATION: {}, PROCESSED: {} examples".format(i, i * BATCH_SIZE))

                    for example, probabilities in zip(example_batch, probabilities_batch):
                        probabilities = probabilities.tolist()
                        predicted_class = probabilities.index(max(probabilities))

                        # Output format
                        # example (23) meta, tranferred, counter
                        # output (TOTAL_CLASS)
                        # class (1)
                        # total: 24 + TOTAL_CLASS columns
                        output_writer.writerow(example.tolist() + probabilities + [predicted_class])
                except tf.errors.OutOfRangeError:
                    logging.info("Input files exhausted")
                    break

                # Continue classification loop
                i += 1

        coord.request_stop()
        coord.join(threads)

    logging.info("DONE")
