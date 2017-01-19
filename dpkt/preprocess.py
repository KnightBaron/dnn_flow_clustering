from __future__ import print_function, division
from socket import inet_ntop, AF_INET
import dpkt
import logging
import csv
import math
import numpy as np
import gzip
import cPickle as pickle
from itertools import imap

"""
|----WINDOW_SIZE-----|
<--DELTA-->|--------------------|

"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
PCAP_FILE = "/home/knightbaron/data/equinix-chicago.dirA.20150917-132500.UTC.anon.pcap.gz"
OUTPUT_FILE = "/home/knightbaron/data/data_500hz_noised.csv.gz"
OUTPUT_FILE_NOIP = "/home/knightbaron/data/data_500hz_noised_noip.csv.gz"
OUTPUT_FILE_NOMETA = "/home/knightbaron/data/data_500hz_noised_nometa.csv.gz"
WINDOW_SIZE = 10000  # ms (10 seconds), must be divisible by SAMPLING_RATE
DELTA = 1000  # ms (1 seconds)
SAMPLING_RATE = 2  # ms
ADD_FLOW_SAMPLING_NOISE = True
FLOW_SAMPLING_RATE = 0.8  # Range (0, 1]
RANDOM_SEED = 42
LOAD_PICKLED_FLOWS = False  # Load PCAP_FILE as pickled flows
SAVE_FLOWS = False  # Ignore when load pickled flows
PICKLE_FILE = "/home/knightbaron/data/equinix-chicago.dirA.20150917-132500.UTC.anon.p.gz"  # Output pickled_flows

# Alternate inputs
# PCAP_FILE = "/home/knightbaron/data/equinix-chicago.dirA.20150219-125911.UTC.anon.pcap.gz"
# PCAP_FILE = "/home/knightbaron/data/equinix-chicago.dirB.20150917-135000.UTC.anon.pcap.gz"

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    flows = dict()

    with gzip.open(PCAP_FILE, "rb") as pcap_file:
        if not LOAD_PICKLED_FLOWS:
            print("LOADING PCAP_FILE...")
            pcap = dpkt.pcap.Reader(pcap_file)

            for (ts, buf) in pcap:  # ts comes in ms
                # print("TIMESTAMP: {}".format(ts))
                try:
                    ip = dpkt.ip.IP(buf)
                    if type(ip) is not dpkt.ip.IP:
                        continue
                    transport = ip.data
                    if (type(transport) is not dpkt.tcp.TCP) and (type(transport) is not dpkt.udp.UDP):
                        continue
                    # print("SOURCE: {}".format(inet_ntop(AF_INET, ip.src)))
                    # print("DESTINATION: {}".format(inet_ntop(AF_INET, ip.dst)))
                    # (proto, src_ip, src_port, dst_ip, dst_port)
                    # proto: 0b0 => TCP, 0b1 => UDP
                    proto = 0b0 if type(transport) is dpkt.tcp.TCP else 0b1
                    key = (proto, ip.src, transport.sport, ip.dst, transport.dport)

                    if ADD_FLOW_SAMPLING_NOISE:
                        if key not in flows:
                            # flows[0] => complete_packets
                            # flows[1] => noised_packets
                            flows[key] = [list(), list()]
                        flows[key][0].append((ts, ip.len))
                        if np.random.uniform() > (1 - FLOW_SAMPLING_RATE):
                            flows[key][1].append((ts, ip.len))
                    else:
                        if key not in flows:
                            flows[key] = list()
                        flows[key].append((ts, ip.len))

                    # DEBUG
                    # if len(flows) > 10000:
                    #     break
                    # END DEBUG

                except dpkt.dpkt.UnpackError:  # Skip non-IP packet
                    pass
                except Exception as e:
                    print("ERROR: {}".format(e.message))
                    raise e
            if SAVE_FLOWS:
                with gzip.open(PICKLE_FILE, "wb") as pickle_file:
                    pickle.dump(flows, pickle_file)
                print("FLOWS SAVED (PICKLED)")
        else:
            print("LOADING PCAP_FILE AS PICKLED FLOWS...")
            flows = pickle.load(pcap_file)

    print("TOTAL FLOWS: {}".format(len(flows)))

    # DEBUG
    # distribution = dict()
    # for (identifier, packets) in flows.iteritems():
    #     duration = packets[-1][0] - packets[0][0]
    #     if duration not in distribution:
    #         distribution[duration] = 0
    #     distribution[duration] += 1
    # for timestamp in sorted(distribution.keys()):
    #     print("{} : {}".format(timestamp, distribution[timestamp]))
    # raw_input()
    # END DEBUG

    with gzip.open(OUTPUT_FILE, "wb") as output_file,\
            gzip.open(OUTPUT_FILE_NOIP, "wb") as output_file_noip,\
            gzip.open(OUTPUT_FILE_NOMETA, "wb") as output_file_nometa:

        for (identifier, packets) in flows.iteritems():  # Loop through each (noised) flow
            # identifier => (proto, ip.src, transport.sport, ip.dst, transport.dport)
            # Timestamp from dpkt is in floating point seconds since epoch (unix timestamp)

            if ADD_FLOW_SAMPLING_NOISE:
                complete_packets = packets[0]
                packets = packets[1]

                # Skip empty flow due to flow sampling noise
                if len(packets) == 0:
                    continue

                # Calculate duration from complete_packets in case of noise adding
                start_time = int(complete_packets[0][0] * 1000)  # Convert to ms
                end_time = int(complete_packets[-1][0] * 1000)  # Convert to ms
            else:
                start_time = int(packets[0][0] * 1000)  # Convert to ms
                end_time = int(packets[-1][0] * 1000)  # Convert to ms

            duration = end_time - start_time
            total_windows = math.ceil(float(duration - WINDOW_SIZE) / float(DELTA))
            total_windows = 0 if (total_windows < 0) else int(total_windows)
            total_windows += 1  # Count the first window (the calculation is 0-indexed)

            # Initialize flow slices
            windows = [
                np.zeros(int(WINDOW_SIZE / SAMPLING_RATE), dtype=int)
                for _ in xrange(total_windows)
            ]
            if ADD_FLOW_SAMPLING_NOISE:
                complete_windows = [
                    np.zeros(int(WINDOW_SIZE / SAMPLING_RATE), dtype=int)
                    for _ in xrange(total_windows)
                ]

            # Loop through each packet in the flow
            for packet in packets:
                timestamp = int(packet[0] * 1000)  # Convert to ms
                relative_timestamp = timestamp - start_time  # Relative to flow start time
                first_window = math.ceil(float((relative_timestamp - WINDOW_SIZE)) / float(DELTA))
                first_window = 0 if (first_window < 0) else int(first_window)
                last_window = math.floor(float(relative_timestamp) / float(DELTA))
                last_window = (total_windows - 1) if last_window > (total_windows - 1) else int(last_window)

                window = first_window
                while window <= last_window:
                    relative_relative_timestamp = relative_timestamp - (DELTA * window)  # Relative to window start time
                    sample_index = int(math.floor(float(relative_relative_timestamp) / float(SAMPLING_RATE)))
                    try:
                        windows[window][sample_index] += packet[1]
                    except IndexError as e:
                        # Handle edge case where sample_index is right at the end of the window
                        if sample_index == (WINDOW_SIZE / SAMPLING_RATE):
                            sample_index -= 1
                            windows[window][sample_index] += packet[1]
                        else:
                            raise e
                    window += 1

            if ADD_FLOW_SAMPLING_NOISE:
                for packet in complete_packets:
                    timestamp = int(packet[0] * 1000)  # Convert to ms
                    relative_timestamp = timestamp - start_time  # Relative to flow start time
                    first_window = math.ceil(float((relative_timestamp - WINDOW_SIZE)) / float(DELTA))
                    first_window = 0 if (first_window < 0) else int(first_window)
                    last_window = math.floor(float(relative_timestamp) / float(DELTA))
                    last_window = (total_windows - 1) if last_window > (total_windows - 1) else int(last_window)

                    window = first_window
                    while window <= last_window:
                        relative_relative_timestamp = relative_timestamp - (DELTA * window)  # Relative to window start time
                        sample_index = int(math.floor(float(relative_relative_timestamp) / float(SAMPLING_RATE)))
                        try:
                            complete_windows[window][sample_index] += packet[1]
                        except IndexError as e:
                            # Handle edge case where sample_index is right at the end of the window
                            if sample_index == (WINDOW_SIZE / SAMPLING_RATE):
                                sample_index -= 1
                                complete_windows[window][sample_index] += packet[1]
                            else:
                                raise e
                        window += 1

                windows = imap(np.append, windows, complete_windows)

            writer = csv.writer(output_file, quoting=csv.QUOTE_NONNUMERIC)
            writer_noip = csv.writer(output_file_noip, quoting=csv.QUOTE_NONNUMERIC)
            writer_nometa = csv.writer(output_file_nometa, quoting=csv.QUOTE_NONNUMERIC)
            for window in windows:
                writer.writerow(
                    (
                        inet_ntop(AF_INET, identifier[1]),  # Source IP
                        inet_ntop(AF_INET, identifier[3]),  # Dest IP
                        identifier[0],  # Protocol: 0b0 => TCP, 0b1 => UDP
                        identifier[2],  # Source Port
                        identifier[4]  # Dest Port
                    ) + tuple(window)
                )
                writer_noip.writerow(
                    (
                        identifier[0],  # Protocol: 0b0 => TCP, 0b1 => UDP
                        identifier[2],  # Source Port
                        identifier[4]  # Dest Port
                    ) + tuple(window)
                )
                writer_nometa.writerow(tuple(window))
            # output_file.flush()

    print("DONE!")
