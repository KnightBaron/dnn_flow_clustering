#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
python classify_sigmoid_3.py &> /work/pongsakorn-u/tensorflow_classified/log/sigmoid/3outputs/output.txt &
export CUDA_VISIBLE_DEVICES="3"
python classify_sigmoid_4.py &> /work/pongsakorn-u/tensorflow_classified/log/sigmoid/4outputs/output.txt &
