#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
python classify_softmax_3.py &> /work/pongsakorn-u/tensorflow_classified/log/softmax/3outputs/output.txt &
export CUDA_VISIBLE_DEVICES="3"
python classify_softmax_4.py &> /work/pongsakorn-u/tensorflow_classified/log/softmax/4outputs/output.txt &
