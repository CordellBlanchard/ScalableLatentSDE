#!/bin/bash

mkdir ../Datasets
wget -r -N -c -np https://physionet.org/files/challenge-2012/1.0.0/ -P ../Datasets

python3 src/datasets/physionet/preprocess.py