#!/bin/bash

python meta.py --filter Male --second $1 --focus combined --first_n_conv 1 --start_n_fc 2
python meta.py --filter Male --second $1 --focus combined --first_n_conv 2 --start_n_fc 1
python meta.py --filter Male --second $1 --focus combined --first_n_conv 3 --start_n_fc 0