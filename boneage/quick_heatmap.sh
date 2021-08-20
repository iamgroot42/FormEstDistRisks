#!/bin/bash

python perf_tests.py --ratio_1 0.2 --ratio_2 0.3
python perf_tests.py --ratio_1 0.2 --ratio_2 0.4
python perf_tests.py --ratio_1 0.2 --ratio_2 0.6
python perf_tests.py --ratio_1 0.2 --ratio_2 0.7
python perf_tests.py --ratio_1 0.2 --ratio_2 0.8

python perf_tests.py --ratio_1 0.3 --ratio_2 0.4
python perf_tests.py --ratio_1 0.3 --ratio_2 0.6
python perf_tests.py --ratio_1 0.3 --ratio_2 0.7
python perf_tests.py --ratio_1 0.3 --ratio_2 0.8

python perf_tests.py --ratio_1 0.4 --ratio_2 0.6
python perf_tests.py --ratio_1 0.4 --ratio_2 0.7
python perf_tests.py --ratio_1 0.4 --ratio_2 0.8

python perf_tests.py --ratio_1 0.6 --ratio_2 0.7
python perf_tests.py --ratio_1 0.6 --ratio_2 0.8

python perf_tests.py --ratio_1 0.7 --ratio_2 0.8