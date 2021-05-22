#!/bin/bash
for i in {1..10}
do
    python perf_tests.py 0.5 $1
done