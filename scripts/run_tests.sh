#!/bin/bash
set -e

MYPYPATH=./typings mypy --ignore-missing-imports -p PyFixedReps

export PYTHONPATH=PyFixedReps
python3 -m unittest discover -p "*test_*.py"
