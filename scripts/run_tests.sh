#!/bin/bash
set -e
mypy -p PyFixedReps
pytest
