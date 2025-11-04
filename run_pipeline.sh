#!/usr/bin/env bash

source ./.venv/bin/activate
python3 run_pipeline.py "$@"
deactivate