#!/bin/bash
python tools/test.py configs/htc_r50.py epoch_20.pth --json_out $AICROWD_PREDICTIONS_OUTPUT_PATH

