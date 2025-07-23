#!/bin/bash

declare -a steps=(1 1  2 3 3)
declare -a channels=(5 7 5 3 5)

for i in "${!steps[@]}"; do
    step="${steps[$i]}"
    ch="${channels[$i]}"
    echo "Running with --step $step and --num_channels $ch"
    python preprocess.py --do_padding True --step "$step" --num_channels "$ch"
done

