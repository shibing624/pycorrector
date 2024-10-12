#!/bin/bash

# Define the models and datasets
models=("kenlm" "macbert" "qwen1.5b" "qwen7b")
datasets=("sighan" "ec_law" "mcsc")

# Loop through each model and dataset combination
for model in "${models[@]}"; do
  for data in "${datasets[@]}"; do
    echo "Evaluating model: $model with dataset: $data"
    python evaluate_models.py --model $model --data $data
    echo "Done evaluating model: $model with dataset: $data"
  done
done