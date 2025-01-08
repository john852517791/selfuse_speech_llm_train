#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
set -x;

# model_path=./checkpoints # Replace with your model path (download from our huggingface repo).
# llm_path=./Qwen2-7B-Instruct # Replace with your Qwen2-7B-Instruct model path.
input_wav=./assets/question.wav # Replace with your input wav file.
output_wav=./answer.wav # Replace with your output wav file.

top_p=0.8
top_k=20
temperature=0.8

# Replace the CUDA_VISIBLE_DEVICES with your GPU ID.
  # --model_path $model_path \
  # --llm_path $llm_path \
CUDA_VISIBLE_DEVICES=7 python3 bin/inference.py \
  --input_wav $input_wav \
  --output_wav $output_wav \
  --top_p ${top_p} \
  --top_k ${top_k} \
  --temperature ${temperature}
