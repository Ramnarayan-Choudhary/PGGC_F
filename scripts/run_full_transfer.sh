#!/bin/bash
#SBATCH --job-name=pgcg_transfer_attack
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Set Environment
export HF_HOME=/home/ramnarayan.ramniwas/.cache/huggingface
cd /home/ramnarayan.ramniwas/MS_projects/PGCG_Final

echo "=== Step 1: Generating Disguise (Surrogate: Qwen 1.5B) ==="
python3 scripts/generate_disguise.py \
  --input_file data/leaky_mcqs_new.jsonl \
  --output_file sanitized_qwen_transfer.json \
  --iterations 5 \
  --model "/home/ramnarayan.ramniwas/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

echo "=== Step 2: Evaluating Accuracy (Target: Qwen 7B) ==="
python3 scripts/evaluate_disguise.py \
  --input_file sanitized_qwen_transfer.json \
  --output_file results_final_qwen_transfer.json \
  --model "/home/ramnarayan.ramniwas/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

echo "=== Pipeline Complete ==="
