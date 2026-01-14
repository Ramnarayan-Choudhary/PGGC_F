import json
import argparse
import sys
from collections import Counter

def calculate_accuracy(file_path, label):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[{label}] File not found: {file_path}")
        return

    if not data:
        print(f"[{label}] No data in file.")
        return

    correct_count = 0
    total_count = 0
    predictions = []
    
    print(f"\n--- Analysis for: {label} ---")
    
    for i, item in enumerate(data):
        # Handle different formats (Baseline vs PGCG)
        
        # PGCG Format
        if "eval_prediction" in item:
            pred = item.get("eval_prediction")
            truth = item.get("answer_text_raw", "") # Or we need to look up correct answer char
            # In run_defense.py we saw: correct_answer_char = item.get("correct") or item.get("answer_index") mapped
            # Let's rely on 'eval_correct' boolean if it exists
            is_correct = item.get("eval_correct", False)
            
            # Explicit check if missing
            if is_correct:
                correct_count += 1
            predictions.append(pred)
            total_count += 1
            
        # Baseline Format
        elif "is_correct_logits" in item:
            # Baseline script saves explicit booleans
            if item.get("is_correct_logits", False):
                correct_count += 1
            predictions.append(item.get("predicted_answer_from_logits"))
            total_count += 1
            
        else:
            # Fallback or unrecognized
            continue

    if total_count == 0:
        print("No valid entries found to evaluate.")
        return

    acc = correct_count / total_count
    print(f"Total Items: {total_count}")
    print(f"Correct:     {correct_count}")
    print(f"Accuracy:    {acc:.2%}")
    print(f"Distribution: {Counter(predictions)}")

def main():
    print("=== Model Accuracy Comparison ===")
    calculate_accuracy("baseline_public_output.json", "Baseline (Public Mode)")
    calculate_accuracy("baseline_private_output.json", "Baseline (Private Mode)")
    calculate_accuracy("results_final_pgcg.json", "PGCG Defense (Full)")
    calculate_accuracy("results_final_pgcg_25iter.json", "PGCG Defense (25 Iterations)")

if __name__ == "__main__":
    main()
