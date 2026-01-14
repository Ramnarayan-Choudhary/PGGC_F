import argparse
import json
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Basic Config
CACHE_DIR = os.getenv("HF_HOME", "/home/ramnarayan.ramniwas/.cache/huggingface")

def get_highest_prob_option(logits, tokenizer, option_chars):
    # Map chars to token IDs
    option_ids = []
    for char in option_chars:
        # Note: Llama 3 tokenizer might encode " A" vs "A". We assuming standard single token lookup here.
        # Ideally we use the same logic as the main script.
        # For Llama 3: "A", "B", "C", "D"
        ids = tokenizer.encode(char, add_special_tokens=False)
        if len(ids) == 0:
            # Fallback
            ids = tokenizer.encode(" " + char, add_special_tokens=False)
        option_ids.append(ids[-1]) # Take the last token if multiple (unlikely for single char)
    
    # Extract logits for these specific IDs
    relevant_logits = logits[torch.tensor(option_ids).to(logits.device)]
    probs = torch.softmax(relevant_logits, dim=0)
    
    best_idx = torch.argmax(probs).item()
    return option_chars[best_idx], probs.tolist()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sanitized Prompts on Target Model")
    parser.add_argument("--input_file", type=str, required=True, help="JSON File containing 'sanitized_prompt' or 'final_prompt'")
    parser.add_argument("--model", type=str, required=True, help="Target Model ID or Path")
    parser.add_argument("--output_file", type=str, default="final_evaluation_results.json")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items.")
    
    # 2. Load Model
    print(f"Loading Target Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    model.eval() # Inference Mode
    
    correct_count = 0
    total = 0
    
    # 3. Evaluate
    for item in tqdm(data, desc="Evaluating"):
        # We look for the final prompt. 
        # run_defense.py saves it as 'final_prompt' (the raw text) or we reconstruct it?
        # Let's check keys. Usually it's good to save the 'sanitized_text' text.
        
        # Fallback: if 'final_prompt' exists, use it.
        prompt = item.get("final_prompt")
        
        # If not, try 'my defence new full prompt'
        if not prompt:
            prompt = item.get("my defence new full prompt")
            
        if not prompt:
            print(f"Skipping item {item.get('doc_id')}: No 'final_prompt' field found.")
            continue
            
        # Determine Ground Truth
        correct_char = None
        if "target_label" in item and len(item["target_label"]) == 1: # Legacy check
             correct_char = item["target_label"]
        elif "correct" in item:
             correct_char = item["correct"]
        elif "answer_index" in item:
             idx = item["answer_index"]
             if isinstance(idx, int) and 0 <= idx < 4:
                 correct_char = ['A', 'B', 'C', 'D'][idx]
        
        if not correct_char:
            # Last resort: Try to find which option matches 'answer_text'
            ans_text = item.get("answer_text")
            options = item.get("options")
            if ans_text and options:
                try:
                    idx = options.index(ans_text)
                    correct_char = ['A', 'B', 'C', 'D'][idx]
                except ValueError:
                    pass

        if not correct_char:
            print(f"Skipping item {item.get('doc_id')}: No ground truth label found (checked 'answer_index', 'correct').")
            continue
            
        # Tokenize
            
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
        # Check Answer (Assuming A, B, C, D)
        pred_char, probs = get_highest_prob_option(logits, tokenizer, ["A", "B", "C", "D"])
        
        is_correct = (pred_char == correct_char)
        if is_correct:
            correct_count += 1
        total += 1
        
        # Update Item
        item["target_prediction"] = pred_char
        item["target_correct"] = is_correct
        item["target_probs"] = probs
        
    # 4. Save and Report
    accuracy = correct_count / total if total > 0 else 0
    print(f"\n=== Evaluation Complete ===")
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total})")
    
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved detailed results to {args.output_file}")

if __name__ == "__main__":
    main()
