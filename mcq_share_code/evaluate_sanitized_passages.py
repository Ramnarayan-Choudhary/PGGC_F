#!/usr/bin/env python3
"""
Evaluate MCQ accuracy on sanitized/rewritten passages from DP-FUSION, baselines, or DP methods.

This script reads output files and evaluates how well the model can answer MCQ
questions based on privacy-preserving paraphrases.

Usage:
# DP-FUSION mode (evaluate beta sweep):
python mcq/evaluate_sanitized_passages.py --input_dir mcq/rewriting --output_file mcq/rewriting/evaluation_results.json --gpu 0 --hf_token YOUR_TOKEN

# Baseline mode (evaluate public and private paraphrases):
python mcq/evaluate_sanitized_passages.py --baselines --baseline_file mcq/baseline_paraphrases_privacy_in_prompt.json --output_file mcq/baseline_evaluation_privacy_in_prompt.json --gpu 0 --hf_token YOUR_TOKEN

# DP methods mode (evaluate single DP-Decoding or DP-Prompt file):
python mcq/evaluate_sanitized_passages.py --dp_methods --dp_methods_file mcq/dp_decoding/dp_decoding_lambda_0.5.json --output_file mcq/dp_decoding_lambda_0.5_eval.json --gpu 0 --hf_token YOUR_TOKEN
python mcq/evaluate_sanitized_passages.py --dp_methods --dp_methods_file mcq/dp_prompt/dp_prompt_clip_5_temp_1.0.json --output_file mcq/dp_prompt_clip_5_temp_1.0_eval.json --gpu 0 --hf_token YOUR_TOKEN
"""

import os
import json
import argparse
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

def format_mcq_prompt(passage, question, options):
    """
    Create MCQ prompt template matching DP-FUSION_single_grp_mcq.py format.

    Args:
        passage (str): Document passage containing information
        question (str): Multiple choice question
        options (list): List of 4 answer options

    Returns:
        str: Formatted prompt for the model
    """
    system_prompt = "Select the correct option based on the passage provided below. You must output one token i.e A,B,C,D that's it nothing else. Do not output any new lines."
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Passage: {passage}
Question: {question}
Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}<|im_end|>
<|im_start|>assistant
The answer token is:"""

def extract_answer_from_text(text):
    """
    Extract predicted answer (A/B/C/D) from generated text.

    Args:
        text (str): Generated text from model

    Returns:
        str or None: Predicted answer letter (A/B/C/D) or None if not found
    """
    # First try: Look for "The answer token is: X" pattern
    match = re.search(r"The answer token is:\s*([A-D])", text)
    if match:
        return match.group(1).strip()

    # Fallback: Look for letter after "assistant"
    match = re.search(r"assistant.*?([A-D])", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def evaluate_entry(entry, model, tokenizer, device, text_key="redacted_text"):
    """
    Evaluate a single MCQ entry.

    Args:
        entry (dict): Entry from output file with paraphrased text and metadata
        model: Loaded language model
        tokenizer: Model tokenizer
        device: torch device
        text_key (str): Key to extract paraphrased text from ("redacted_text", "public_paraphrase", or "private_paraphrase")

    Returns:
        dict: Evaluation result with prediction and correctness
    """
    # Extract data from entry
    redacted_text = entry.get(text_key, "")
    metadata = entry.get("metadata", {})

    doc_id = metadata.get("doc_id", "unknown")
    question = metadata.get("question", "")
    options = metadata.get("options", [])
    answer_index = metadata.get("answer_index", -1)
    answer_text = metadata.get("answer_text", "")

    # Validate data
    if not redacted_text or not question or len(options) != 4:
        return {
            "doc_id": doc_id,
            "error": "Missing or invalid data",
            "is_correct": False
        }

    # Format MCQ prompt
    prompt = format_mcq_prompt(redacted_text, question, options)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract predicted answer
    predicted_answer = extract_answer_from_text(generated_text)

    # Get correct answer letter
    correct_answer_letter = ["A", "B", "C", "D"][answer_index] if 0 <= answer_index < 4 else None

    # Check correctness
    is_correct = (predicted_answer == correct_answer_letter) if predicted_answer and correct_answer_letter else False

    return {
        "doc_id": doc_id,
        "question": question,
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer_letter,
        "correct_answer_text": answer_text,
        "is_correct": is_correct,
        "generated_text": generated_text
    }

def evaluate_beta_file(beta_value, input_file, model, tokenizer, device, text_key="redacted_text"):
    """
    Evaluate all entries in a single beta output file.

    Args:
        beta_value (str): Beta value (e.g., "0.001")
        input_file (str): Path to input JSON file
        model: Loaded language model
        tokenizer: Model tokenizer
        device: torch device
        text_key (str): Key to extract paraphrased text from

    Returns:
        dict: Results with accuracy and per-entry details
    """
    print(f"\n{'='*60}")
    print(f"Evaluating beta = {beta_value}")
    print(f"{'='*60}")

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Evaluate each entry
    results = []
    correct_count = 0

    for entry in tqdm(data, desc=f"Beta {beta_value}"):
        result = evaluate_entry(entry, model, tokenizer, device, text_key=text_key)
        results.append(result)

        if result.get("is_correct", False):
            correct_count += 1

    # Calculate metrics
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    print(f"\nResults for beta = {beta_value}:")
    print(f"  Accuracy: {correct_count}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

    return {
        "beta": beta_value,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "results": results
    }

def evaluate_baseline_file(input_file, model, tokenizer, device):
    """
    Evaluate both public and private paraphrases from baseline file.

    Args:
        input_file (str): Path to baseline JSON file
        model: Loaded language model
        tokenizer: Model tokenizer
        device: torch device

    Returns:
        dict: Results for both public and private paraphrases
    """
    print(f"\n{'='*60}")
    print(f"Evaluating baseline file: {input_file}")
    print(f"{'='*60}")

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Evaluate PUBLIC paraphrases
    print("\n--- Evaluating PUBLIC paraphrases ---")
    public_results = []
    public_correct = 0

    for entry in tqdm(data, desc="Public"):
        result = evaluate_entry(entry, model, tokenizer, device, text_key="public_paraphrase")
        public_results.append(result)
        if result.get("is_correct", False):
            public_correct += 1

    # Evaluate PRIVATE paraphrases
    print("\n--- Evaluating PRIVATE paraphrases ---")
    private_results = []
    private_correct = 0

    for entry in tqdm(data, desc="Private"):
        result = evaluate_entry(entry, model, tokenizer, device, text_key="private_paraphrase")
        private_results.append(result)
        if result.get("is_correct", False):
            private_correct += 1

    # Calculate metrics
    total = len(data)
    public_accuracy = public_correct / total if total > 0 else 0.0
    private_accuracy = private_correct / total if total > 0 else 0.0

    print(f"\nResults:")
    print(f"  PUBLIC:  {public_correct}/{total} = {public_accuracy:.4f} ({public_accuracy*100:.2f}%)")
    print(f"  PRIVATE: {private_correct}/{total} = {private_accuracy:.4f} ({private_accuracy*100:.2f}%)")

    return {
        "public": {
            "accuracy": public_accuracy,
            "correct": public_correct,
            "total": total,
            "results": public_results
        },
        "private": {
            "accuracy": private_accuracy,
            "correct": private_correct,
            "total": total,
            "results": private_results
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate MCQ accuracy on sanitized passages")
    parser.add_argument("--input_dir", type=str, default="mcq/rewriting",
                        help="Directory containing output_beta_*.json files (for DP-FUSION mode)")
    parser.add_argument("--output_file", type=str, default="mcq/rewriting/evaluation_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU to use")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Hugging Face token")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model ID to use")

    # Baseline mode arguments
    parser.add_argument("--baselines", action="store_true",
                        help="Evaluate baseline paraphrases (public and private)")
    parser.add_argument("--baseline_file", type=str,
                        help="Path to baseline JSON file (required if --baselines is set)")

    # DP methods mode arguments
    parser.add_argument("--dp_methods", action="store_true",
                        help="Evaluate DP-Decoding or DP-Prompt paraphrases")
    parser.add_argument("--dp_methods_file", type=str,
                        help="Path to DP method output file (required if --dp_methods is set)")

    args = parser.parse_args()

    # Validate arguments
    if args.baselines and not args.baseline_file:
        parser.error("--baseline_file is required when --baselines is set")

    if args.dp_methods and not args.dp_methods_file:
        parser.error("--dp_methods_file is required when --dp_methods is set")

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/l/users/nils.lukas/models"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.hf_token,
        cache_dir="/l/users/nils.lukas/models"
    )
    print("Model loaded successfully")

    all_results = {}

    if args.dp_methods:
        # ============================================
        # DP METHODS MODE: Evaluate single DP-Decoding or DP-Prompt file
        # ============================================
        print("\n" + "="*60)
        print("DP METHODS EVALUATION MODE")
        print("="*60)

        # Extract config name from filename
        filename = os.path.basename(args.dp_methods_file)
        config_name = filename.replace('.json', '')

        # Load data
        with open(args.dp_methods_file, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries from {args.dp_methods_file}")
        print(f"Config: {config_name}")

        # Evaluate each entry using redacted_text key
        results = []
        correct_count = 0

        for entry in tqdm(data, desc=f"Evaluating {config_name}"):
            result = evaluate_entry(entry, model, tokenizer, device, text_key="redacted_text")
            results.append(result)

            if result.get("is_correct", False):
                correct_count += 1

        # Calculate metrics
        total = len(results)
        accuracy = correct_count / total if total > 0 else 0.0

        print(f"\nResults for {config_name}:")
        print(f"  Total samples evaluated: {total}")
        print(f"  Correct predictions: {correct_count}")
        print(f"  Accuracy: {correct_count}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

        all_results[config_name] = {
            "config": config_name,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total,
            "samples_evaluated": total,
            "results": results
        }

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save results
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")

        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {args.output_file}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{config_name}: {correct_count:3d}/{total:3d} = {accuracy:.4f} ({accuracy*100:6.2f}%)")

    elif args.baselines:
        # ============================================
        # BASELINE MODE: Evaluate public and private
        # ============================================
        print("\n" + "="*60)
        print("BASELINE EVALUATION MODE")
        print("="*60)

        baseline_results = evaluate_baseline_file(args.baseline_file, model, tokenizer, device)
        all_results["baseline"] = baseline_results

        # Save results
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")

        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {args.output_file}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        pub_acc = baseline_results['public']['accuracy']
        pub_correct = baseline_results['public']['correct']
        pub_total = baseline_results['public']['total']

        priv_acc = baseline_results['private']['accuracy']
        priv_correct = baseline_results['private']['correct']
        priv_total = baseline_results['private']['total']

        print(f"PUBLIC:  {pub_correct:3d}/{pub_total:3d} = {pub_acc:.4f} ({pub_acc*100:6.2f}%)")
        print(f"PRIVATE: {priv_correct:3d}/{priv_total:3d} = {priv_acc:.4f} ({priv_acc*100:6.2f}%)")

    else:
        # ============================================
        # DP-FUSION MODE: Evaluate beta sweep
        # ============================================
        print("\n" + "="*60)
        print("DP-FUSION EVALUATION MODE")
        print("="*60)

        # Find all beta output files
        beta_values = ["0.001", "0.01", "0.1", "0.5", "1.0", "2.5", "5.0", "10.0"]

        for beta in beta_values:
            input_file = os.path.join(args.input_dir, f"output_beta_{beta}.json")

            if not os.path.exists(input_file):
                print(f"\nWarning: File not found: {input_file}")
                continue

            # Evaluate this beta value
            beta_results = evaluate_beta_file(beta, input_file, model, tokenizer, device)
            all_results[f"beta_{beta}"] = beta_results

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save results
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")

        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {args.output_file}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        for beta_key in sorted(all_results.keys()):
            result = all_results[beta_key]
            beta = result['beta']
            accuracy = result['accuracy']
            correct = result['correct']
            total = result['total']
            print(f"Beta {beta:>6s}: {correct:3d}/{total:3d} = {accuracy:.4f} ({accuracy*100:6.2f}%)")

if __name__ == "__main__":
    main()
