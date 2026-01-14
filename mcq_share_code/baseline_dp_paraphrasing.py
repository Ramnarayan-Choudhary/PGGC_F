#!/usr/bin/env python3
"""
Generate DP-Decoding and DP-Prompt baseline paraphrases.

This script uses the SAME inference loop as baseline_dp_mcq.py with DP mechanisms.
It generates paraphrases using:
- DP-Decoding: λ*softmax(z/T) + (1-λ)*uniform
- DP-Prompt: clamp logits to [-clip_width/2, +clip_width/2]

Usage:
DP-Decoding: python mcq/baseline_dp_paraphrasing.py --dp_method dp_decoding --dp_lambda 0.5 --temperature 1.0 --input_file input_converted.json --start 0 --end 100 --gpu 0 --output_file mcq/dp_decoding_paraphrases.json --hf_token YOUR_TOKEN
DP-Prompt: python mcq/baseline_dp_paraphrasing.py --dp_method dp_prompt --clip_width 5.0 --temperature 1.0 --input_file input_converted.json --start 0 --end 100 --gpu 0 --output_file mcq/dp_prompt_paraphrases.json --hf_token YOUR_TOKEN
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from bisect import bisect_right
import time
import torch.nn.functional as F
import pickle
import argparse
import math
from typing import Tuple

# Command line argument parsing
parser = argparse.ArgumentParser(description="Generate DP baseline paraphrases (DP-Decoding or DP-Prompt)")
parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
parser.add_argument("--start", type=int, required=True, help="Start index")
parser.add_argument("--end", type=int, required=True, help="End index")
parser.add_argument("--gpu", type=str, required=True, help="GPU to use")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID")
parser.add_argument("--output_file", type=str, required=True, help="Output file")
parser.add_argument("--output_file_pickle", type=str, default="", help="Output pickle file")
parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
parser.add_argument("--max_tokens", type=int, default=900, help="Max tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

# DP mechanism arguments
parser.add_argument("--dp_method", type=str, choices=["dp_decoding", "dp_prompt"], required=True,
                    help="DP mechanism: 'dp_decoding' or 'dp_prompt'")
parser.add_argument("--dp_lambda", type=float, default=0.5,
                    help="λ for DP-Decoding (uniform mix weight)")
parser.add_argument("--clip_width", type=float, default=5.0,
                    help="Clipping width for DP-Prompt")

args = parser.parse_args()

# Set default pickle file if not provided
if not args.output_file_pickle:
    args.output_file_pickle = args.output_file.replace('.json', '.pkl')

# Configure GPU
# NOTE: Commented out to allow shell script to control GPU assignment via CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Constants
PLACEHOLDER_TOKEN = "_"
HF_TOKEN = args.hf_token
MAX_TOKENS_GLOBAL = args.max_tokens

print(f"DP Method: {args.dp_method}")
if args.dp_method == "dp_decoding":
    print(f"DP-Decoding λ: {args.dp_lambda}")
else:
    print(f"DP-Prompt clip_width: {args.clip_width}")
print(f"Temperature: {args.temperature}")

def log(message, level="INFO"):
    """Simple logging function with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def replace_sequences_with_placeholder_fast(text, word_offsets, placeholder, tokenizer):
    """Replace tokens falling within provided word offset ranges with placeholder tokens."""
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']

    word_offsets = sorted(word_offsets, key=lambda x: x[0])
    starts = [wo[0] for wo in word_offsets]
    ends = [wo[1] for wo in word_offsets]

    for i, (t_start, t_end) in enumerate(offsets):
        if t_start == t_end:
            continue
        idx = bisect_right(starts, t_end)
        while idx > 0:
            idx -= 1
            w_start, w_end = starts[idx], ends[idx]
            if w_end > t_start and w_start < t_end:
                input_ids[i] = placeholder_id
                break

    return input_ids

def format_prompt_new_template(tokenizer, private_doc, placeholder):
    """Create paraphrasing prompt (same as DP-FUSION)."""
    return(f"""<|im_start|>system
You are given a passage that may contain placeholders (underscores) or incomplete data. Your job is to produce a natural paraphrase. Do not use any underscores or placeholders in your output. If data is missing, just omit it or paraphrase gracefully. Do not output anything except the paraphrase. Make sure retain all information from the source document.<|im_end|>
<|im_start|>user
Document:\n\n{private_doc}
Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, as {placeholder} indicates redacted text. To ensure the generated text is as a natural as possible, you must never output the {placeholder} themselves. <|im_end|>
<|im_start|>assistant
Sure. Here is the paraphrased document without underscores or placeholders:""")

def apply_dp_mechanism(
    logits: torch.Tensor,
    dp_method: str,
    dp_lambda: float,
    clip_width: float,
    temperature: float
) -> Tuple[torch.Tensor, float]:
    """
    Apply DP mechanism to raw logits (EXACT COPY from baseline_dp_mcq.py).

    Returns:
        probs: Probability distribution to sample from
        eps_step: Privacy cost (epsilon) for this single token
    """
    V = logits.shape[-1]

    if dp_method == "dp_decoding":
        # DP-Decoding: q' = λ*softmax(z/T) + (1-λ)*uniform
        scaled = logits / temperature
        q = F.softmax(scaled, dim=-1)
        uniform = torch.full_like(q, 1.0 / V)

        q_prime = dp_lambda * q + (1.0 - dp_lambda) * uniform

        # ε_step = log(1 + (|V|-1)λ / (1-λ))
        if dp_lambda <= 0.0:
            eps_step = 0.0
        elif dp_lambda >= 1.0:
            eps_step = float('inf')
        else:
            eps_step = math.log(1.0 + (V - 1.0) * dp_lambda / (1.0 - dp_lambda))

        return q_prime, eps_step

    elif dp_method == "dp_prompt":
        # DP-Prompt: clamp logits to [-clip_width/2, +clip_width/2]
        b1 = -clip_width / 2.0
        b2 = +clip_width / 2.0
        proc_logits = torch.clamp(logits, b1, b2)
        scaled = proc_logits / temperature
        probs = F.softmax(scaled, dim=-1)

        # ε_step = 2 * clip_width / T
        eps_step = 2.0 * clip_width / temperature

        return probs, eps_step

    else:
        raise ValueError(f"Unknown dp_method: {dp_method}")

def dp_generation_incremental(
    token_ids, model, tokenizer,
    temperature=1.0, max_new_tokens=50, device_map=None,
    dp_method="dp_decoding", dp_lambda=0.5, clip_width=5.0
):
    """
    DP generation with EXACT same inference loop as baseline_dp_mcq.py.
    Applies DP-Decoding or DP-Prompt mechanism.
    """
    eos_id = tokenizer.eos_token_id

    # Initialize epsilon tracking
    epsilon_total = 0.0
    n_tokens_sampled = 0

    # Determine device
    if device_map and isinstance(device_map, dict):
        first_device = next(iter(device_map.values()))
        device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    else:
        device = model.device

    # Ensure token_ids is 1D tensor on device
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    token_ids = token_ids.to(device)

    # === INITIAL PASS: Process prefix to build cache ===
    # EXACT SAME as baseline_dp_mcq.py
    input_batch = token_ids.unsqueeze(0)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        outputs = model(input_ids=input_batch, use_cache=True, past_key_values=None)
    past = outputs.past_key_values

    # Extract logits for the last token
    last_logits = outputs.logits[:, -1, :].squeeze(0)

    # Apply DP mechanism to get sampling distribution
    p_dist, eps_step = apply_dp_mechanism(
        last_logits,
        dp_method=dp_method,
        dp_lambda=dp_lambda,
        clip_width=clip_width,
        temperature=temperature
    )
    epsilon_total += eps_step

    # Sample the first new token
    next_token = torch.multinomial(p_dist, 1).item()
    n_tokens_sampled += 1

    # Append the new token to the sequence
    token_ids = torch.cat([token_ids, torch.tensor([next_token], device=device)], dim=0)

    # === INCREMENTAL LOOP: Process only the new token each step ===
    # EXACT SAME as baseline_dp_mcq.py
    for step in range(1, max_new_tokens):
        # Create new input: tensor of shape (1, 1) with the new token
        new_token_input = torch.tensor([[next_token]], device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids=new_token_input, past_key_values=past, use_cache=True)
        past = outputs.past_key_values

        # Extract logits for the new token
        last_logits = outputs.logits[:, -1, :].squeeze(0)

        # Apply DP mechanism
        p_dist, eps_step = apply_dp_mechanism(
            last_logits,
            dp_method=dp_method,
            dp_lambda=dp_lambda,
            clip_width=clip_width,
            temperature=temperature
        )
        epsilon_total += eps_step

        # Sample the next token
        next_token = torch.multinomial(p_dist, 1).item()
        n_tokens_sampled += 1

        # Append the new token to the sequence
        token_ids = torch.cat([token_ids, torch.tensor([next_token], device=device)], dim=0)

        # ======= END‑OF‑STEP CLEAN‑UP =======
        del outputs, last_logits
        torch.cuda.empty_cache()
        # ====================================

        if next_token == eos_id:
            break

    # Return the final text
    final_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()

    return final_text, epsilon_total, n_tokens_sampled

def process_entry(entry, model, tokenizer, device_map=None):
    """Process a single entry to generate DP paraphrase from private document."""
    output_entry = {}

    passage_lines = entry["passage"]
    private_entities_all = entry["private_entities"]

    # Combine all lines into a single passage
    MAX_CHARS = 10000
    passage_new = ""
    going_offset = 0

    for i, line_text in enumerate(passage_lines):
        line_len = len(line_text) + 1
        if going_offset + line_len > MAX_CHARS:
            break

        passage_new += line_text + " "
        going_offset += line_len

    MAX_TOKENS = min(tokenizer(passage_new, return_tensors="pt")["input_ids"].shape[1], MAX_TOKENS_GLOBAL)

    # Create prompt with PRIVATE (original) document
    private_prompt = format_prompt_new_template(tokenizer, passage_new, placeholder=PLACEHOLDER_TOKEN)

    # Build private version (original, non-redacted)
    private_ids = tokenizer(private_prompt, add_special_tokens=False)['input_ids']

    # Generate paraphrase using DP mechanism on PRIVATE document
    print(f"Generating DP paraphrase ({args.dp_method}) from private document...")
    paraphrase_full, epsilon, n_tokens = dp_generation_incremental(
        private_ids,
        model,
        tokenizer,
        temperature=args.temperature,
        max_new_tokens=MAX_TOKENS,
        device_map=device_map,
        dp_method=args.dp_method,
        dp_lambda=args.dp_lambda,
        clip_width=args.clip_width
    )

    # Extract clean paraphrase (same as DP-FUSION)
    if "Here is the paraphrased document without underscores or placeholders:" in paraphrase_full:
        redacted_text = paraphrase_full.split("Here is the paraphrased document without underscores or placeholders:")[1]
    else:
        redacted_text = paraphrase_full

    # Build output entry (matching DP-FUSION format)
    output_entry = {
        "passage": entry["passage"],
        "private_entities": entry["private_entities"],
    }

    if "metadata" in entry:
        output_entry["metadata"] = entry["metadata"]

    output_entry.update({
        "dp_method": args.dp_method,
        "dp_lambda": args.dp_lambda if args.dp_method == "dp_decoding" else None,
        "clip_width": args.clip_width if args.dp_method == "dp_prompt" else None,
        "temperature": args.temperature,
        "redacted_text": redacted_text,
        "epsilon": epsilon,
        "n_tokens": n_tokens,
    })

    print(f"ε={epsilon:.2f}, {n_tokens} tokens")
    print(f"Redacted text: {redacted_text[:100]}...")
    print("=" * 50)

    return output_entry

def main():
    start_time = time.time()
    log(f"Starting DP paraphrase generation ({args.dp_method}) for indices {args.start} to {args.end}")

    # Load input data
    try:
        with open(args.input_file, "r") as json_file:
            big_data = json.load(json_file)
        data = big_data[args.start:args.end]
        log(f"Loaded {len(data)} entries from {args.input_file}")
    except Exception as e:
        log(f"Error loading input data: {str(e)}", "ERROR")
        return

    # Load model and tokenizer
    try:
        log(f"Loading model {args.model_id}")
        device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map=device_map,
            cache_dir="/l/users/nils.lukas/models"
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            token=HF_TOKEN,
            cache_dir="/l/users/nils.lukas/models"
        )
        log("Model loaded successfully")
    except Exception as e:
        log(f"Error loading model: {str(e)}", "ERROR")
        return

    # Process entries
    output_data = []

    for idx, entry in enumerate(tqdm(data, desc="Processing entries")):
        result = process_entry(entry, model, tokenizer, device_map)
        output_data.append(result)

        # Save intermediate results
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            with open(args.output_file_pickle, "wb") as f:
                pickle.dump(output_data, f)
            log(f"Saved intermediate results after processing {idx + 1} entries")
        except Exception as e:
            log(f"Error saving intermediate results: {str(e)}", "ERROR")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final results
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        with open(args.output_file_pickle, "wb") as f:
            pickle.dump(output_data, f)
        log(f"Saved final results to {args.output_file}")
    except Exception as e:
        log(f"Error saving final results: {str(e)}", "ERROR")

    elapsed_time = time.time() - start_time
    log(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
