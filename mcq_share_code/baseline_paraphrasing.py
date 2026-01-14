#!/usr/bin/env python3
"""
Generate public and private baseline paraphrases for comparison with DP-FUSION.

This script uses the SAME inference loop as baseline_dp_mcq.py but WITHOUT any DP mechanisms.
It generates two paraphrases per entry:
1. Public paraphrase: Using redacted passage (entities replaced with _)
2. Private paraphrase: Using full passage (all entities visible)

Usage:
python mcq/baseline_paraphrasing.py --input_file input_converted.json --start 0 --end 100 --gpu 0 --output_file mcq/baseline_paraphrases_privacy_in_prompt.json --hf_token YOUR_TOKEN --temperature 1.0 --max_tokens 900

"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import numpy as np
from bisect import bisect_left, bisect_right
import time
import torch.nn.functional as F
import pickle
import argparse

# Command line argument parsing
parser = argparse.ArgumentParser(description="Generate public/private baseline paraphrases")
parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
parser.add_argument("--start", type=int, required=True, help="Start index")
parser.add_argument("--end", type=int, required=True, help="End index")
parser.add_argument("--gpu", type=str, required=True, help="GPU to use")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID")
parser.add_argument("--output_file", type=str, default="mcq/baseline_paraphrases.json", help="Output file")
parser.add_argument("--output_file_pickle", type=str, default="mcq/baseline_paraphrases.pkl", help="Output pickle file")
parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
parser.add_argument("--max_tokens", type=int, default=900, help="Max tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

args = parser.parse_args()

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Constants
PLACEHOLDER_TOKEN = "_"
HF_TOKEN = args.hf_token
MAX_TOKENS_GLOBAL = args.max_tokens

def log(message, level="INFO"):
    """Simple logging function with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def replace_sequences_with_placeholder_fast(text, word_offsets, placeholder, tokenizer):
    """
    Replace tokens falling within provided word offset ranges with placeholder tokens.
    """
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

def format_prompt_new_template_privacy(tokenizer, private_doc, placeholder):
    """
    Create a model-agnostic prompt for paraphrasing without placeholder markers.
    
    Args:
        tokenizer: Hugging Face tokenizer
        private_doc (str): Document to paraphrase (may contain underscores)
        
    Returns:
        str: Formatted prompt for the model
    """
    return(f"""<|im_start|>system
You are given a passage that may contain placeholders (underscores) or incomplete data. Your job is to produce a natural paraphrase of this for ensuring privacy. Do not use any underscores or placeholders in your output. If data is missing, just omit it or paraphrase gracefully. Do not output anything except the paraphrase. Make sure retain all information from the source document.<|im_end|>
<|im_start|>user
Document:\n\n{private_doc}
Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, as {placeholder} indicates redacted text. To ensure the generated text is as a natural as possible, you must never output the {placeholder} themselves. <|im_end|>
<|im_start|>assistant
Sure. Here is the paraphrased document without underscores or placeholders:""")

def format_prompt_new_template(tokenizer, private_doc, placeholder):
    """
    Create paraphrasing prompt (same as DP-FUSION_simple_fixed.py).
    """
    return(f"""<|im_start|>system
You are given a passage that may contain placeholders (underscores) or incomplete data. Your job is to produce a natural paraphrase. Do not use any underscores or placeholders in your output. If data is missing, just omit it or paraphrase gracefully. Do not output anything except the paraphrase. Make sure retain all information from the source document.<|im_end|>
<|im_start|>user
Document:\n\n{private_doc}
Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, as {placeholder} indicates redacted text. To ensure the generated text is as a natural as possible, you must never output the {placeholder} themselves. <|im_end|>
<|im_start|>assistant
Sure. Here is the paraphrased document without underscores or placeholders:""")

def simple_generation_incremental(
    token_ids, model, tokenizer,
    temperature=1.0, max_new_tokens=50, device_map=None
):
    """
    Simple generation with EXACT same inference loop as baseline_dp_mcq.py
    but WITHOUT any DP mechanisms. Pure temperature-based sampling.

    This ensures fair comparison - the only difference between this and DP methods
    is the absence of privacy mechanisms.
    """
    eos_id = tokenizer.eos_token_id

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
    input_batch = token_ids.unsqueeze(0)  # Add batch dimension: (1, seq_len)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        outputs = model(input_ids=input_batch, use_cache=True, past_key_values=None)
    past = outputs.past_key_values

    # Extract logits for the last token
    last_logits = outputs.logits[:, -1, :].squeeze(0)  # shape: (vocab_size,)

    # Apply temperature and softmax (NO DP mechanism)
    scaled = last_logits / temperature
    p_dist = F.softmax(scaled, dim=-1)

    # Sample the first new token
    next_token = torch.multinomial(p_dist, 1).item()

    # Append the new token to the sequence
    token_ids = torch.cat([token_ids, torch.tensor([next_token], device=device)], dim=0)

    # === INCREMENTAL LOOP: Process only the new token each step ===
    # EXACT SAME as baseline_dp_mcq.py
    for step in range(1, max_new_tokens):
        # Create new input: tensor of shape (1, 1) with the new token
        new_token_input = torch.tensor([[next_token]], device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids=new_token_input, past_key_values=past, use_cache=True)
        past = outputs.past_key_values  # Update the cache

        # Extract logits for the new token
        last_logits = outputs.logits[:, -1, :].squeeze(0)

        # Apply temperature and softmax (NO DP mechanism)
        scaled = last_logits / temperature
        p_dist = F.softmax(scaled, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(p_dist, 1).item()

        # Append the new token to the sequence
        token_ids = torch.cat([token_ids, torch.tensor([next_token], device=device)], dim=0)

        # ======= END‑OF‑STEP CLEAN‑UP =======
        # EXACT SAME as baseline_dp_mcq.py
        del outputs, last_logits
        torch.cuda.empty_cache()
        # ====================================

        if next_token == eos_id:
            break

    # Return the final text
    final_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()

    return final_text

def process_entry(entry, model, tokenizer, device_map=None):
    """
    Process a single entry to generate public and private baseline paraphrases.
    """
    output_entry = {}

    passage_lines = entry["passage"]
    private_entities_all = entry["private_entities"]

    # Combine all lines into a single passage with offsets adjustment
    MAX_CHARS = 10000

    passage_new = ""
    offsets_new = {}
    going_offset = 0

    for i, line_text in enumerate(passage_lines):
        line_len = len(line_text) + 1

        if going_offset + line_len > MAX_CHARS:
            break

        entity_info_list = private_entities_all[i] if i < len(private_entities_all) else []
        for entity_info in entity_info_list:
            s, e = entity_info["offset"]
            e_type = entity_info["type"]
            offsets_new.setdefault(e_type, []).append((s + going_offset, e + going_offset))

        passage_new += line_text + " "
        going_offset += line_len

    MAX_TOKENS = min(tokenizer(passage_new, return_tensors="pt")["input_ids"].shape[1], MAX_TOKENS_GLOBAL)

    # Create prompts for both public and private (using privacy variant)
    private_prompt = format_prompt_new_template_privacy(tokenizer, passage_new, placeholder=PLACEHOLDER_TOKEN)

    # Find where the actual document starts in the prompt
    document_marker = "Document:\n\n"
    document_start_index = private_prompt.find(document_marker)
    if document_start_index == -1:
        raise ValueError("Cannot find 'Document:\\n\\n' in the prompt.")

    # Calculate prefix length
    prefix_length = document_start_index + len(document_marker)

    # Adjust offsets to account for the prefix
    for key in offsets_new.keys():
        offsets_new[key] = [
            (start + prefix_length, end + prefix_length)
            for start, end in offsets_new[key]
        ]

    # Build private version (original prompt)
    private_ids = tokenizer(private_prompt, add_special_tokens=False)['input_ids']

    # Build public version (all entities redacted)
    all_offsets = []
    for off_list in offsets_new.values():
        all_offsets.extend(off_list)
    public_ids = replace_sequences_with_placeholder_fast(private_prompt, all_offsets, PLACEHOLDER_TOKEN, tokenizer)

    # Generate paraphrases using EXACT same inference loop
    print(f"Generating public paraphrase...")
    public_paraphrase = simple_generation_incremental(
        public_ids,
        model,
        tokenizer,
        temperature=args.temperature,
        max_new_tokens=MAX_TOKENS,
        device_map=device_map
    )

    print(f"Generating private paraphrase...")
    private_paraphrase = simple_generation_incremental(
        private_ids,
        model,
        tokenizer,
        temperature=args.temperature,
        max_new_tokens=MAX_TOKENS,
        device_map=device_map
    )

    # Decode full prompts for saving
    public_text = tokenizer.decode(public_ids, skip_special_tokens=True)
    private_text = tokenizer.decode(private_ids, skip_special_tokens=True)

    # Extract clean paraphrases (after the prompt)
    public_paraphrase_clean = public_paraphrase
    private_paraphrase_clean = private_paraphrase

    if "Here is the paraphrased document without underscores or placeholders:" in public_paraphrase:
        public_paraphrase_clean = public_paraphrase.split("Here is the paraphrased document without underscores or placeholders:")[1]

    if "Here is the paraphrased document without underscores or placeholders:" in private_paraphrase:
        private_paraphrase_clean = private_paraphrase.split("Here is the paraphrased document without underscores or placeholders:")[1]

    # Build output entry (same structure as DP-FUSION rewriting)
    output_entry = {
        "passage": entry["passage"],
        "private_entities": entry["private_entities"],
    }

    if "metadata" in entry:
        output_entry["metadata"] = entry["metadata"]

    output_entry.update({
        "public_text": public_text,
        "private_text": private_text,
        "public_paraphrase": public_paraphrase_clean,
        "private_paraphrase": private_paraphrase_clean,
        "public_paraphrase_full": public_paraphrase,
        "private_paraphrase_full": private_paraphrase,
    })

    print(f"Public paraphrase: {public_paraphrase_clean[:100]}...")
    print(f"Private paraphrase: {private_paraphrase_clean[:100]}...")
    print("=" * 50)

    return output_entry

def main():
    start_time = time.time()
    log(f"Starting baseline paraphrase generation for indices {args.start} to {args.end}")

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
