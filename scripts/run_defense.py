import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.utils import read_jsonl, write_jsonl, get_highest_prob_option
from src.privacy import tokenize_with_positions, extract_secret_entities_with_adjusted_positions, get_token_positions_from_text, mark_protected_tokens
from src.attack import gcg_run

def main():
    parser = argparse.ArgumentParser(description="Run PGCG Defense")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--iterations", type=int, default=Config.DEFAULT_ITERATIONS, help="Number of GCG iterations")
    parser.add_argument("--k_cosine", type=int, default=Config.DEFAULT_K_COSINE, help="K Cosine Top Candidates")
    parser.add_argument("--pass_check", type=float, default=0.5, help="Pass check threshold")
    parser.add_argument("--total_to_do", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--model", type=str, default=Config.MODEL_ID, help="Model ID")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_index", type=int, default=None, help="End index for processing")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with a tiny random model (no download required)")
    
    args = parser.parse_args()
    
    # Validation
    Config.validate()
    
    # Load Model
    print(f"Loading model: {args.model}")
    print(f"Device: {Config.DEVICE}")
    print(f"DEBUG: Running with iterations: {args.iterations}")
    
    if args.debug:
        print("WARNING: Running in DEBUG mode with a MOCK model. Results will be meaningless junk, but the pipeline will run.")
        
        # Create a detailed Mock Model to bypass PyTorch/Environment specific crashing issues
        class MockModel(torch.nn.Module):
            def __init__(self, vocab_size=1000):
                super().__init__()
                self.vocab_size = vocab_size
                self.device = torch.device('cpu')
                # Dummy embedding layer to satisfy get_input_embeddings().weight checks
                self.embedding = torch.nn.Embedding(vocab_size, 64)
                
            def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
                # Return random logits
                if inputs_embeds is not None:
                    batch, seq, _ = inputs_embeds.shape
                elif input_ids is not None:
                    batch, seq = input_ids.shape
                else:
                    batch, seq = 1, 10
                    
                logits = torch.randn(batch, seq, self.vocab_size)
                # Return object with logits attribute
                class Output:
                    pass
                o = Output()
                o.logits = logits
                return o
                
            def get_input_embeddings(self):
                return self.embedding
                
            def eval(self):
                pass
                
            def zero_grad(self):
                pass

        model = MockModel(vocab_size=1000)
        
        # Dummy Tokenizer (Previous one was fine, just need to ensure vocab size matches)
        class DummyTokenizer:
            vocab_size = 1000
            all_special_ids = [0, 1]
            pad_token_id = 0
            eos_token_id = 1
            def __call__(self, text, **kwargs):
                tokens = text.split()
                # Ensure simple list return
                input_ids = [abs(hash(t)) % 1000 for t in tokens]
                offsets = []
                start = 0
                for t in tokens:
                    end = start + len(t)
                    offsets.append((start, end))
                    start = end + 1
                return {
                    'input_ids': input_ids, 
                    'offset_mapping': offsets, 
                    'attention_mask': [1] * len(input_ids)
                }
            
            def encode(self, text, *args, **kwargs):
                return [abs(hash(t)) % 1000 for t in text.split()]
            def convert_ids_to_tokens(self, ids):
                return [f"tok_{i}" for i in ids]
            def convert_tokens_to_ids(self, tokens):
                if isinstance(tokens, str): return abs(hash(tokens)) % 1000
                return [abs(hash(t)) % 1000 for t in tokens]
            def decode(self, ids, **kwargs):
                if isinstance(ids, int): ids = [ids]
                return " ".join([f"tok_{i}" for i in ids])
                
        tokenizer = DummyTokenizer()
            
            
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=Config.HF_TOKEN)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=Config.HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=Config.CACHE_DIR,
            quantization_config=quantization_config
        )
    
    # ...
    
    # Load Data (Handle both JSON and JSONL)
    if args.input_file.endswith(".jsonl"):
        input_data = read_jsonl(args.input_file)
    else:
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
            
    # Apply Index Slicing for Parallelism
    if args.end_index is None:
        args.end_index = len(input_data)
        
    print(f"Processing chunk: {args.start_index} to {args.end_index}")
    input_data = input_data[args.start_index : args.end_index]
            
    # Resume Logic: Check if output file exists and load it
    data = input_data
    if os.path.exists(args.output_file) and not args.debug:
        try:
            with open(args.output_file, 'r') as f:
                existing_data = json.load(f)
            
            print(f"Resuming from {args.output_file}. Loaded {len(existing_data)} items.")
            
            # Simple Resume: If existing data exists, load it into `data` (which is input_data slice) where possible
            # But primarily rely on input_data for structure.
            if len(existing_data) > 0:
                 print("Merging existing results...")
                 for i, item in enumerate(existing_data):
                     if i < len(data):
                         data[i] = item
                         
        except Exception as e:
            print(f"Could not load existing output file for resume: {e}. Starting fresh.")
            
    if args.total_to_do:
        data = data[:args.total_to_do]
        
    print(f"Processing {len(data)} items...")
    
    processed_count = 0
    correct_count = 0
    
    # Calculate initial accuracy from resumed data
    for item in data:
        if "eval_correct" in item:
            processed_count += 1
            if item["eval_correct"]:
                correct_count += 1
                
    print(f"Already processed: {processed_count}. Current Accuracy: {correct_count/processed_count:.2%}" if processed_count > 0 else "Starting fresh.")
    
    correct_count = 0
    
    for i, item in enumerate(data):
        print(f"\n--- Item {i+1}/{len(data)} ---")
        
        # Check if already processed
        if "eval_correct" in item and "my defence new full prompt" in item:
            print(f"Skipping Item {i+1} (Already processed).")
            continue
            
        passage = item.get("passage", item.get("chunk_text"))
        question = item.get("question", "What is the secret?") 
        options = item.get("options", ["A", "B", "C", "D"])
        
        if not passage:
            print(f"Skipping item {i} due to missing passage text.")
            continue
            
        # Determine correct answer
        if "answer_index" in item:
            correct_idx = item["answer_index"]
            correct_answer_char = ['A', 'B', 'C', 'D'][correct_idx]
        else:
            correct_answer_char = item.get("correct", "A")

        s_words = item.get("S", "")
        
        system_prompt = "Select the correct option based on the passage provided below. You must output one token i.e A,B,C,D that's it nothing else. Do not output any new lines."
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|> Passage: {passage} Question: {question}, Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]} <|eot_id|><|start_header_id|>assistant<|end_header_id|>The answer token is:"""
        
        # 1. Tokenize
        token_positions = tokenize_with_positions(prompt_template, tokenizer)
        
        # 2. Identify Protected Tokens
        try:
            secret_entities = extract_secret_entities_with_adjusted_positions(item, tokenizer, prompt_template)
            leakage_tokens = get_token_positions_from_text(s_words, passage, prompt_template)
            marked_tokens = mark_protected_tokens(token_positions, secret_entities, leakage_tokens)
        except Exception as e:
            print(f"Skipping identification for item {i}: {e}")
            continue

        # 3. GCG Attack (Defense Optimization)
        option_chars = ['A', 'B', 'C', 'D']
        
        try:
            updated_token_maps = gcg_run(
                model=model, 
                tokenizer=tokenizer, 
                token_maps=marked_tokens, 
                iterations=args.iterations, 
                options=option_chars, 
                k_cosine=args.k_cosine, 
                pass_check=args.pass_check
            )
        except Exception as e:
            print(f"GCG Run failed for item {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Reconstruct Prompt
        full_prompt_tokens = [tp['token_id'] for tp in updated_token_maps]
        full_text = tokenizer.decode(full_prompt_tokens)
        
        # Store Result
        item["my defence new full prompt"] = full_text
        
        # 4. Evaluate Accuracy on NEW prompt
        # We need to compute logits for the final prompt
        with torch.no_grad():
            input_tensor = torch.tensor([full_prompt_tokens]).to(Config.DEVICE)
            output = model(input_tensor)
            logits = output.logits
            
            # Get prediction (last token logits for A, B, C, D)
            last_token_logits = logits[0, -1, :]
            predicted_char, probs = get_highest_prob_option(last_token_logits, tokenizer, option_chars)
            
        is_correct = (predicted_char == correct_answer_char)
        if is_correct:
            correct_count += 1
            
        item["eval_prediction"] = predicted_char
        item["eval_correct"] = is_correct
        item["eval_probs"] = probs
        
        print(f"Prediction: {predicted_char} (Correct: {correct_answer_char}) -> {'✅' if is_correct else '❌'}")
        
        processed_count += 1
        current_acc = correct_count / processed_count
        print(f"Running Accuracy: {current_acc:.2%} ({correct_count}/{processed_count})")
        
        # Intermediate Save
        if (i+1) % 1 == 0:
            with open(args.output_file, "w") as f:
                json.dump(data, f, indent=4)
        
        # Clear memory after each item
        torch.cuda.empty_cache()
        import gc
        gc.collect()
                
    # Final Accuracy
    final_acc = correct_count / processed_count if processed_count > 0 else 0
    print(f"\nFinal Accuracy: {final_acc:.2%} ({correct_count}/{processed_count})")
    
    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Done. Processed {processed_count}/{len(data)} items. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
