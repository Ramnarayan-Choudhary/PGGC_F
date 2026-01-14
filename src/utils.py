import json
from pathlib import Path
from typing import List, Dict, Sequence

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def read_jsonl(path: str | Path, limit: int | None = None) -> List[Dict[str, object]]:
    path = Path(path)
    samples: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples

def write_jsonl(path: str | Path, data: List[Dict[str, object]]):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item) + "\n")

import torch

def get_highest_prob_option(prob_tensor, tokenizer, options):
    """
    Given a tensor of logits/probs for the last token, find the option (A,B,C,D) with highest prob.
    """
    option_ids = tokenizer.convert_tokens_to_ids(options)
    # Extract logits for just the option tokens
    # prob_tensor is expected to be [vocab_size]
    option_logits = prob_tensor[option_ids]
    
    best_idx = torch.argmax(option_logits).item()
    best_option = options[best_idx]
    
    # Return best option and a dict of probs/logits for debugging
    return best_option, {opt: logit.item() for opt, logit in zip(options, option_logits)}
