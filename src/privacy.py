import torch
from transformers import PreTrainedTokenizer

def tokenize_with_positions(text: str, tokenizer: PreTrainedTokenizer):
    """
    Tokenizes text and returns tokens with their start/end character positions.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_positions = []
    for token_id, token, offsets in zip(input_ids, tokens, offset_mapping):
        start, end = offsets
        token_positions.append({
            'token': token,
            'token_id': token_id,
            'start': start,
            'end': end
        })
    return token_positions

def extract_secret_entities_with_adjusted_positions(item: dict, tokenizer: PreTrainedTokenizer, prompt_text: str):
    """
    Extracts secret entities from the item and adjusts their positions based on the 'Passage:' marker in the prompt.
    """
    # Locate the exact position of "Passage:" within the full prompt text
    passage_marker = "Passage: "
    padding_offset = prompt_text.find(passage_marker) 
    
    if padding_offset == -1:
         # Try without space
        passage_marker = "Passage:"
        padding_offset = prompt_text.find(passage_marker)
        
    if padding_offset == -1:
        raise ValueError("'Passage:' marker not found in the prompt text.")
        
    padding_offset += len(passage_marker)

    entity_positions = []
    
    # Handle both old "secret entities" format and new "chunk_private_spans" format
    secret_entities = item.get("secret entities", item.get("chunk_private_spans", []))
    
    for entity in secret_entities:
        word = entity.get("text")
        if not word: 
            continue
            
        # Determine start/end from available keys
        if "start" in entity and "end" in entity:
            start_in_passage = entity["start"]
            end_in_passage = entity["end"]
        elif "chunk_segments" in entity:
            # chunk_segments is usually [[start, end]]
            if len(entity["chunk_segments"]) > 0:
                start_in_passage = entity["chunk_segments"][0][0]
                end_in_passage = entity["chunk_segments"][0][1]
            else:
                continue
        else:
            # Fallback if no position info (should typically not happen in this dataset)
            continue
            
        start = start_in_passage + padding_offset
        end = end_in_passage + padding_offset
        
        entity_positions.append({
            'word': word,
            'start': start,
            'end': end
        })

    return entity_positions

def get_token_positions_from_text(target_text: str, passage: str, full_prompt: str) -> list:
    """
    Finds positions of words from target_text (space separated) within the passage in the full_prompt.
    """
    passage_start_offset = full_prompt.find(passage)
    if passage_start_offset == -1:
        # Fallback: simple search in full prompt, though less safe
        passage_start_offset = 0 
        
    words = target_text.split()
    token_positions = []
    passage_lower = passage.lower()

    for word in words:
        word_lower = word.lower()
        word_start_in_passage = passage_lower.find(word_lower)
        
        if word_start_in_passage != -1:
            word_start = word_start_in_passage + passage_start_offset
            word_end = word_start + len(word)
            
            token_positions.append({
                'word': word,
                'start': word_start,
                'end': word_end,
            })

    return token_positions

def should_we_mark(range1, range2):
    start1, end1 = range1
    start2, end2 = range2
    return start1 < end2 and start2 < end1

def mark_protected_tokens(token_positions: list, secret_entities: list, leakage_tokens: list) -> list:
    """
    Marks tokens as protected if they overlap with secret entities or leakage tokens.
    """
    protected_ranges = []
    for entity in secret_entities:
        protected_ranges.append({
            'range': (entity['start'], entity['end']),
            'word': entity['word'],
            'source': 'secret_entities'
        })
    for leak in leakage_tokens:
        protected_ranges.append({
            'range': (leak['start'], leak['end']),
            'word': leak['word'],
            'source': 'leakage_tokens'
        })

    marked_tokens = []
    for token in token_positions:
        token_range = (token['start'], token['end'])
        
        protected_by = None
        for protected_range in protected_ranges:
            if should_we_mark(token_range, protected_range['range']):
                protected_by = protected_range
                break 

        marked_tokens.append({
            'token': token['token'],
            'token_id': token['token_id'],
            'start': token['start'],
            'end': token['end'],
            'protected': 1 if protected_by else 0,
            'protected_by': protected_by,
            'old str': token['token'] # Initialize old str
        })

    return marked_tokens
