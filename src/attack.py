import torch
import random
from tqdm import tqdm
from .loss import compute_loss_new
from .config import Config
import gc

def get_embedding_matrix(model):
    return model.get_input_embeddings().weight

def get_embeddings(model, input_ids):
    return model.get_input_embeddings()(input_ids)

def gcg_iter(input_ids, input_slice, model, tokenizer, options):
    embed_weights = get_embedding_matrix(model)
    device = model.device

    # Create one-hot representation for the tokens to perturb
    # Optimize: avoid creating full one-hot if not needed, but for now keep logic similar to ensure correctness
    vocab_size = embed_weights.shape[0]
    num_perturbed = input_ids[input_slice].shape[0]
    
    one_hot = torch.zeros(
        num_perturbed,
        vocab_size,
        device=device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        1.0
    )
    one_hot.requires_grad_(True)

    input_embeds = (one_hot @ embed_weights).unsqueeze(0) # (1, seq_len_slice, dim)
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()

    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1
    )

    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits
    one_hot_grad = one_hot.grad # This will be None until backward is called on loss
    
    # Return one_hot tensor so we can check gradients later
    return logits, one_hot

def select(one_hot_grad, current_token_id, model, tokenizer, prompt_ids, index, original_logits, option_indices, top_percent=0.5, k=10):
    """
    Select best candidate token based on gradients.
    """
    grad = one_hot_grad.squeeze(0) # (vocab_size,)
    grad[current_token_id] = float('inf') # Don't select self

    top_k_values, top_k_indices = torch.topk(-grad, k)

    num_top_candidates = max(1, int(top_percent * k))
    selected_indices = top_k_indices[:num_top_candidates]

    min_loss = float('inf')
    best_token_id = current_token_id

    for token_id in selected_indices:
        new_prompt = prompt_ids.clone()
        new_prompt[index] = token_id
        
        input_ids = new_prompt.unsqueeze(0).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        loss, _, _ = compute_loss_new(original_logits, logits, option_indices)
        
        if loss < min_loss:
            min_loss = loss
            best_token_id = token_id

    return best_token_id

def gcg_run(model, tokenizer, token_maps, iterations, options, k_cosine, pass_check, log_file=None):
    torch.cuda.empty_cache()
    gc.collect()
    device = model.device
    
    # Prepare input_ids
    input_ids = torch.tensor([token_map['token_id'] for token_map in token_maps], device=device)
    input_ids = input_ids.unsqueeze(0) # (1, seq_len)

    # Initial inference
    with torch.no_grad():
        embeddings_weight = get_embedding_matrix(model)
        print(f"DEBUG: Embedding weight shape: {embeddings_weight.shape}")
        print(f"DEBUG: Input IDs shape: {input_ids.shape}")
        print(f"DEBUG: Input IDs min/max: {input_ids.min()}, {input_ids.max()}")
        
        embeddings = get_embeddings(model, input_ids)
        outputs = model(inputs_embeds=embeddings)
        original_logits = outputs.logits

    option_indices = tokenizer.convert_tokens_to_ids(options)
    
    perturbed_already = set()

    # Main Loop
    iterator = tqdm(range(iterations), desc="GCG Iterations")
    for iteration in iterator:
        if iteration % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        batch_loss = 0
        updates = 0
        
        for index, token in enumerate(token_maps):
            if token['protected'] == 1:
                current_token_id = token['token_id']
                
                # Perturbation Strategy: Random initialization for first time
                if index not in perturbed_already:
                    current_embedding = embeddings[0, index, :]
                    embedding_matrix = get_embedding_matrix(model)
                    
                    # Compute cosine similarity
                    current_embedding_norm = current_embedding / current_embedding.norm()
                    embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
                    cosine_similarities = torch.matmul(embedding_matrix_norm, current_embedding_norm)
                    
                    # Mask special tokens and self
                    cosine_similarities[current_token_id] = -float('inf')
                    cosine_similarities[tokenizer.all_special_ids] = -float('inf')

                    top_k_values, top_k_indices = torch.topk(cosine_similarities, k_cosine)
                    index_closest = random.choice(top_k_indices.tolist())

                    token['token_id'] = index_closest
                    token['old str'] = token['token'] # Save original
                    token['token'] = tokenizer.decode([index_closest])
                    
                    input_ids[0, index] = index_closest
                    perturbed_already.add(index)

                # Gradient Step
                input_slice = slice(index, index + 1)
                model.zero_grad()
                
                logits_new, one_hot = gcg_iter(input_ids[0], input_slice, model, tokenizer, options)
                
                loss, _, _ = compute_loss_new(original_logits, logits_new, option_indices)
                
                loss.backward()
                one_hot_grad = one_hot.grad
                
                select_new_id = select(one_hot_grad, token['token_id'], model, tokenizer, input_ids[0], index, original_logits, option_indices, pass_check)
                
                if select_new_id != token['token_id']:
                    token['token_id'] = select_new_id
                    token['token'] = tokenizer.decode([select_new_id])
                    input_ids[0, index] = select_new_id
                    updates += 1
                
                batch_loss += loss.item()
                
        iterator.set_postfix({"loss": f"{batch_loss:.4f}", "updates": updates})

    return token_maps
