import torch
import torch.nn.functional as F

def compute_loss(original_logits, new_logits, option_indices):
    """
    Computes the MSE loss between the original logits and the new logits for the specified option indices.
    
    Args:
        original_logits: Tensor of shape (batch_size, seq_len, vocab_size) or similar
        new_logits: Tensor of shape (batch_size, seq_len, vocab_size)
        option_indices: List or Tensor of indices corresponding to options (A, B, C, D)
        
    Returns:
        loss: a scalar tensor representing the MSE loss
        original_options_logits: the logits for the options from the original model output
        new_options_logits: the logits for the options from the new model output
    """
    # Assuming logits shape matches what was seen in the original script: (batch_size, seq_len, vocab_size)
    # We take the logits of the last token (prediction)
    
    # Ensure option_indices is a tensor
    if not isinstance(option_indices, torch.Tensor):
        option_indices = torch.tensor(option_indices, device=original_logits.device)
    
    original_options_logits = original_logits[:, -1, option_indices]
    new_options_logits = new_logits[:, -1, option_indices]
    
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(new_options_logits, original_options_logits)
    
    return loss, original_options_logits, new_options_logits

def compute_loss_new(original_logits, new_logits, option_indices):
    """
    Alias for compute_loss, as both names were used in the original codebase.
    """
    return compute_loss(original_logits, new_logits, option_indices)
