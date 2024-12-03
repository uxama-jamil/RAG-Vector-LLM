import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling to get a single vector representation for a sentence.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def normalize_embeddings(embedding):
    """
    Normalize embeddings to unit length.
    """
    return F.normalize(embedding, p=2, dim=1)