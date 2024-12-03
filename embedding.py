# from transformers import AutoTokenizer, AutoModel
# import torch

# # Load the tokenizer and model
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# def mean_pooling(token_embeddings, attention_mask):
#     """
#     Apply mean pooling to get a single sentence embedding.
#     """
#     mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
#     sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
#     sum_mask = torch.sum(mask, dim=1)
#     return sum_embeddings / sum_mask

# def generate_embedding(sentence):
#     """
#     Generate an embedding for a given sentence.
#     """
#     encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     return mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask']).squeeze().tolist()