# import torch
# from transformers import AutoTokenizer, AutoModel
# from preprocess import preprocess_text

# # Load pre-trained model and tokenizer
# MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# def mean_pooling(token_embeddings, attention_mask):
#     """
#     Perform mean pooling to create a single sentence embedding.
#     - token_embeddings: Output from the transformer model.
#     - attention_mask: Mask to ignore padding tokens.
#     """
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#     return sum_embeddings / sum_mask

# def encode_sentence(sentence):
#     """
#     Tokenize, encode, and generate embeddings for a sentence.
#     """
#     # Step 1: Preprocess the input sentence
#     sentence = preprocess_text(sentence)
    
#     # Step 2: Tokenize the sentence
#     encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    
#     # Step 3: Pass through the transformer model
#     with torch.no_grad():
#         model_output = model(**encoded_input)
    
#     # Step 4: Perform mean pooling
#     embedding = mean_pooling(model_output.last_hidden_state, encoded_input["attention_mask"])
    
#     # Step 5: Normalize the embedding
#     embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
#     return embedding.squeeze(0).tolist()