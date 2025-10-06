from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import os
import torch
from dotenv import load_dotenv

load_dotenv('account.env')
login(os.getenv("HUGGINGFACE_TOKEN"))


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# text = "User login failed due to wrong password"
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)

# # Mean pooling (ignoring padding)
# attention_mask = inputs['attention_mask']
# mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
# masked_embeddings = last_hidden_state * mask
# sum_embeddings = torch.sum(masked_embeddings, dim=1)
# sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
# sentence_embedding = sum_embeddings / sum_mask  # (1, 768)
# print(sentence_embedding)

log_file_path = r"..\dataset\Apache.log"
with open(log_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines if line.strip()]
embeddings = []
print(f"Total log lines to process: {len(lines)}")
for text in lines:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask
    embeddings.append(sentence_embedding.squeeze(0).detach().numpy())

import numpy as np
np.save("apache_embeddings.npy", embeddings)
