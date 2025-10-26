from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import os
import torch
from dotenv import load_dotenv

load_dotenv('account.env')
login(os.getenv("HUGGINGFACE_TOKEN"))


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

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

log_file_path = r".\combined_logs_preprocessed.txt"
with open(log_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines if line.strip()]
embeddings = []
print(f"Total log lines to process: {len(lines)}")
print("Device used:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("Model used:", model.config._name_or_path)

# ============================================================================
# OPTIMASI: Batch Processing + torch.no_grad()
# ============================================================================
batch_size = 16  # Ukuran batch (sesuaikan dengan VRAM GPU)
print(f"\n🚀 Processing dengan batch size: {batch_size}")
print("="*80)

for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    
    # Tokenize batch sekaligus (dengan padding)
    inputs = tokenizer(
        batch, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True  # Padding untuk batch processing
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Gunakan torch.no_grad() untuk menghemat memori (tidak perlu gradient)
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)

    # Mean pooling untuk setiap item dalam batch
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)  # (batch_size, 768)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask  # (batch_size, 768)
    
    # Convert to numpy dan tambahkan ke list
    batch_embeddings = sentence_embeddings.detach().cpu().numpy()
    embeddings.extend(batch_embeddings)
    
    # Progress indicator
    if (i // batch_size) % 10 == 0:
        progress = min(i + batch_size, len(lines))
        print(f"✓ Processed {progress}/{len(lines)} logs ({progress/len(lines)*100:.1f}%)")

import numpy as np

print("\n" + "="*80)
print(f"✓ SELESAI! Total embeddings: {len(embeddings)}")
print(f"✓ Shape embeddings: ({len(embeddings)}, 768)")
print("="*80)

np.save("combined_embeddings.npy", embeddings)
print(f"✓ Embeddings disimpan ke: combined_embeddings.npy")
