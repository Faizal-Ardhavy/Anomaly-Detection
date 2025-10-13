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

import numpy as np


def process_log_file(log_file_path, output_name=None):
    """
    Process a log file and save embeddings to a .npy file.
    
    Args:
        log_file_path: Path to the log file
        output_name: Name for output .npy file (without extension)
                     If None, derives from input filename
    """
    if output_name is None:
        # Extract filename without extension
        output_name = os.path.splitext(os.path.basename(log_file_path))[0] + "_embeddings"
    
    print(f"\nProcessing: {log_file_path}")
    
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
        return None
    
    lines = [line.strip() for line in lines if line.strip()]
    embeddings = []
    print(f"Total log lines to process: {len(lines)}")
    
    for i, text in enumerate(lines):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(lines)} lines")
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_embeddings = last_hidden_state * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask
            embeddings.append(sentence_embedding.squeeze(0).detach().numpy())
    
    output_path = f"{output_name}.npy"
    np.save(output_path, embeddings)
    print(f"Saved embeddings to: {output_path}")
    print(f"Shape: {np.array(embeddings).shape}")
    
    return output_path


# Main execution
if __name__ == "__main__":
    # List of log files to process
    # You can add more log files to this list
    log_files = [
        r"..\dataset\Apache.log",
        # Add more log files here, for example:
        # r"..\dataset\Nginx.log",
        # r"..\dataset\System.log",
    ]
    
    # Process each log file
    for log_file in log_files:
        if os.path.exists(log_file):
            process_log_file(log_file)
        else:
            print(f"Warning: Log file not found: {log_file}")
