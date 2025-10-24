import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def log_to_vector(text: str, model_name: str = "bert-base-uncased",
                  tokenizer: AutoTokenizer = None, model: AutoModel = None,
                  device: torch.device = None) -> np.ndarray:
    """
    Konversi single raw log string -> 1D numpy vector (sentence embedding)
    - Memuat tokenizer/model jika tidak diberikan.
    - Menggunakan mean pooling atas last_hidden_state dengan attention mask.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        attention_mask = inputs["attention_mask"]  # (1, seq_len)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(torch.float32)

        masked_embeddings = last_hidden_state * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask  # (1, hidden_dim)

        return sentence_embedding.squeeze(0).cpu().numpy()