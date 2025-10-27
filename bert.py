from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import os
import torch
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

load_dotenv('account.env')
login(os.getenv("HUGGINGFACE_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

print("="*100)
print("🤖 BERT EMBEDDING GENERATOR - BATCH MODE")
print("="*100)
print(f"\n✓ Device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Model: {model.config._name_or_path}")

# ============================================================================
# SETUP DIRECTORIES
# ============================================================================
input_dir = Path("../after_PreProcessed_Dataset")
output_dir = Path("../dataset_vector")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Input directory: {input_dir}")
print(f"✓ Output directory: {output_dir}")

# Find all .txt files
txt_files = list(input_dir.glob("*.txt"))
print(f"\n✓ Found {len(txt_files)} preprocessed files to process")

if len(txt_files) == 0:
    print("\n⚠ No .txt files found in input directory!")
    exit(1)

# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================
batch_size = 16  # Ukuran batch (sesuaikan dengan VRAM GPU)
print(f"\n⚙️  Batch size: {batch_size}")

# Statistics
total_files_processed = 0
total_lines_processed = 0
failed_files = []
start_time = time.time()

print("\n" + "="*100)
print("🚀 STARTING BATCH PROCESSING")
print("="*100)

# ============================================================================
# PROCESS EACH FILE
# ============================================================================
for file_idx, txt_file in enumerate(txt_files, 1):
    try:
        # Output filename
        output_filename = txt_file.stem + "_embeddings.npy"
        output_path = output_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            print(f"\n[{file_idx}/{len(txt_files)}] ⏭️  SKIPPED: {txt_file.name}")
            print(f"    (Already exists: {output_filename})")
            continue
        
        print(f"\n[{file_idx}/{len(txt_files)}] 📖 Reading: {txt_file.name}")
        
        # Read file
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            print(f"    ⚠️  SKIPPED: File is empty")
            continue
        
        print(f"    ✓ Total lines: {len(lines):,}")
        print(f"    ⚙️  Generating embeddings...")
        
        # Generate embeddings
        embeddings = []
        
        # Progress bar for this file
        with tqdm(total=len(lines), desc="    Processing", unit="lines", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate embeddings with no gradient
                with torch.no_grad():
                    outputs = model(**inputs)

                last_hidden_state = outputs.last_hidden_state

                # Mean pooling
                attention_mask = inputs['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                masked_embeddings = last_hidden_state * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy
                batch_embeddings = sentence_embeddings.detach().cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                # Update progress
                pbar.update(len(batch))
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Save to file
        np.save(output_path, embeddings_array)
        
        print(f"    ✓ Embeddings shape: {embeddings_array.shape}")
        print(f"    ✓ Saved to: {output_filename}")
        
        total_files_processed += 1
        total_lines_processed += len(lines)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user!")
        break
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        failed_files.append((txt_file.name, str(e)))
        continue

# ============================================================================
# SUMMARY
# ============================================================================
elapsed_time = time.time() - start_time

print("\n" + "="*100)
print("📊 SUMMARY")
print("="*100)
print(f"\n✓ Total files processed: {total_files_processed}/{len(txt_files)}")
print(f"✓ Total log lines embedded: {total_lines_processed:,}")
print(f"✓ Output directory: {output_dir}")
print(f"✓ Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if total_lines_processed > 0:
    print(f"✓ Average speed: {total_lines_processed/elapsed_time:.0f} lines/second")

if failed_files:
    print(f"\n⚠️  Failed files: {len(failed_files)}")
    for file, error in failed_files[:10]:
        print(f"  - {file}: {error}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files)-10} more errors")

print("\n" + "="*100)
print("✅ BERT EMBEDDING GENERATION SELESAI!")
print("="*100)
