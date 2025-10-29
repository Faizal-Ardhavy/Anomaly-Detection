from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import os
import torch
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import gc

load_dotenv('account.env')
login(os.getenv("HUGGINGFACE_TOKEN"))

# ============================================================================
# OPTIMIZED GPU SETTINGS
# ============================================================================
# Enable TF32 for faster computation on Ampere+ GPUs (minor precision loss, huge speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN auto-tuner for optimal conv algorithms
torch.backends.cudnn.benchmark = True

# Set optimal threading for CPU operations
torch.set_num_threads(12)  # i7-8700 has 12 threads

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*100)
print("🤖 BERT EMBEDDING GENERATOR - ULTRA OPTIMIZED MODE")
print("="*100)

# Load model with optimizations
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Set model to eval mode and enable inference optimizations
model.eval()

# Enable gradient checkpointing if needed (saves VRAM at cost of slight speed)
# model.gradient_checkpointing_enable()  # Uncomment if OOM errors

print(f"\n✓ Device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"✓ TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"✓ CPU Threads: {torch.get_num_threads()}")
print(f"✓ Model: {model.config._name_or_path}")

# ============================================================================
# SETUP DIRECTORIES
# ============================================================================
input_dir = Path("../after_PreProcessed_Dataset")
output_dir = Path("../dataset_vector")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Input directory: {input_dir}")
print(f"✓ Output directory: {output_dir}")

# Find all .txt files and SORT them alphabetically for consistent order
txt_files = sorted(list(input_dir.glob("*.txt")))
print(f"\n✓ Found {len(txt_files)} preprocessed files to process")

if len(txt_files) == 0:
    print("\n⚠ No .txt files found in input directory!")
    exit(1)

# Check how many already processed (for resume info)
already_processed = 0
for txt_file in txt_files:
    output_filename = txt_file.stem + "_embeddings.npy"
    output_path = output_dir / output_filename
    if output_path.exists():
        already_processed += 1

if already_processed > 0:
    print(f"✓ Already processed: {already_processed} files (will be skipped)")
    print(f"✓ Remaining to process: {len(txt_files) - already_processed} files")
else:
    print(f"✓ Starting fresh - no files processed yet")

# ============================================================================
# BATCH PROCESSING CONFIGURATION - OPTIMIZED FOR GTX 1660 6GB
# ============================================================================
# Aggressive batch size for GTX 1660 6GB
if torch.cuda.is_available():
    available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # GTX 1660: Push to max with dynamic padding
    if available_vram_gb >= 5.5:  # GTX 1660 has ~5.79GB
        batch_size = 128  # Increased from 96
    elif available_vram_gb >= 4:
        batch_size = 96
    elif available_vram_gb >= 3:
        batch_size = 64
    else:
        batch_size = 32
else:
    batch_size = 8  # CPU fallback

# Allow manual override
MANUAL_BATCH_SIZE = None  # Set to number to override, or None for auto
if MANUAL_BATCH_SIZE:
    batch_size = MANUAL_BATCH_SIZE
    print(f"\n⚙️  Batch size (MANUAL): {batch_size}")
else:
    print(f"\n⚙️  Batch size (AUTO-TUNED): {batch_size}")

# Disable AMP for GTX 1660 - overhead > benefit
use_amp = False
print(f"✓ Mixed Precision (AMP): DISABLED (GTX 1660 faster without AMP)")

# DataLoader settings for faster CPU->GPU pipeline
num_workers = 4  # Parallel data loading
pin_memory = True  # Faster CPU->GPU transfer
persistent_workers = True  # Keep workers alive

print(f"✓ DataLoader workers: {num_workers}")
print(f"✓ Pin memory: {pin_memory}")

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
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n[{file_idx}/{len(txt_files)}] ⏭️  SKIPPED: {txt_file.name}")
            print(f"    (Already exists: {output_filename}, size: {file_size_mb:.2f} MB)")
            total_files_processed += 1  # Count as processed
            continue
        
        print(f"\n[{file_idx}/{len(txt_files)}] 📖 Reading: {txt_file.name}")
        
        # Get file size info
        file_size_mb = txt_file.stat().st_size / (1024 * 1024)
        print(f"    File size: {file_size_mb:.2f} MB")
        
        # Read file in chunks for large files (memory efficient)
        if file_size_mb > 500:  # For files > 500MB
            print(f"    📖 Reading in streaming mode (large file)...")
            lines = []
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        else:
            # Read file normally
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            print(f"    ⚠️  SKIPPED: File is empty")
            continue
        
        print(f"    ✓ Total lines: {len(lines):,}")
        print(f"    ⚙️  Generating embeddings...")
        
        # Calculate batches
        num_batches = (len(lines) + batch_size - 1) // batch_size
        if torch.cuda.is_available():
            print(f"    📊 Batches: {num_batches:,} (batch_size={batch_size})")
        
        # Generate embeddings with optimizations
        embeddings = []
        
        # Pre-allocate numpy array for better performance
        embeddings_array = np.zeros((len(lines), 768), dtype=np.float32)
        current_idx = 0
        
        # Progress bar for this file
        with tqdm(total=len(lines), desc="    Processing", unit="lines", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                actual_batch_size = len(batch)
                
                # Tokenize batch with DYNAMIC padding (faster!)
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128,
                    padding=True,  # Dynamic padding to longest in batch (not max_length!)
                    return_attention_mask=True
                )
                
                # Move to GPU with non-blocking transfer
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    last_hidden_state = outputs.last_hidden_state

                    # Optimized mean pooling
                    attention_mask = inputs['attention_mask']
                    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    sentence_embeddings = sum_embeddings / sum_mask
                    
                    # Direct copy to pre-allocated array (faster than extend/append)
                    embeddings_array[current_idx:current_idx+actual_batch_size] = sentence_embeddings.cpu().numpy()
                    current_idx += actual_batch_size
                
                # Update progress
                pbar.update(actual_batch_size)
                
                # Clear cache every 1000 batches
                if i % (batch_size * 1000) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Trim array to actual size (in case of early exit)
        embeddings_array = embeddings_array[:current_idx]
        
        # Clear memory
        del lines
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save to file with compression
        print(f"    💾 Saving embeddings...")
        np.save(output_path, embeddings_array)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"    ✓ Embeddings shape: {embeddings_array.shape}")
        print(f"    ✓ File size: {file_size_mb:.2f} MB")
        print(f"    ✓ Saved to: {output_filename}")
        
        total_files_processed += 1
        total_lines_processed += len(embeddings_array)
        
        # Show VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"    📊 VRAM: {vram_used:.2f}GB used, {vram_reserved:.2f}GB reserved")
        
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
skipped_count = already_processed

print("\n" + "="*100)
print("📊 SUMMARY")
print("="*100)
print(f"\n✓ Total files in dataset: {len(txt_files)}")
print(f"✓ Files processed this session: {total_files_processed - skipped_count}")
print(f"✓ Files skipped (already done): {skipped_count}")
print(f"✓ Total completed: {total_files_processed}/{len(txt_files)}")
print(f"✓ Total log lines embedded: {total_lines_processed:,}")
print(f"✓ Output directory: {output_dir}")
print(f"✓ Time elapsed this session: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if total_lines_processed > 0:
    print(f"✓ Average speed: {total_lines_processed/elapsed_time:.0f} lines/second")

# Show progress percentage
progress_pct = (total_files_processed / len(txt_files)) * 100
print(f"\n📈 Overall Progress: {progress_pct:.1f}% complete")
remaining = len(txt_files) - total_files_processed
if remaining > 0:
    print(f"⏳ Remaining files: {remaining}")
    if total_lines_processed > 0:
        est_time_min = (remaining * elapsed_time / max(total_files_processed - skipped_count, 1)) / 60
        print(f"⏱️  Estimated time to finish: {est_time_min:.1f} minutes ({est_time_min/60:.1f} hours)")

if failed_files:
    print(f"\n⚠️  Failed files: {len(failed_files)}")
    for file, error in failed_files[:10]:
        print(f"  - {file}: {error}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files)-10} more errors")

print("\n" + "="*100)
print("✅ BERT EMBEDDING GENERATION SELESAI!")
print("="*100)
