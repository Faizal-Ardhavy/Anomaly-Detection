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
# Automatic batch size calculation based on available VRAM
if torch.cuda.is_available():
    available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # GTX 1660 6GB: Can handle batch_size 32-64 for BERT-base
    # Rule of thumb: ~100MB per sample for BERT-base
    if available_vram_gb >= 6:
        batch_size = 64  # Maksimal untuk 6GB VRAM
    elif available_vram_gb >= 4:
        batch_size = 32
    else:
        batch_size = 16
else:
    batch_size = 8  # CPU fallback

# Allow manual override
MANUAL_BATCH_SIZE = None  # Set to number to override, or None for auto
if MANUAL_BATCH_SIZE:
    batch_size = MANUAL_BATCH_SIZE
    print(f"\n⚙️  Batch size (MANUAL): {batch_size}")
else:
    print(f"\n⚙️  Batch size (AUTO-TUNED): {batch_size}")

# Pin memory for faster CPU->GPU transfer
pin_memory = torch.cuda.is_available()

# Use mixed precision for 2x speedup (requires GPU with Tensor Cores)
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
if use_amp:
    print(f"✓ Mixed Precision (AMP): ENABLED (2x faster)")
else:
    print(f"✓ Mixed Precision (AMP): DISABLED (GPU too old or CPU mode)")

# Prefetch factor for data loading
prefetch_factor = 4  # Load 4 batches ahead

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
        
        # Estimate VRAM usage
        estimated_vram_gb = (len(lines) / batch_size) * 0.1  # ~100MB per batch
        if torch.cuda.is_available():
            print(f"    📊 Estimated VRAM usage: {estimated_vram_gb:.2f} GB")
        
        # Generate embeddings with optimizations
        embeddings = []
        
        # Initialize AMP scaler if using mixed precision
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # Progress bar for this file
        with tqdm(total=len(lines), desc="    Processing", unit="lines", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                
                # Tokenize batch with optimized settings
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128,  # Reduced from 512 - log lines rarely need 512 tokens
                    padding='max_length',  # Consistent tensor size for better GPU utilization
                    return_attention_mask=True
                )
                
                # Pin memory for faster transfer
                if pin_memory:
                    inputs = {k: v.pin_memory().to(device, non_blocking=True) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate embeddings with mixed precision (if supported)
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)

                last_hidden_state = outputs.last_hidden_state

                # Optimized mean pooling
                attention_mask = inputs['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy efficiently
                batch_embeddings = sentence_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                # Update progress
                pbar.update(len(batch))
                
                # Clear cache periodically to prevent VRAM fragmentation
                if i % (batch_size * 100) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Convert to numpy array efficiently
        embeddings_array = np.array(embeddings, dtype=np.float32)  # Use float32 instead of float64
        
        # Clear memory
        del embeddings, lines
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
