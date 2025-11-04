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
import shutil

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
primary_output_dir = Path("../dataset_vector")
backup_output_dir = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector")

# Fungsi untuk cek ruang disk yang tersedia
def get_free_space_gb(path):
    """Get free disk space in GB for the given path"""
    try:
        # Buat folder jika belum ada untuk bisa cek disk space
        path.mkdir(parents=True, exist_ok=True)
        stat = shutil.disk_usage(path)
        return stat.free / (1024**3)  # Convert to GB
    except Exception as e:
        print(f"⚠️  Error checking disk space for {path}: {e}")
        return 0

# Buat kedua folder jika belum ada
primary_output_dir.mkdir(parents=True, exist_ok=True)
backup_output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Input directory: {input_dir}")
print(f"✓ Primary output: {primary_output_dir}")
print(f"✓ Backup output: {backup_output_dir}")

# Both locations to check for already processed files
check_locations = [primary_output_dir, backup_output_dir]
print(f"\n🔍 Will check for existing files in BOTH locations:")
for loc in check_locations:
    print(f"   - {loc}")

# Find all .txt files and SORT them alphabetically for consistent order
txt_files = sorted(list(input_dir.glob("*.txt")))
print(f"\n✓ Found {len(txt_files)} preprocessed files to process")

if len(txt_files) == 0:
    print("\n⚠ No .txt files found in input directory!")
    exit(1)

# Check how many already processed (check BOTH locations)
already_processed = 0
already_processed_files = {}  # Track which file is in which location

for txt_file in txt_files:
    output_filename = txt_file.stem + "_embeddings.npy"
    file_exists = False
    
    # Check in both locations
    for check_dir in check_locations:
        check_path = check_dir / output_filename
        if check_path.exists():
            already_processed += 1
            already_processed_files[output_filename] = check_path
            file_exists = True
            break  # Found in one location, no need to check others
    
if already_processed > 0:
    print(f"✓ Already processed: {already_processed} files (will be skipped)")
    print(f"✓ Remaining to process: {len(txt_files) - already_processed} files")
    print(f"   Files found in:")
    
    # Count files per location
    primary_count = sum(1 for p in already_processed_files.values() if primary_output_dir in p.parents)
    backup_count = sum(1 for p in already_processed_files.values() if backup_output_dir in p.parents)
    
    if primary_count > 0:
        print(f"   - Primary location: {primary_count} files")
    if backup_count > 0:
        print(f"   - Backup location: {backup_count} files")
else:
    print(f"✓ Starting fresh - no files processed yet")

# ============================================================================
# BATCH PROCESSING CONFIGURATION - OPTIMIZED FOR GTX 1660 6GB
# ============================================================================
# AGGRESSIVE batch size for GTX 1660 6GB
# With pre-tokenization strategy, we can push MUCH harder!
if torch.cuda.is_available():
    available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # GTX 1660 6GB: With dynamic padding, we can go 256-512!
    # Each sample ~20-30MB VRAM (depends on actual token length)
    if available_vram_gb >= 5.5:  # GTX 1660 has ~5.79GB
        batch_size = 256  # 2x increase! (was 128)
    elif available_vram_gb >= 4:
        batch_size = 192
    elif available_vram_gb >= 3:
        batch_size = 128
    else:
        batch_size = 64
else:
    batch_size = 16  # CPU fallback

# Allow manual override for experimentation
# Try 384 or 512 if you want to push even harder!
MANUAL_BATCH_SIZE = 512  # Set to 384 or 512 to experiment

# COMPARISON MODE: Toggle between pre-tokenization vs per-batch tokenization
# ⚠️  IMPORTANT: Use True for maximum speed (3-4x faster!)
# Set to False ONLY for comparison/benchmark testing
USE_PRE_TOKENIZATION = True  # True = FAST (tokenize once), False = SLOW (old method)

if MANUAL_BATCH_SIZE:
    batch_size = MANUAL_BATCH_SIZE
    print(f"\n⚙️  Batch size (MANUAL): {batch_size}")
else:
    print(f"\n⚙️  Batch size (AUTO-TUNED): {batch_size}")

print(f"✓ Tokenization mode: {'PRE-TOKENIZATION (Fast)' if USE_PRE_TOKENIZATION else 'PER-BATCH (Old/Slow method)'}")

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
        
        # Skip if already processed in ANY location
        file_exists_somewhere = False
        existing_location = None
        
        for check_dir in check_locations:
            check_path = check_dir / output_filename
            if check_path.exists():
                file_exists_somewhere = True
                existing_location = check_path
                break
        
        if file_exists_somewhere:
            file_size_mb = existing_location.stat().st_size / (1024 * 1024)
            print(f"\n[{file_idx}/{len(txt_files)}] ⏭️  SKIPPED: {txt_file.name}")
            print(f"    (Already exists at: {existing_location.parent.name}/{output_filename}, size: {file_size_mb:.2f} MB)")
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
        
        # Estimasi ukuran file output (lines * 768 * 4 bytes untuk float32)
        estimated_size_bytes = len(lines) * 768 * 4
        estimated_size_gb = estimated_size_bytes / (1024**3)
        print(f"    📊 Estimated output size: {estimated_size_gb:.2f} GB")
        
        # Tentukan output directory berdasarkan ruang yang tersedia untuk file ini
        primary_free_space = get_free_space_gb(primary_output_dir)
        
        # Tambahkan buffer 10% untuk keamanan
        required_space_gb = estimated_size_gb * 1.1
        
        if primary_free_space >= required_space_gb:
            output_dir = primary_output_dir
            print(f"    ✓ Will save to PRIMARY location (free: {primary_free_space:.2f} GB)")
        else:
            output_dir = backup_output_dir
            backup_free_space = get_free_space_gb(backup_output_dir)
            print(f"    ⚠️  PRIMARY insufficient (free: {primary_free_space:.2f} GB, need: {required_space_gb:.2f} GB)")
            print(f"    ✓ Will save to BACKUP location (free: {backup_free_space:.2f} GB)")
        
        output_path = output_dir / output_filename
        
        print(f"    ⚙️  Generating embeddings...")
        
        # Calculate batches
        num_batches = (len(lines) + batch_size - 1) // batch_size
        if torch.cuda.is_available():
            print(f"    📊 Batches: {num_batches:,} (batch_size={batch_size})")
        
        # Generate embeddings with optimizations
        embeddings_array = np.zeros((len(lines), 768), dtype=np.float32)
        current_idx = 0
        
        # Choose tokenization strategy based on mode
        if USE_PRE_TOKENIZATION:
            # ============================================================
            # METHOD 1: PRE-TOKENIZATION (FAST) - Tokenize once
            # ============================================================
            print(f"    🔧 Pre-tokenizing all lines (this may take a moment)...")
            all_inputs = tokenizer(
                lines,  # Tokenize ALL at once!
                return_tensors="pt",
                truncation=True,
                max_length=64,  # ← REDUCED from 128! Log lines are short
                padding=True,
                return_attention_mask=True
            )
            print(f"    ✓ Tokenization complete! Starting GPU processing...")
            
            # Auto batch size adjustment with OOM recovery
            current_batch_size = batch_size
            oom_retry_count = 0
            max_oom_retries = 3
            
            # Progress bar
            with tqdm(total=len(lines), desc="    Processing", unit="lines", 
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                i = 0
                while i < len(lines):
                    end_idx = min(i + current_batch_size, len(lines))
                    actual_batch_size = end_idx - i
                    
                    try:
                        # Slice pre-tokenized inputs
                        batch_inputs = {
                            k: v[i:end_idx].to(device, non_blocking=True) 
                            for k, v in all_inputs.items()
                        }
                        
                        # Generate embeddings (pure GPU work, FAST!)
                        with torch.no_grad():
                            outputs = model(**batch_inputs)
                            last_hidden_state = outputs.last_hidden_state

                            # Optimized mean pooling
                            attention_mask = batch_inputs['attention_mask']
                            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                            sentence_embeddings = sum_embeddings / sum_mask
                            
                            # Direct copy to pre-allocated array
                            embeddings_array[current_idx:current_idx+actual_batch_size] = sentence_embeddings.cpu().numpy()
                            current_idx += actual_batch_size
                        
                        # Update progress
                        pbar.update(actual_batch_size)
                        
                        # Move to next batch
                        i = end_idx
                        oom_retry_count = 0  # Reset retry counter on success
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                            # OOM Error detected!
                            if oom_retry_count >= max_oom_retries:
                                print(f"\n    ✗ OOM Error: Failed after {max_oom_retries} retries")
                                print(f"    Current batch_size: {current_batch_size}")
                                print(f"    Try setting MANUAL_BATCH_SIZE to a smaller value (e.g., 64 or 128)")
                                raise
                            
                            # Auto-reduce batch size
                            old_batch_size = current_batch_size
                            current_batch_size = max(16, current_batch_size // 2)
                            oom_retry_count += 1
                            
                            print(f"\n    ⚠️  OOM detected! Auto-reducing batch_size: {old_batch_size} → {current_batch_size}")
                            print(f"    Retry {oom_retry_count}/{max_oom_retries}...")
                            
                            # Clear VRAM
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                            # Retry this batch with smaller size (don't increment i)
                            continue
                        else:
                            # Other RuntimeError, re-raise
                            raise
            
            # Clean up pre-tokenized data
            del all_inputs
            
        else:
            # ============================================================
            # METHOD 2: PER-BATCH TOKENIZATION (OLD METHOD) - For comparison
            # ============================================================
            print(f"    ⚙️  Using per-batch tokenization (old method)...")
            
            # Auto batch size adjustment with OOM recovery
            current_batch_size = batch_size
            oom_retry_count = 0
            max_oom_retries = 3
            
            # Progress bar
            with tqdm(total=len(lines), desc="    Processing", unit="lines", 
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                i = 0
                while i < len(lines):
                    end_idx = min(i + current_batch_size, len(lines))
                    actual_batch_size = end_idx - i
                    batch = lines[i:end_idx]
                    
                    try:
                        # Tokenize THIS batch only (OLD METHOD - CPU bottleneck!)
                        batch_inputs = tokenizer(
                            batch,
                            return_tensors="pt",
                            truncation=True,
                            max_length=64,  # ← REDUCED from 128!
                            padding=True,
                            return_attention_mask=True
                        )
                        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
                        
                        # Generate embeddings
                        with torch.no_grad():
                            outputs = model(**batch_inputs)
                            last_hidden_state = outputs.last_hidden_state

                            # Optimized mean pooling
                            attention_mask = batch_inputs['attention_mask']
                            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                            sentence_embeddings = sum_embeddings / sum_mask
                            
                            # Direct copy to pre-allocated array
                            embeddings_array[current_idx:current_idx+actual_batch_size] = sentence_embeddings.cpu().numpy()
                            current_idx += actual_batch_size
                        
                        # Update progress
                        pbar.update(actual_batch_size)
                        
                        # Move to next batch
                        i = end_idx
                        oom_retry_count = 0  # Reset retry counter on success
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                            # OOM Error detected!
                            if oom_retry_count >= max_oom_retries:
                                print(f"\n    ✗ OOM Error: Failed after {max_oom_retries} retries")
                                print(f"    Current batch_size: {current_batch_size}")
                                print(f"    Try setting MANUAL_BATCH_SIZE to a smaller value")
                                raise
                            
                            # Auto-reduce batch size
                            old_batch_size = current_batch_size
                            current_batch_size = max(16, current_batch_size // 2)
                            oom_retry_count += 1
                            
                            print(f"\n    ⚠️  OOM detected! Auto-reducing batch_size: {old_batch_size} → {current_batch_size}")
                            print(f"    Retry {oom_retry_count}/{max_oom_retries}...")
                            
                            # Clear VRAM
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                            # Retry this batch with smaller size (don't increment i)
                            continue
                        else:
                            # Other RuntimeError, re-raise
                            raise
        
        # Trim array to actual size
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
print(f"✓ Output directories:")
print(f"   - Primary: {primary_output_dir}")
print(f"   - Backup: {backup_output_dir}")
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
