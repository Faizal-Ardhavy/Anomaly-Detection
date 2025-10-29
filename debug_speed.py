"""
DEBUG SCRIPT - Identify bottleneck in BERT processing
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
from pathlib import Path

print("="*80)
print("🔍 BERT SPEED DIAGNOSTIC")
print("="*80)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ Device: {device}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("✗ CUDA not available! Running on CPU (VERY SLOW)")
    exit(1)

# Load model
print("\n📦 Loading BERT model...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()
print(f"✓ Model loaded in {time.time()-start:.2f}s")

# Test data - simulate log lines
print("\n🧪 Creating test data...")
test_lines = [
    "error connection timeout after 5000ms",
    "user admin logged in successfully",
    "warning memory usage high 85 percent",
    "info service started on port 8080",
    "debug processing request id 12345"
] * 100  # 500 lines

batch_sizes = [256, 384, 512]

print(f"✓ Test dataset: {len(test_lines)} lines")

# ============================================================================
# TEST 1: Pre-tokenization speed
# ============================================================================
print("\n" + "="*80)
print("TEST 1: TOKENIZATION SPEED")
print("="*80)

start = time.time()
all_inputs = tokenizer(
    test_lines,
    return_tensors="pt",
    truncation=True,
    max_length=128,
    padding=True,
    return_attention_mask=True
)
tokenize_time = time.time() - start

print(f"\n✓ Tokenized {len(test_lines)} lines in {tokenize_time:.4f}s")
print(f"✓ Speed: {len(test_lines)/tokenize_time:.0f} lines/sec")
print(f"✓ Time per line: {tokenize_time/len(test_lines)*1000:.2f}ms")

if tokenize_time > 5:
    print("⚠️  WARNING: Tokenization is VERY SLOW!")
    print("   Expected: <1 second for 500 lines")
    print("   Possible issues: CPU bottleneck, slow disk I/O")

# ============================================================================
# TEST 2: GPU Inference speed (different batch sizes)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: GPU INFERENCE SPEED")
print("="*80)

for batch_size in batch_sizes:
    print(f"\n🔬 Testing batch_size={batch_size}:")
    
    # Warm up GPU
    with torch.no_grad():
        batch_inputs = {k: v[:batch_size].to(device) for k, v in all_inputs.items()}
        _ = model(**batch_inputs)
    
    torch.cuda.synchronize()
    
    # Actual test
    num_batches = min(5, len(test_lines) // batch_size)
    times = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(test_lines))
        
        batch_inputs = {
            k: v[start_idx:end_idx].to(device, non_blocking=True)
            for k, v in all_inputs.items()
        }
        
        start = time.time()
        with torch.no_grad():
            outputs = model(**batch_inputs)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling
            attention_mask = batch_inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            # Transfer to CPU
            embeddings = sentence_embeddings.cpu().numpy()
        
        torch.cuda.synchronize()
        batch_time = time.time() - start
        times.append(batch_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  ✓ Average time per batch: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  ✓ Speed: {batch_size/avg_time:.0f} lines/sec")
    print(f"  ✓ Time per line: {avg_time/batch_size*1000:.2f}ms")
    
    if avg_time > 1.0:
        print(f"  ⚠️  WARNING: This is TOO SLOW!")
        print(f"     Expected: ~0.05-0.2s per batch")
        print(f"     Your actual: {avg_time:.2f}s per batch")
        print(f"     Slowdown: {avg_time/0.1:.1f}x slower than expected!")

# ============================================================================
# TEST 3: Full pipeline (like actual bert.py)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: FULL PIPELINE TEST (Pre-tokenization method)")
print("="*80)

batch_size = 384  # Optimal size

# Simulate actual processing
print(f"\n🔬 Processing {len(test_lines)} lines with batch_size={batch_size}")

# Step 1: Pre-tokenize (already done above)
print(f"✓ Pre-tokenization: {tokenize_time:.2f}s")

# Step 2: GPU processing
start = time.time()
embeddings_array = np.zeros((len(test_lines), 768), dtype=np.float32)
current_idx = 0

for i in range(0, len(test_lines), batch_size):
    end_idx = min(i + batch_size, len(test_lines))
    
    batch_inputs = {
        k: v[i:end_idx].to(device, non_blocking=True)
        for k, v in all_inputs.items()
    }
    
    with torch.no_grad():
        outputs = model(**batch_inputs)
        last_hidden_state = outputs.last_hidden_state
        
        attention_mask = batch_inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        
        actual_batch_size = end_idx - i
        embeddings_array[current_idx:current_idx+actual_batch_size] = sentence_embeddings.cpu().numpy()
        current_idx += actual_batch_size

torch.cuda.synchronize()
gpu_time = time.time() - start

total_time = tokenize_time + gpu_time

print(f"✓ GPU processing: {gpu_time:.2f}s")
print(f"✓ Total time: {total_time:.2f}s")
print(f"✓ Overall speed: {len(test_lines)/total_time:.0f} lines/sec")

print("\n" + "="*80)
print("📊 EXPECTED vs ACTUAL")
print("="*80)

expected_speed = 10000  # lines/sec for GTX 1660
actual_speed = len(test_lines) / total_time

print(f"\nExpected speed: ~{expected_speed:,} lines/sec")
print(f"Your actual speed: {actual_speed:.0f} lines/sec")

if actual_speed < expected_speed / 10:
    print(f"\n❌ CRITICAL: You are {expected_speed/actual_speed:.0f}x SLOWER than expected!")
    print("\nPossible causes:")
    print("1. ❌ GPU not being used (running on CPU)")
    print("2. ❌ Slow CPU (tokenization bottleneck)")
    print("3. ❌ Network/disk I/O issues (server storage)")
    print("4. ❌ CPU thermal throttling")
    print("5. ❌ Memory swapping (not enough RAM)")
    print("\nRun this on your server and share the output!")
elif actual_speed < expected_speed / 2:
    print(f"\n⚠️  WARNING: You are {expected_speed/actual_speed:.1f}x slower than expected")
    print("\nPossible optimizations needed")
else:
    print(f"\n✅ GOOD: Performance is acceptable!")
    print(f"   Only {expected_speed/actual_speed:.1f}x slower than optimal")

# ============================================================================
# TEST 4: Check if GPU is actually being used
# ============================================================================
print("\n" + "="*80)
print("TEST 4: GPU UTILIZATION CHECK")
print("="*80)

print("\n🔥 Running intensive GPU workload for 5 seconds...")
print("   CHECK YOUR GPU MONITOR NOW!")
print("   Power should spike to ~130W if GPU is working")

x = torch.randn(4096, 4096).to(device)
start = time.time()
while time.time() - start < 5:
    y = torch.matmul(x, x)
    y = torch.matmul(y, y)
torch.cuda.synchronize()

print("✓ Workload complete!")
print("\n   If power stayed at 50-70W, GPU is NOT being used properly!")
print("   If power spiked to 120-130W, GPU is working correctly!")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE!")
print("="*80)
print("\nPlease share the output of this script for analysis.")
