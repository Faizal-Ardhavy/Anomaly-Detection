# BGL Log Preprocessing Plan for BERT Anomaly Detection

## 📊 **Original Format Analysis**

### **BGL Log Structure:**
```
[UNIX_TS] [DATE] [NODE] [TIMESTAMP] [NODE] [COMPONENT] [SUBSYSTEM] [LEVEL] [MESSAGE]

Example:
1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

### **Field Analysis:**

| Field | Example | Keep? | Reason |
|-------|---------|-------|--------|
| UNIX Timestamp | `1117838570` | ❌ No | Tidak semantic, redundant |
| Date | `2005.06.03` | ❌ No | Tidak semantic, redundant |
| Node ID 1 | `R02-M1-N0-C:J12-U11` | ✅ Normalize | Hardware context → `<NODE>` |
| Timestamp | `2005-06-03-15.42.50.363779` | ❌ No | Tidak semantic |
| Node ID 2 | `R02-M1-N0-C:J12-U11` | ❌ No | Duplicate |
| Component | `RAS` | ✅ **KEEP** | Semantic: system component |
| Subsystem | `KERNEL` | ✅ **KEEP** | Semantic: subsystem type |
| Level | `INFO` | ✅ **KEEP** | **CRITICAL** for anomaly |
| Message | `instruction cache...` | ✅ **KEEP** | Core semantic content |

---

## 🎯 **Preprocessing Strategy**

### **Step 1: Field Extraction**
```python
# Parse fixed-width fields (9 fields)
parts = line.split(maxsplit=8)

component = parts[5]   # RAS, KERNSOCK, etc.
subsystem = parts[6]   # KERNEL, APP, etc.
level = parts[7]       # FATAL, ERROR, WARNING, INFO
message = parts[8]     # Free text
```

### **Step 2: Log Level Prioritization**
```python
# Anomaly detection priority
CRITICAL_LEVELS = ['FATAL', 'SEVERE', 'FAILURE']  # High priority
WARNING_LEVELS = ['ERROR', 'WARNING']              # Medium priority
NORMAL_LEVELS = ['INFO', 'DEBUG']                  # Low priority

# Keep level in text for BERT semantic understanding
```

### **Step 3: Message Normalization**

#### **3.1 Variable Abstraction:**
```python
# Numbers (integers, floats, hex)
"corrected 123 times" → "corrected <NUM> times"
"0x1a2b3c4d" → "<HEX>"
"error code 404" → "error code <NUM>"

# Node/Hardware IDs
"R02-M1-N0-C:J12-U11" → "<NODE>"
"R34-M0-NC-I:J18-U11" → "<NODE>"

# Timestamps (any format)
"2005-06-03-15.42.50.363779" → (remove)
"15:42:50" → (remove)

# IP Addresses
"192.168.1.100" → "<IP>"
"10.0.0.1:8080" → "<IP>"

# File Paths
"/var/log/kern.log" → "<PATH>"
"/usr/bin/python" → "<PATH>"

# URLs
"http://example.com/api" → "<URL>"
```

#### **3.2 Text Cleaning:**
```python
# Lowercase (for consistency)
"INSTRUCTION CACHE" → "instruction cache"

# Remove special characters (keep only alphanumeric, spaces, <>)
"error: failed!" → "error failed"

# Collapse multiple spaces
"error    failed" → "error failed"

# Remove leading/trailing whitespace
```

#### **3.3 Keep Domain-Specific Terms:**
```python
# DON'T normalize these - semantic meaning!
- "cache", "parity", "corrected", "failure"
- "socket", "closed", "communication"
- "kernel", "memory", "processor"
```

---

## 🔄 **Processing Pipeline**

### **Input (Raw BGL):**
```
1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

### **Step-by-Step Transformation:**

**Step 1: Extract relevant fields**
```
Component: RAS
Subsystem: KERNEL
Level: INFO
Message: instruction cache parity error corrected
```

**Step 2: Combine into preprocessed format**
```
RAS KERNEL INFO instruction cache parity error corrected
```

**Step 3: Normalize message**
```
ras kernel info instruction cache parity error corrected
```

### **Output (Preprocessed):**
```
ras kernel info instruction cache parity error corrected
```

---

## 📏 **Expected Token Length**

### **Analysis:**
```python
# Original line: ~130 characters (9 fields)
Original tokens: ~25-30 tokens

# After preprocessing: ~60 characters (4 fields)
Preprocessed tokens: ~10-15 tokens

# Reduction: ~50% fewer tokens
# Benefits:
# - Faster BERT processing (less padding)
# - Better semantic focus
# - Reduced noise
```

### **Optimal BERT Config:**
```python
max_length = 32  # BGL logs are short after preprocessing
batch_size = 512  # Can go higher with shorter sequences
```

---

## 🎨 **Example Transformations**

### **Example 1: INFO Log**
```
Input:
1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected

Output:
ras kernel info instruction cache parity error corrected
```

### **Example 2: FATAL Log**
```
Input:
1136390405 2006.01.04 R34-M0-NC-I:J18-U11 2006-01-04-08.00.05.233639 R34-M0-NC-I:J18-U11 KERNSOCK KERNEL FATAL idoproxy communication failure socket closed

Output:
kernsock kernel fatal idoproxy communication failure socket closed
```

### **Example 3: With Numbers**
```
Input:
1117838580 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.43.00.132832 R02-M1-N0-C:J12-U11 RAS KERNEL WARNING memory error detected at address 0x1a2b3c threshold 95 percent

Output:
ras kernel warning memory error detected at address <HEX> threshold <NUM> percent
```

---

## ✅ **Why This Approach Works for BERT:**

### **1. Semantic Preservation:**
- ✅ Component type (RAS, KERNSOCK) → System context
- ✅ Subsystem (KERNEL, APP) → Functional area
- ✅ Log level (FATAL, ERROR) → **Anomaly indicator**
- ✅ Message content → Core semantic meaning

### **2. Noise Reduction:**
- ❌ Timestamps → Tidak semantic value
- ❌ Node IDs → Normalized to `<NODE>`
- ❌ Specific numbers → Normalized to `<NUM>`
- Result: Focus on patterns, not specific values

### **3. BERT Optimization:**
```python
# Short sequences = faster processing
Original: 25-30 tokens → Preprocessed: 10-15 tokens
Speedup: ~2x faster inference
Batch size: Can increase from 256 → 512+
```

### **4. Anomaly Detection:**
```python
# Similar normal patterns cluster together:
"ras kernel info instruction cache parity error corrected"
"ras kernel info instruction cache parity error corrected"
"ras kernel info instruction cache parity error corrected"

# Anomalies stand out:
"kernsock kernel fatal idoproxy communication failure socket closed"
                  ^^^^^ Different level + different message
```

---

## 🔨 **Implementation Priority**

### **Must Have (Critical):**
1. ✅ Extract Component, Subsystem, Level, Message
2. ✅ Lowercase normalization
3. ✅ Remove timestamps and node IDs
4. ✅ Normalize numbers to `<NUM>`
5. ✅ Remove special characters
6. ✅ Collapse whitespace

### **Should Have (Important):**
1. ✅ Normalize hex values to `<HEX>`
2. ✅ Normalize IP addresses to `<IP>`
3. ✅ Normalize file paths to `<PATH>`
4. ✅ Remove duplicate entries

### **Nice to Have (Optional):**
1. ⭐ Normalize URLs to `<URL>`
2. ⭐ Normalize email to `<EMAIL>`
3. ⭐ Domain-specific term dictionary

---

## 📊 **Expected Results**

### **Dataset Statistics (Estimated):**
```
Original BGL.log: 4.7M lines, ~600MB
After preprocessing: 4.5M lines, ~300MB (50% reduction)

Token distribution:
- Average: 12 tokens per line
- Max: 25 tokens per line
- Min: 5 tokens per line

Optimal BERT config:
- max_length: 32
- batch_size: 512
- Expected speed: 3,000-4,000 lines/sec
```

### **Quality Metrics:**
```python
# Duplicate reduction: ~5-10%
Original: 4.7M lines
After dedup: 4.2-4.5M lines

# Semantic preservation: ~95%+
# (All critical information retained)
```

---

## 🚀 **Next Steps**

1. **Implement preprocessing script** based on this plan
2. **Test on sample data** (first 1000 lines)
3. **Validate output quality**
4. **Benchmark BERT speed** with different max_length
5. **Run full dataset preprocessing**
6. **Generate BERT embeddings**
7. **Train anomaly detection model**
