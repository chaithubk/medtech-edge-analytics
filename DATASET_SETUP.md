# Sepsis Detection Dataset Setup Guide

## Quick Start: Use PhysioNet Challenge 2019 (Recommended)

### Step 1: Download PhysioNet Challenge 2019 Data

1. Go to: https://physionet.org/content/challenge-2019/1.0.0/
2. Click "Download" (requires free PhysioNet account)
3. Extract the `training_data/` directory
4. Copy CSV files to your project:

```bash
mkdir -p data/raw_physionet
cp /path/to/challenge-2019/training_data/*.csv data/raw_physionet/
```

**Expected output:** ~40k CSV files named `p000001.csv`, `p000002.csv`, etc.

### Step 2: Verify Data

```bash
ls -lh data/raw_physionet/ | head -5
# Should show files like: p000001.csv, p000002.csv, ...

wc -l data/raw_physionet/*.csv | head -5
# Each patient has multiple time-series rows
```

### Step 3: Test Data Loading

```bash
python3 src/load_physionet_data.py
```

**Expected output:**
```
✓ Found X patient(s) with sepsis:
  - p000001
  - p000045
  ...

Dataset statistics:
  Sepsis cases: ~2000
  Healthy cases: ~38000
  Class balance: 5.2% positive
```

### Step 4: Train Model

```bash
python3 src/train_and_convert.py
```

The pipeline will automatically use PhysioNet data if available.

---

## Alternative: Use Synthea (Fallback)

If you don't want to download PhysioNet, generate Synthea data with more patients:

```bash
# Generate 5000-10000 patients (increases sepsis likelihood)
SYNTHEA_PATIENTS=5000 SYNTHEA_SEED=42 bash scripts/generate_synthea_data.sh

# Verify sepsis cases
python3 src/flatten_fhir.py
```

---

## Dataset Comparison

| Feature | PhysioNet 2019 | Synthea | MIMIC-IV |
|---------|---|---|---|
| **Sepsis Cases** | ~2,000 | ~10-50* | ~15,000 |
| **Patients** | ~40,000 | 2,500-10,000* | ~100,000+ |
| **Class Balance** | 5-6% | <1-2%* | ~15% |
| **Access** | Public | Generated | Credentialed |
| **Setup Time** | 5 min | 30 min | 1-2 days |
| **Realism** | Very High | Medium | Highest |

*Depends on seed and patient count

---

## PhysioNet Data Format

Each patient has a CSV with columns like:
```
Hour,HR,O2Sat,SBP,DBP,Temp,RR,WBC,Lactate,Creatinine,SIRS,qSOFA,SepsisLabel
0,78,95,100,60,36.5,18,8.5,2.0,1.0,2,0,0
1,80,96,102,62,36.6,19,8.7,2.1,1.0,2,0,0
...
48,85,94,110,65,38.2,22,12.0,4.5,1.2,3,1,1  <- Sepsis occurred
```

**Key columns:**
- `SepsisLabel = 1` indicates sepsis onset
- Time-series data (1 row per hour)
- Multiple vital signs per patient

---

## Recommended Workflow

**For immediate training (next 1 hour):**
```bash
# 1. Download PhysioNet (~10 min download, 5 min extract)
# 2. Copy to data/raw_physionet/
# 3. Run: python3 src/train_and_convert.py
```

**For production (1-2 weeks):**
```bash
# Add MIMIC-IV Sepsis-3 cohort for more diverse data
# Requires PhysioNet credentialing: https://physionet.org/
```

---

## Troubleshooting

### "No dataset available!"
```bash
# Verify PhysioNet files exist:
ls data/raw_physionet/*.csv | wc -l
# Should show ~40,000

# If empty, download from: https://physionet.org/content/challenge-2019/1.0.0/
```

### "SepsisLabel column not found"
```bash
# Check CSV headers:
head -1 data/raw_physionet/p000001.csv

# If different format, update PHYSIONET_FEATURE_MAP in src/load_physionet_data.py
```

### "Class imbalance still too low"
```bash
# Use larger patient subset or combine PhysioNet + MIMIC-IV
# Or apply class weights in train_and_convert.py (already implemented)
```

---

## Next Steps

1. **Download PhysioNet Challenge 2019 data**
2. **Extract to `data/raw_physionet/`**
3. **Run:** `python3 src/train_and_convert.py`
4. **Monitor:** Check `artifacts/reports/pipeline_report.md` for metrics

Your model is now training on real sepsis data with ~2,000 positive cases! 🎉
