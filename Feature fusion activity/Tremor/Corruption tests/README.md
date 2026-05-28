# Tremor Sensor Fusion Ablation Framework

This folder contains the full pipeline for the sensor-subset ablation study
used in the master's thesis. The framework trains a lightweight MLP fusion
classifier on top of pre-extracted CNN embeddings, evaluates every possible
subset of sensors, and measures how robust each combination is under signal
corruption.

---

## Scripts overview

| Script | Role |
|---|---|
| `v2_fusion_concat_ablation_study_tremor.py` | Core ablation script — trains and evaluates one k-value sweep |
| `v2_run_ablation_parallel.py` | Runs all k=1..8 sweeps in parallel across 2 GPUs |
| `v2_run_ablation_multirun_two_gpus.py` | Repeats the full ablation N times with different seeds |

---

## How it works

### 1. Inputs — pre-extracted CNN embeddings

The script does **not** train CNNs. It reads 128-dimensional embedding vectors
that have already been extracted by the sensor CNNs and saved as `.npz` files.
Each `.npz` contains embeddings for all windows, split by train/val/test, plus
the activity label and subject ID for each window.

The embedding files are read from:
```
data/Tremor_datagenerator_files/<variant>/<embeddings_folder_name>/<Sensor>_embeddings.npz
```

### 2. Two separate embedding pools

Training and test embeddings come from **different data variants** on purpose:

- **Training/validation** — loaded from `tremor_variants` (one or more variants
  concatenated together as augmentation). These contain tremor-augmented signals
  but no sensor corruption.

- **Test** — loaded from `corrupt_test_variant`. This variant applies a specific
  sensor corruption (e.g. AWGN at a fixed noise level) that is **never seen
  during training**. This setup simulates real-world deployment where a trained
  model encounters unexpected sensor degradation.

### 3. Subject-based data splitting

Splits are re-computed from subject IDs at load time to prevent data leakage:

| Split | Subjects |
|---|---|
| Train | 1, 3, 4, 6, 8, 9 |
| Validation | 2, 7 |
| Test | 5, 10 |

All windows from all original splits are pooled first, then re-divided by
subject ID. Correctness is verified by assertion before training starts.

### 4. Fusion model

For a subset of `k` sensors, the `k` embedding vectors (each 128-dimensional)
are concatenated into a single `128k`-dimensional input vector, which is fed
to an MLP classifier head:

```
Input (128k) → Linear(64) → ReLU → Dropout(0.4)
             → Linear(64) → ReLU → Dropout(0.4)
             → Linear(num_classes)
```

A new model is trained from scratch for every sensor combination.

### 5. Training

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Loss: CrossEntropyLoss
- Max epochs: 200
- Early stopping: patience=30 epochs, min_delta=1e-4, monitored on **validation macro F1**
- Best model state is restored before test evaluation

### 6. Sensor subset ablation

For each subset size `k` defined in `ABLATION_K`, the script trains and
evaluates all `C(8, k)` sensor combinations independently. Full sweep:

| k | Combinations |
|---|---|
| 1 | 8 |
| 2 | 28 |
| 3 | 56 |
| 4 | 70 |
| 5 | 56 |
| 6 | 28 |
| 7 | 8 |
| 8 | 1 |
| **Total** | **255** |

Results are written to a `.jsonl` log and a ranked `.txt`/`.csv` report.

---

## Running the full ablation (recommended)

### Single multi-seed run (all 255 combinations, N seeds, 2 GPUs):
```bash
python v2_run_ablation_multirun_two_gpus.py --n_runs 10 --gpus 0,1
```

This runs seeds 42–51 sequentially. Within each seed, k=1..8 are distributed
across the two GPUs in parallel (~10 min per seed on typical hardware).

### Explicit seeds:
```bash
python v2_run_ablation_multirun_two_gpus.py --seeds 42,43,44
```

### Single seed, all k, 2 GPUs (one run):
```bash
python v2_run_ablation_parallel.py --gpus 0,1 --seed 42
```

### Single k-value only (for debugging):
Edit `ABLATION_K = [2]` in `v2_fusion_concat_ablation_study_tremor.py`, then:
```bash
CUDA_VISIBLE_DEVICES=0 python v2_fusion_concat_ablation_study_tremor.py
```

---

## Configuration

All settings are at the top of `v2_fusion_concat_ablation_study_tremor.py`.

### Training data variants (`tremor_variants`)
Controls which tremor variants are concatenated for train/val:
```python
# Clean only:
tremor_variants = ["s4_w4_fs50_tremor_clean"]

# Mixed severity (used in thesis experiments):
tremor_variants = ["s4_w4_fs50_tremor_clean",
                   "s4_w4_fs50_tremor_mild_mod",
                   "s4_w4_fs50_tremor_mod_severe"]
```

### Test corruption variant (`corrupt_test_variant`)
The variant used exclusively for test evaluation:
```python
corrupt_test_variant = "s4_w4_fs50_tremor_clean_awgn_a100"  # AWGN at alpha=1.0
```

### Embeddings folder (`embeddings_folder_name`)
Must match which CNN was used for extraction:
```python
embeddings_folder_name = "ExtractedFeatures_mixed"   # mixed-trained CNNs
embeddings_folder_name = "ExtractedFeatures_clean"   # clean-trained CNNs
```

### Subset sizes (`ABLATION_K`)
```python
ABLATION_K = [8]          # only the full 8-sensor set
ABLATION_K = [1,2,3,4,5,6,7,8]  # full ablation sweep
```

### Output tag (`ABLATION_REPORT_TAG`)
Used to name the output folders so results from different configurations
do not overwrite each other:
```python
ABLATION_REPORT_TAG = "mixed_train_awgn_a100"
```
This creates:
```
corruption_logs/ablation_reports_mixed_train_awgn_a100/
corruption_logs/ablation_json_files_mixed_train_awgn_a100/
```

---

## Output structure

```
corruption_logs/
  ablation_reports_<tag>/
    ablation_report_k<k>_seed<s>_<timestamp>.txt   ← ranked human-readable report
    ablation_report_k<k>_seed<s>_<timestamp>.csv   ← same, as CSV
  ablation_json_files_<tag>/
    ablation_k<k>_seed<s>_<timestamp>.jsonl        ← one JSON line per combination
```

Each `.jsonl` line contains: sensors used, test accuracy, test macro F1,
val macro F1, training epochs run, and ablation metadata.

After all seeds have finished, run the aggregation script to compute
mean ± std per combination across all seeds:
```bash
python corruption_logs_old_cnn/analyze_results/analyze_multirun.py
```

---

## Available sensors

```python
ALL_SENSORS = ["Acc_ankle", "Acc_arm", "Gyro_ankle", "Gyro_arm",
               "Mag_ankle", "Mag_arm", "Acc_chest", "ECG"]
```
