# BEHAVIOR-1K Training & Testing Instructions

## Overview

This guide explains how to train and test Genie-Envisioner (GE-Base/GE-Act) on the BEHAVIOR-1K dataset.

## Files Reference

| File | Purpose |
|------|---------|
| `data/b1k_dataset.py` | B1K dataset loader |
| `b1k_stuff/compute_b1k_stats.py` | Compute normalization statistics |
| `configs/ltx_model/b1k/video_model_b1k.yaml` | Video adaptation config |
| `configs/ltx_model/b1k/action_model_b1k.yaml` | Action training config |
| `configs/ltx_model/b1k/behavior1k_stats.json` | Normalization statistics |

---

## Step 1: Compute Normalization Statistics

Compute action/state statistics from the dataset:

```bash
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner
conda activate genie_envisioner

# Full dataset (all 50 tasks)
python b1k_stuff/compute_b1k_stats.py \
    --data_root /shared_work/physical_intelligence/2025-challenge-demos \
    --save_path configs/ltx_model/b1k/behavior1k_stats.json

# Or specific tasks only (faster)
python b1k_stuff/compute_b1k_stats.py \
    --data_root /shared_work/physical_intelligence/2025-challenge-demos \
    --save_path configs/ltx_model/b1k/behavior1k_stats.json \
    --task_ids 0 1 2
```

---

## Step 2: Video Adaptation Training

Fine-tune GE-Base on B1K videos:

```bash
# Single GPU
bash scripts/train.sh main.py configs/ltx_model/b1k/video_model_b1k.yaml
```

---

## Step 3: Action Post-Training

Train the action expert (after video adaptation):

```bash
bash scripts/train.sh main.py configs/ltx_model/b1k/action_model_b1k.yaml
```

---

## Step 4: Testing GE-Base Video Generation (Zero-Shot)

Before training, you can test GE-Base's generalization on BEHAVIOR-1K data using the pretrained AGIBOT-World weights.

For step-by-step testing of GE-Base video generation on BEHAVIOR-1K data, use the interactive notebook:

```
b1k_stuff/zero_shot_b1k_testing.ipynb
```

The notebook covers:
1. Extracting frames from B1K videos
2. Creating sample folders with memory frames
3. Running GE-Base inference (slow/fast models)
4. Creating ground truth videos for comparison

---

## Step 5: Run Evaluation with BEHAVIOR-1K Simulator

This step runs the trained GE-ACT model with the BEHAVIOR-1K simulator.

### Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ BEHAVIOR-1K │         │ WebSocket Client │         │ WebSocket Server │
│ (Simulator) │   ───>  │   (eval.py)      │   ───>  │(serve_ge_act.py) │
└─────────────┘         └──────────────────┘         └──────────────────┘
     GPU 0                                                  GPU 1
```

### Terminal 1: Start GE-ACT Server (GPU 1)

```bash
# Navigate to GE-ACT directory
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner

# Activate environment
conda activate genie_envisioner

# Start server (test mode - returns zero actions)
CUDA_VISIBLE_DEVICES=1 python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight /shared_work/physical_intelligence/ge_weights/ge_act_calvin.safetensors \
    --task_name "turning_on_radio" \
    --host 0.0.0.0 \
    --port 8000 \
    --test_mode \
    --save_debug_images \
    --debug_image_dir /shared_work/physical_intelligence/ruiheng/tmp
```

**Wait for output:**
```
================================================================================
OUTPUT DIRECTORY:
  Actual: /shared_work/.../tmp/ge_act_20260109_120841
  Symlink: /shared_work/.../tmp/ge_act_latest
================================================================================
Server ready! Listening on 0.0.0.0:8000
```

### Terminal 2: Run BEHAVIOR-1K Evaluation (GPU 0)

**IMPORTANT:** Start server first, wait for "Server ready!" message.

```bash
# Navigate to BEHAVIOR-1K directory
cd /shared_work/physical_intelligence/BEHAVIOR-1K

# Activate environment
conda activate behavior

# Run evaluation with GUI
CUDA_VISIBLE_DEVICES=0 python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=/shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest \
    headless=false
```

**Note:** Remove `headless=false` for headless mode (faster).

### Terminal 3: Check Outputs

```bash
ls -lh /shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest/
```

**Expected structure:**
```
ge_act_latest/ -> ge_act_20260109_115808/
├── images/                     # Debug images from GE-ACT
│   ├── head/
│   ├── left_wrist/
│   └── right_wrist/
├── videos/                     # Episode videos from BEHAVIOR-1K
├── metrics/                    # Task metrics from BEHAVIOR-1K
└── ge_act_metadata.json        # Run metadata
```

---

## Step 6: Run with Actual Model (Not Yet Implemented)

> **⚠️ NOT IMPLEMENTED:** The actual model inference path is not correctly implemented yet. The action space mapping from GE-ACT (Calvin-trained, 22-dim) to BEHAVIOR-1K R1Pro (23-dim) requires proper implementation. Currently only test mode (zero actions) works reliably.

**TODO:**
- [ ] Implement proper action space mapping from Calvin to R1Pro
- [ ] Train GE-ACT on BEHAVIOR-1K data
- [ ] Validate action outputs are in correct format/range

Once implemented, remove `--test_mode` to use actual model inference:

```bash
CUDA_VISIBLE_DEVICES=1 python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight /shared_work/physical_intelligence/ge_weights/ge_act_calvin.safetensors \
    --task_name "turning_on_radio" \
    --host 0.0.0.0 \
    --port 8000 \
    --save_debug_images \
    --debug_image_dir /shared_work/physical_intelligence/ruiheng/tmp
```


---

## Appendix: Key Config Parameters

### Video Model (`video_model_b1k.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_mode` | `video_only` | Video generation training |
| `sample_size` | `[256, 256]` | Output resolution |
| `valid_cam` | 3 cameras | head, left_wrist, right_wrist |
| `chunk` | 9 | Video frames to predict |
| `n_previous` | 4 | Memory frames |

### Debug Mode

For quick testing, the configs include debug settings:

```yaml
# In video_model_b1k.yaml
data:
  train:
    task_ids: [0]                  # Only load task 0
    max_episodes_per_task: 5       # Only 5 episodes
```

To train on full dataset, comment out or remove these lines.

---

## Appendix: Key Files for Evaluation

| File | Location | Purpose |
|------|----------|---------|
| `serve_ge_act_b1k.py` | `web_infer_scripts/` | WebSocket server entry point |
| `ge_act_b1k_wrapper.py` | `web_infer_utils/` | Observation/action adapter |
| `MVActor.py` | `web_infer_utils/` | Core GE-ACT inference |
