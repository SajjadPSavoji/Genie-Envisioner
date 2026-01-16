# Genie-Envisioner Inference Testing Plan

## Overview

This plan outlines how to test all available inference capabilities in the Genie-Envisioner repository using the model weights you have available.

## Available Model Weights ✅ CONFIRMED

### GE Weights (`/shared_work/physical_intelligence/ge_weights/`)
| File | Purpose |
|------|---------|
| `ge_act_libero_goal.safetensors` | GE-Act for LIBERO Goal tasks ✅ |
| `ge_act_libero_object.safetensors` | GE-Act for LIBERO Object tasks ✅ |
| `ge_act_libero_10.safetensors` | GE-Act for LIBERO-10 tasks ✅ |
| `ge_act_libero_spatial.safetensors` | GE-Act for LIBERO Spatial tasks ✅ |
| `ge_act_calvin.safetensors` | GE-Act for Calvin tasks ✅ NEW |
| `GE_base_fast_v0.1.safetensors` | Fast video generation ✅ |
| `ge_base_slow_v0.1.safetensors` | High-quality video generation ✅ |
| `ge_sim_cosmos_v0.1.safetensors` | Action-conditioned world model ✅ |

### LTX-Video (`/shared_work/physical_intelligence/LTX-Video/`)
- `vae/` - VAE encoder/decoder ✅
- `tokenizer/` - Text tokenizer ✅
- `text_encoder/` - T5 text encoder ✅
- `scheduler/` - Diffusion scheduler ✅
- `model_index.json` - Model configuration ✅
- Multiple model versions (2b, 13b) available

---

## Testing Plan

### Phase 1A: GE-Act LIBERO Evaluation ✅ COMPLETED

**Status**: Finished

**What it tested**: Action prediction accuracy in LIBERO simulation benchmark

**Results location**: `evaluation_results/libero/<task_suite>/`

---

### Phase 1B: GE-Act Calvin Evaluation

> [!IMPORTANT]
> Calvin weights downloaded ✅ - Still requires Calvin environment setup

**Expected Performance** (from paper):
| Len-1 | Len-2 | Len-3 | Len-4 | Len-5 | Avg. Subtasks |
|-------|-------|-------|-------|-------|---------------|
| 0.950 | 0.898 | 0.857 | 0.808 | 0.747 | 4.260 |

#### Step 1: Download Calvin Dataset

```bash
cd /shared_work/physical_intelligence/calvin/dataset

# Download full ABCD dataset (~600GB)
bash download_data.sh ABCD
```

#### Step 2: Install Calvin Repository

```bash
cd /shared_work/physical_intelligence/calvin
bash install.sh
```

> [!WARNING]
> **Common Installation Issues (Python 3.10+)**
> 
> 1. **`pyhash` fails to build**: Comment out in `calvin_models/requirements.txt`, then patch `calvin_agent/evaluation/utils.py`:
>    ```python
>    # Replace: import pyhash / hasher = pyhash.fnv1_32()
>    # With:
>    class SimpleHasher:
>        def __call__(self, s):
>            return abs(hash(s)) & 0xFFFFFFFF
>    hasher = SimpleHasher()
>    ```
> 
> 2. **`MulticoreTSNE` fails**: Comment out in `requirements.txt` (not needed for eval)
> 
> 3. **Version conflicts**: Comment out strict versions (`torch==1.13.1`, `pytorch-lightning==1.8.6`, `hydra-core==1.1.1`) from `requirements.txt`
> 
> 4. **Missing `pytorch-lightning`**: Run `pip install pytorch-lightning`
> 
> 5. **`collections.Mapping` error**: Add this at the start of `eval_calvin.py`:
>    ```python
>    import collections, collections.abc
>    for name in ['Mapping', 'MutableMapping', 'Sequence', 'MutableSequence', 'Set', 'MutableSet', 'Callable', 'Iterable', 'Iterator']:
>        if not hasattr(collections, name):
>            setattr(collections, name, getattr(collections.abc, name))
>    ```
> 
> 6. **`fractions.gcd` error**: Run `pip install --upgrade networkx`

#### Step 3: Update Paths

Edit `experiments/eval_calvin.py` (lines 40-41):
```python
CALVIN_ROOT = "/shared_work/physical_intelligence/calvin"
CALVIN_DATASET = "/shared_work/physical_intelligence/calvin/dataset/task_ABCD_D"
```

Edit `configs/ltx_model/calvin/action_model_calvin.yaml`:
```yaml
pretrained_model_name_or_path: /shared_work/physical_intelligence/LTX-Video
diffusion_model:
  model_path: /shared_work/physical_intelligence/ge_weights/ge_act_calvin.safetensors
```

#### Step 4: Run Evaluation

```bash
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner

python experiments/eval_calvin.py -s 0 -e 1000 -d 0 \
    -r evaluation_results/calvin \
    -c configs/ltx_model/calvin/action_model_calvin.yaml \
    -w /shared_work/physical_intelligence/ge_weights/ge_act_calvin.safetensors
```

**Results location**: `evaluation_results/calvin/`


---

### Phase 2: GE-Base Video Generation ✅ TESTED

#### Step 2.1: Configure GE-Base-Slow

Edit `configs/ltx_model/video_model_infer_slow.yaml`:
```yaml
pretrained_model_name_or_path: /shared_work/physical_intelligence/LTX-Video
diffusion_model:
  model_path: /shared_work/physical_intelligence/ge_weights/ge_base_slow_v0.1.safetensors
```

#### Step 2.2: Run GE-Base-Slow Inference (Both Samples)

> [!TIP]
> Use `--n_chunk N` to generate longer videos autoregressively:
> - `--n_chunk 1` → ~1.9s (57 frames, default)
> - `--n_chunk 2` → ~3.8s (114 frames)
> - `--n_chunk 4` → ~7.6s (228 frames)
> - `--n_chunk 8` → ~15.2s (456 frames)

```bash
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner

# Sample 0 (default length ~1.9s)
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_slow.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path evaluation_results/video_gen/slow_sample_0

# Sample 0 (longer ~7.6s with 4 chunks)
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_slow.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path evaluation_results/video_gen/slow_sample_0_chunk4 \
    --n_chunk 4

# Sample 1
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_slow.yaml \
    --image_root video_gen_examples/sample_1 \
    --prompt_txt_file video_gen_examples/sample_1/prompt.txt \
    --output_path evaluation_results/video_gen/slow_sample_1
```

#### Step 2.3: Configure GE-Base-Fast

Edit `configs/ltx_model/video_model_infer_fast.yaml`:
```yaml
pretrained_model_name_or_path: /shared_work/physical_intelligence/LTX-Video
diffusion_model:
  model_path: /shared_work/physical_intelligence/ge_weights/GE_base_fast_v0.1.safetensors
```

#### Step 2.4: Run GE-Base-Fast Inference (Both Samples)

> [!TIP]
> Use `--n_chunk N` to generate longer videos autoregressively:
> - `--n_chunk 1` → ~0.8s (25 frames, default)
> - `--n_chunk 2` → ~1.6s (50 frames)
> - `--n_chunk 4` → ~3.2s (100 frames)
> - `--n_chunk 8` → ~6.4s (200 frames)

```bash
# Sample 0 (default length ~0.8s)
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_fast.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path evaluation_results/video_gen/fast_sample_0

# Sample 0 (longer ~3.2s with 4 chunks)
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_fast.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path evaluation_results/video_gen/fast_sample_0_chunk4 \
    --n_chunk 4

# Sample 1
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_fast.yaml \
    --image_root video_gen_examples/sample_1 \
    --prompt_txt_file video_gen_examples/sample_1/prompt.txt \
    --output_path evaluation_results/video_gen/fast_sample_1
```

**Expected outputs** (4 videos total):
- `evaluation_results/video_gen/slow_sample_0/video.mp4`
- `evaluation_results/video_gen/slow_sample_1/video.mp4`
- `evaluation_results/video_gen/fast_sample_0/video.mp4`
- `evaluation_results/video_gen/fast_sample_1/video.mp4`

---

### Phase 3: GE-Sim (Action-Conditioned World Model)

> [!IMPORTANT]
> Cosmos VAE/tokenizer downloaded to `/shared_work/physical_intelligence/Cosmos-Predict2-2B-Video2World/`

#### Configuration Options

**Option A: Single-GPU (Reduced Resolution)**
- Config: `configs/cosmos_model/acwm_cosmos.yaml`
- Resolution: `[192, 256]`, chunk: `13`
- Script: `gesim_video_gen_examples/infer_gesim.py`

**Option B: Multi-GPU (Full Resolution)** ✅ RECOMMENDED
- Config: `configs/cosmos_model/acwm_cosmos_multigpu.yaml`
- Resolution: `[384, 512]`, chunk: `25`  
- Script: `gesim_video_gen_examples/infer_gesim_multigpu.py`
- Uses both GPUs: VAE+TextEncoder on cuda:0, Diffusion on cuda:1

#### Step 3.1: Run GE-Sim Multi-GPU (Full Resolution)

```bash
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner
mkdir -p evaluation_results/gesim_video_gen

# Sample 0 - Multi-GPU (full resolution)
python gesim_video_gen_examples/infer_gesim_multigpu.py \
    --config_file=configs/cosmos_model/acwm_cosmos_multigpu.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_0_hires \
    --multi_gpu

# Sample 1
python gesim_video_gen_examples/infer_gesim_multigpu.py \
    --config_file=configs/cosmos_model/acwm_cosmos_multigpu.yaml \
    --image_root=gesim_video_gen_examples/sample_1 \
    --extrinsic_root=gesim_video_gen_examples/sample_1 \
    --intrinsic_root=gesim_video_gen_examples/sample_1 \
    --action_path=gesim_video_gen_examples/sample_1/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_1_hires \
    --multi_gpu

# Sample 2
python gesim_video_gen_examples/infer_gesim_multigpu.py \
    --config_file=configs/cosmos_model/acwm_cosmos_multigpu.yaml \
    --image_root=gesim_video_gen_examples/sample_2 \
    --extrinsic_root=gesim_video_gen_examples/sample_2 \
    --intrinsic_root=gesim_video_gen_examples/sample_2 \
    --action_path=gesim_video_gen_examples/sample_2/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_2_hires \
    --multi_gpu
```

#### Step 3.2: Run GE-Sim Single-GPU (Reduced Resolution, Alternative)

```bash
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner

# Sample 0
CUDA_VISIBLE_DEVICES=1 python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_0

# Sample 1
CUDA_VISIBLE_DEVICES=1 python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_1 \
    --extrinsic_root=gesim_video_gen_examples/sample_1 \
    --intrinsic_root=gesim_video_gen_examples/sample_1 \
    --action_path=gesim_video_gen_examples/sample_1/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_1

# Sample 2
CUDA_VISIBLE_DEVICES=1 python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_2 \
    --extrinsic_root=gesim_video_gen_examples/sample_2 \
    --intrinsic_root=gesim_video_gen_examples/sample_2 \
    --action_path=gesim_video_gen_examples/sample_2/actions.npy \
    --output_path=evaluation_results/gesim_video_gen/sample_2
```

**Expected outputs**:
- Multi-GPU: `evaluation_results/gesim_video_gen/sample_{0,1,2}_hires/video.mp4`
- Single-GPU: `evaluation_results/gesim_video_gen/sample_{0,1,2}/video.mp4`

---

### Phase 4: Open-Loop Policy Validation

This generates open-loop action prediction plots to verify model training quality.

```bash
bash scripts/infer.sh main.py \
    configs/ltx_model/policy_model_lerobot.yaml \
    /path/to/your/checkpoint.safetensors \
    evaluation_results/openloop \
    libero
```

---

### Phase 5: Web Server Deployment (Advanced)

For real robot deployment testing:

#### Step 5.1: Configure Server

Edit `web_infer_scripts/run_server.sh`:
```bash
IP_ADDRESS_OF_SERVER="localhost"  # or your actual IP
DOMAIN_NAME="libero"  # your dataset name
# Update checkpoint path
```

#### Step 5.2: Start Server
```bash
bash web_infer_scripts/run_server.sh
```

#### Step 5.3: Test Client
```bash
bash web_infer_scripts/run_simple_client.sh
```

---

## Verification Plan

| Phase | Test | Verification Method | Result |
|-------|------|---------------------|--------|
| 1A | GE-Act LIBERO | Check log files for success rates | ✅ Tested (performance varies from paper) |
| 1B | GE-Act Calvin | Check success rates across task lengths | ⏳ Pending (weights downloaded) |
| 2 | GE-Base Video | Visually inspect generated `.mp4` files | ✅ Tested |
| 3 | GE-Sim | Verify generated video matches action sequence | ✅ Tested |
| 4 | Open-Loop | Check generated plots match expected trajectories | ⏳ Skipped |
| 5 | Web Server | Verify client receives valid action predictions | ⏳ Skipped |
