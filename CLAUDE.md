# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This workspace focuses on **Genie-Envisioner** - a unified world foundation platform for robotic manipulation consisting of:
- **GE-Act**: Action prediction model (policy) based on LTX-Video architecture
- **GE-Sim**: Video generation world model based on Cosmos2 or LTX-Video

The system uses diffusion-based transformers and supports multiple robotic datasets including AgiBotWorld, LIBERO, Calvin, and BEHAVIOR-1K.

**Related directories** (reference materials or model weights):
- **BEHAVIOR-1K/**: Reference simulation benchmark (evaluation environment)
- **b1k-baselines/**: Reference baseline implementations (not actively developed)
- **calvin/**: Reference CALVIN benchmark
- **LTX-Video/**: Model weights used by GE-Act
- **Cosmos-Predict2-2B-Video2World/**: Model weights used by GE-Sim

## Code Accuracy and Verification Protocol

Before writing any code in this repository, follow this protocol:

### 1. Question and Verify

After drafting code mentally or in your response, pause and ask yourself:
- Does this code actually solve the problem correctly?
- Are there edge cases I haven't considered?
- Could this introduce bugs, security vulnerabilities, or performance issues?
- Are there assumptions I'm making that might be wrong?
- Is this the simplest correct solution, or am I overcomplicating it?

### 2. Test Your Code

When appropriate, you should:
- Write and run tests to verify the code works as expected
- Test edge cases and boundary conditions
- Verify that existing functionality still works (no regressions)
- If you cannot test directly, explain what tests would be needed

### 3. Ask for Clarification

You have full freedom to ask the user questions when:
- Requirements are ambiguous or unclear
- Multiple valid approaches exist and user preference matters
- You're unsure about existing system behavior or constraints
- You need to understand the broader context
- You're making assumptions that could significantly impact the solution

### 4. Be Transparent

When you have doubts:
- Explicitly state your uncertainties
- Explain your reasoning and assumptions
- Present trade-offs when multiple solutions exist
- Admit when you need to investigate further before writing code

**Remember**: It's better to ask questions and write accurate code than to confidently write incorrect code. Correctness is prioritized over speed.

**Why this matters for Genie Envisioner**:
- Training runs are expensive (time + GPU hours)
- Wrong configurations can waste significant compute resources
- Action predictions affect real robot behavior and safety
- Dataset formats must be exact (LeRobot format requirements)
- Multi-stage training has dependencies (video adaptation → action training)
- Config errors may only surface after hours of training

## Implementation Testing Requirements

When implementing or modifying code:

1. **Always verify your implementation** - After writing code, double-check that it works as intended.

2. **Test standalone functionality** - Test only the specific functions or components you've modified. Ensure each function works independently without relying on external context.

3. **Use command-line testing** - When possible, test your implementation by:
   - Creating a temporary test file (e.g., `test_temp.py`, `temp_test.js`)
   - Running the code with sample inputs to verify behavior
   - Deleting the test file after verification is complete

4. **Environment setup** - If the project uses a conda environment, activate it before testing:
   ```bash
   conda activate genie_envisioner
   ```

5. **Cleanup** - Always remove any temporary test files created during verification.

6. **Ask questions when needed** - Feel free to ask clarifying questions at any point to better understand requirements, resolve ambiguities, or improve your workflow.

Example workflow:
- Write/modify the function
- Create test_temp.py with the function and test cases
- Activate conda environment: `conda activate genie_envisioner`
- Run `python test_temp.py` to verify
- Delete test_temp.py after successful verification

## Directory Structure

```
Genie-Envisioner/
├── configs/                  # Model and training configurations
│   ├── ltx_model/           # LTX-based GE-Act configs
│   └── cosmos_model/        # Cosmos-based GE-Sim configs
├── data/                    # Dataset loaders
│   └── utils/              # Dataset utilities and statistics
├── models/                  # Model architectures
│   ├── ltx_models/         # LTX-Video architecture
│   ├── cosmos_models/      # Cosmos architecture
│   └── pipeline/           # Inference pipelines
├── runner/                  # Training and inference runners
├── scripts/                 # Training and utility scripts
├── web_infer_scripts/       # Deployment scripts
├── web_infer_utils/         # WebSocket server utilities
├── experiments/             # Evaluation scripts
├── gesim_video_gen_examples/ # GE-Sim inference examples
└── video_gen_examples/      # Video generation examples
```

## Common Commands

### Environment Setup

```bash
conda create -n genie_envisioner python=3.10.4
conda activate genie_envisioner
pip install -r requirements.txt
```

### Training

Multi-GPU training using torchrun (automatically detects all GPUs):
```bash
bash scripts/train.sh main.py configs/ltx_model/video_model.yaml
```

Single-node multi-GPU training (explicit GPU count):
```bash
NGPU=8 bash scripts/train.sh main.py configs/ltx_model/policy_model_lerobot.yaml
```

Multi-node training (requires setting WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT):
```bash
WORLD_SIZE=2 RANK=0 MASTER_ADDR=node1 MASTER_PORT=29500 bash scripts/train.sh main.py configs/ltx_model/video_model.yaml
```

### Inference

Action inference (policy evaluation):
```bash
bash scripts/infer.sh main.py \
    configs/ltx_model/policy_model_lerobot.yaml \
    path/to/checkpoint.safetensors \
    path/to/output \
    DATASETNAME
```

Video generation:
```bash
bash scripts/infer.sh main.py \
    configs/ltx_model/video_model_infer_slow.yaml \
    path/to/checkpoint.safetensors \
    path/to/output \
    DATASETNAME
```

Simple video generation example:
```bash
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_slow.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path path/to/output
```

### GE-Sim (Cosmos2-based) Inference

Single GPU:
```bash
python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=results/
```

Multi-GPU:
```bash
python gesim_video_gen_examples/infer_gesim_multigpu.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=results/
```

Prepare custom input data:
```bash
python gesim_video_gen_examples/get_example_gesim_inputs.py \
    --data_root=${YOUR_AGIBOTWORLD_ROOT} \
    --task_id=${TASK_id} \
    --episode_id=${EPI_ID} \
    --save_root=gesim_video_gen_examples/sample_0 \
    --valid_start=0 \
    --valid_end=300
```

### Dataset Statistics

Calculate action statistics for normalization:
```bash
# Note: data_root points to the data/ subfolder containing parquet files
python scripts/get_statistics.py \
    --data_root PATH/TO/DATASET/data \
    --data_name DATASETNAME \
    --data_type joint \
    --action_key action \
    --state_key observation.state \
    --save_path PATH/OF/stats.json
```

### Deployment

Start GE-Act server (WebSocket-based):
```bash
# Edit web_infer_scripts/run_server.sh to set IP_ADDRESS_OF_SERVER and DOMAIN_NAME
bash web_infer_scripts/run_server.sh
```

Run simple test client:
```bash
bash web_infer_scripts/run_simple_client.sh
```

BEHAVIOR-1K integration server:
```bash
CUDA_VISIBLE_DEVICES=1 python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight path/to/checkpoint.safetensors \
    --task_name "task_name" \
    --host 0.0.0.0 \
    --port 8000
```

### Simulation Benchmarks

Evaluate on Calvin:
```bash
bash experiments/eval_calvin.sh
```

Evaluate on LIBERO:
```bash
bash experiments/eval_libero.sh
```

## Architecture

### Core Components

1. **Models** (`models/`)
   - `ltx_models/`: LTX-Video based transformer and VAE implementations
     - `transformer_ltx_multiview.py`: Multi-view transformer with action expert capabilities
     - `autoencoder_kl_ltx.py`: Video VAE for encoding/decoding frames
     - `ltx_attention_processor.py`: Custom attention mechanisms
   - `cosmos_models/`: Cosmos2-based world model implementations
   - `action_patches/`: Action conditioning modules
   - `pipeline/`: Diffusion pipelines for inference
     - `custom_pipeline.py`: LTX-based video+action pipeline
     - `gesim_pipeline.py`: Cosmos2-based world model pipeline
     - `gesim_pipeline_multigpu.py`: Multi-GPU version

2. **Data Loading** (`data/`)
   - `agibotworld_dataset.py`: AgiBotWorld dataset loader
   - `lerobot_like_dataset.py`: Generic LeRobot-format dataset loader
   - `libero_dataset.py`: LIBERO simulation dataset
   - `utils/`: Dataset utilities for statistics, transforms, etc.

3. **Training & Inference** (`runner/`)
   - `ge_trainer.py`: Main training loop with accelerate/deepspeed support
   - `ge_inferencer.py`: Inference runner for evaluation

4. **Web Inference** (`web_infer_utils/`, `web_infer_scripts/`)
   - `MVActor.py`: Multi-view actor wrapper for inference
   - `ge_act_b1k_wrapper.py`: BEHAVIOR-1K specific adapter
   - `server.py`: WebSocket server utilities
   - `openpi_client/`: OpenPI protocol client implementation

### GE-Act (Action Model)

- **Base**: LTX-Video diffusion transformer architecture
- **Extension**: Action prediction head added to transformer
- **Training Strategy**: Two-stage approach
  1. Video adaptation: Finetune video generation on target domain
  2. Action post-training: Train action head with frozen/partially frozen video model
- **Input**: Multi-view images (e.g., top_head, hand_left, hand_right) + proprioceptive state
- **Output**: Action predictions (absolute, delta, or relative joint positions)
- **Data Format**: LeRobot-style `.parquet` episodes with separate video files

### GE-Base (Video Generation Foundation)

- **Base**: LTX-Video diffusion transformer architecture
- **Purpose**: Foundation video generation model that GE-Act and GE-Sim build upon
- **Key Concept**: Memory frames - sparse conditioning frames from history that serve as "anchor points" for temporal coherence

#### Memory Frames

Memory frames are sparsely sampled historical frames that guide the diffusion model to generate temporally coherent videos.

**Configuration** (`n_previous` or `n_prev` parameter):
- Default: `sample_n_frames - chunk` (e.g., 64 - 1 = 63 frames during training)
- Inference: Typically 4-8 frames (configurable)

**Selection Strategies** (in `data/agibotworld_dataset.py:229-263`):
1. **Uniform mode**: Evenly spaced frames across the history window
2. **Random mode**: Randomly selected from candidate pool

**Multi-chunk inference** (in `models/pipeline/custom_pipeline.py:977-983`):
```python
new_mem_idxs = torch.linspace(0, video_list.shape[2]-1, n_prev).round().long()
new_mems = video_list[:, :, new_mem_idxs, :, :].clone()
```

**Key Implementation Details**:

1. **Separate VAE encoding** (`utils/data_utils.py:97-129`):
   - Memory frames are encoded individually since they're sparsely sampled
   - `get_latents()` function handles mem and video separately

2. **Memory frames stay fixed during diffusion** (`models/pipeline/custom_pipeline.py:941-947`):
   ```python
   video_noise_pred = video_noise_pred[:, :, n_prev:]  # Only predict future frames
   latents = torch.cat([latents[:, :, :n_prev], pred_latents], dim=2)  # Keep memories
   ```

3. **Training-time noise conditioning** (`utils/data_utils.py:272-323`):
   - Memory frames receive small random timesteps during training
   - `gen_noise_from_condition_frame_latent()` implements this
   - Model learns to use them as conditioning signals

**Key Files**:
- Dataset: `data/agibotworld_dataset.py`, `data/lerobot_like_dataset.py`
- VAE encoding: `utils/data_utils.py`
- Inference pipeline: `models/pipeline/custom_pipeline.py`
- Inference script: `video_gen_examples/infer.py`

### GE-Sim (World Model)

- **Base**: Cosmos2 Predict2-2B-Video2World or LTX-Video
- **Purpose**: Action-conditioned video generation for simulation
- **Input**:
  - Sparse history frames (multi-view images)
  - Action sequences
  - Camera extrinsics (camera-to-base transforms)
  - Camera intrinsics
- **Output**: Generated video frames showing predicted future
- **Key Function**: `get_cam2base()` in `gesim_video_gen_examples/get_example_gesim_inputs.py` - computes camera extrinsics from end-effector poses

Input format for `infer_gesim.py`:
- Images: Multi-view frames as PNG/JPG
- Extrinsics: `.npy` files with shape `(num_frames, 4, 4)`
- Intrinsics: `.npy` files with shape `(num_cameras, 4)` or `(num_frames, num_cameras, 4)`
- Actions: `.npy` file with shape `(num_frames, action_dim)`

### Configuration System

All configs are YAML-based in `configs/`:

- **LTX Model configs** (`configs/ltx_model/`):
  - `video_model.yaml`: Video generation pre-training
  - `policy_model.yaml`: Action model post-training
  - `video_model_lerobot.yaml`: Task-specific video adaptation for LeRobot datasets
  - `policy_model_lerobot.yaml`: Action post-training for LeRobot datasets
  - `video_model_infer_slow.yaml` / `video_model_infer_fast.yaml`: Inference configs for different latencies
  - Domain-specific configs in `libero/`, `calvin/`, `b1k/` subdirectories

- **Cosmos Model configs** (`configs/cosmos_model/`):
  - `acwm_cosmos.yaml`: Single-GPU GE-Sim config
  - `acwm_cosmos_multigpu.yaml`: Multi-GPU GE-Sim config

Key config parameters:
- `pretrained_model_name_or_path`: Path to VAE/tokenizer weights
- `diffusion_model.model_path`: Path to checkpoint (.safetensors)
- `train_mode`: `'video_only'`, `'action_full'`, or hybrid modes
- `return_action`/`return_video`: Control what the model predicts
- `diffusion_model.config.action_expert`: Enable/disable action expert module

### Training Pipeline

The training flow (`main.py` with mode=`train`):

1. Load config and instantiate Runner (typically `ge_trainer.py:Trainer`)
2. `prepare_dataset()`: Load train/val datasets based on config
3. `prepare_models()`: Load VAE, text encoder, and diffusion transformer
4. `prepare_trainable_parameters()`: Set up parameter groups (e.g., freeze VAE)
5. `prepare_optimizer()`: Create optimizer (AdamW, Prodigy, or CAME)
6. `prepare_for_training()`: Initialize accelerator, wrap models with DDP/DeepSpeed
7. `prepare_trackers()`: Set up tensorboard/wandb logging
8. `train()`: Main training loop with gradient accumulation and checkpointing

### Inference Pipeline

The inference flow (`main.py` with mode=`infer`):

1. Load config and instantiate Runner (typically `ge_inferencer.py:Inferencer`)
2. Override checkpoint path via `--checkpoint_path` argument
3. `prepare_val_dataset()`: Load validation dataset
4. `prepare_models()`: Load models with checkpoint weights
5. `infer()`: Generate predictions and save outputs
   - For action models: Predict action sequences, create open-loop validation plots
   - For video models: Generate videos from sparse frames + prompts

## Training Workflows

### Overview of Training Stages

Genie Envisioner follows a multi-stage training strategy:

1. **GE-Base Pretraining**: Train video generation model on large-scale robotic dataset (AgiBotWorld)
2. **Task-specific Video Adaptation** (Optional but recommended): Fine-tune video generation on target domain/robot
3. **Action Post-Training**: Add and train action expert module for policy learning

### Stage 1: GE-Base Pretraining

**Purpose**: Train a foundational video generation model on diverse robotic manipulation data.

**Prerequisites**:
- Download [AgiBotWorld dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)
- Download LTX-Video base weights:
  - [ltx-video-2b-v0.9.safetensors](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors) (transformer backbone)
  - [text_encoder](https://huggingface.co/Lightricks/LTX-Video/tree/main/text_encoder) (T5 encoder)
  - [tokenizer](https://huggingface.co/Lightricks/LTX-Video/tree/main/tokenizer) (T5 tokenizer)
  - [vae](https://huggingface.co/Lightricks/LTX-Video/tree/main/vae) (Video autoencoder)
  - [model_index.json](https://huggingface.co/Lightricks/LTX-Video/blob/main/model_index.json)

**Config**: `configs/ltx_model/video_model.yaml`

**Key settings**:
```yaml
pretrained_model_name_or_path: /path/to/ltx-video/base  # VAE + tokenizer + text_encoder
diffusion_model:
  model_path: /path/to/ltx-video-2b-v0.9.safetensors
  config:
    action_expert: false  # No action module during pretraining

return_action: false
return_video: true
train_mode: 'video_only'

data:
  train:
    data_roots: ["/path/to/AgiBotWorld-Beta"]
    task_info_root: ["/path/to/AgiBotWorld-Beta/task_info"]
    domains: ["agibotworld"]
    dataset_info_cache_path: "/path/to/cache/train"  # Cache metadata for faster loading
    valid_cam: ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right']
    chunk: 9                # Number of future frames to generate
    n_previous: 4           # Number of memory frames (context)
    sample_n_frames: 900    # Total frames per episode to sample from
```

**Training command**:
```bash
bash scripts/train.sh main.py configs/ltx_model/video_model.yaml
```

**What happens**:
- Model learns to generate future video frames conditioned on memory frames and text prompts
- VAE and text encoder are frozen; only transformer weights are trained
- Output: GE-Base checkpoint (e.g., `ge_base_fast.safetensors` or `ge_base_slow.safetensors`)

**Training modes available**:
- `ge_base_fast`: Lower frame rate, optimized for low-latency applications
- `ge_base_slow`: Higher frame rate, synchronized with action dynamics (better for action prediction)

### Stage 2: Task-specific Video Adaptation (Finetuning)

**Purpose**: Adapt GE-Base to unseen robots or new task domains for better performance.

**When to use**: Although GE-Base has zero-shot capability, this step is **strongly recommended** for:
- Robots with different morphology than AgiBotWorld
- New camera configurations
- Domain-specific visual characteristics
- Custom task distributions

**Prerequisites**:
- LeRobot-format dataset of your target domain
- Pretrained GE-Base checkpoint
- Action statistics (see Dataset Statistics section)

**Config**: `configs/ltx_model/video_model_lerobot.yaml`

**Key settings**:
```yaml
pretrained_model_name_or_path: /path/to/ltx-video/base
diffusion_model:
  model_path: /path/to/ge_base_fast.safetensors  # Start from pretrained GE-Base
  config:
    action_expert: false  # Still no action module

return_action: false
return_video: true
train_mode: 'video_only'  # Only train video generation

data:
  train:
    data_roots: ["/path/to/your/dataset"]
    domains: ["your_dataset_name"]
    valid_cam: ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
    stat_file: /path/to/your_dataset_stats.json  # Action normalization stats
    chunk: 9
    action_chunk: 54        # Must be multiple of chunk (54 = 6 actions/frame × 9 frames)
    n_previous: 4
    previous_pick_mode: 'random'  # How to select memory frames: 'random' or 'uniform'
```

**Training command**:
```bash
bash scripts/train.sh main.py configs/ltx_model/video_model_lerobot.yaml
```

**What happens**:
- Fine-tunes the transformer on domain-specific video generation
- Model learns visual patterns specific to your robot/environment
- Output: Video-adapted checkpoint (e.g., `video_adapted_epoch_X.safetensors`)

### Stage 3: Action Post-Training (Policy Learning)

**Purpose**: Train the action expert module to predict robot actions from visual observations.

**Prerequisites**:
- Video-adapted checkpoint from Stage 2 (or GE-Base if skipping Stage 2)
- LeRobot-format dataset with action labels
- Action statistics file

**Config**: `configs/ltx_model/policy_model_lerobot.yaml`

**Key settings**:
```yaml
pretrained_model_name_or_path: /path/to/ltx-video/base
diffusion_model:
  model_path: /path/to/video_adapted.safetensors  # Load video-adapted weights
  config:
    action_expert: true   # Enable action expert module
    action_in_channels: 14              # Action feature dimension
    action_num_attention_heads: 16      # Attention heads for action expert
    action_attention_head_dim: 32       # Head dimension for action expert

return_action: true
return_video: false    # Only train action prediction (freeze video model)
train_mode: 'action_full'

add_state: false  # Whether to include proprioceptive state in action chunk

data:
  train:
    data_roots: ["/path/to/your/dataset"]
    domains: ["your_dataset_name"]
    valid_cam: ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
    stat_file: /path/to/your_dataset_stats.json
    action_key: "action"           # Key in parquet files
    state_key: "observation.state" # Key for proprioception
    action_type: "delta"           # "absolute", "delta", or "relative"
    action_space: "joint"          # "joint" or "eef" (end-effector)
    chunk: 9
    action_chunk: 54               # Predict 54 future actions
    n_previous: 4
```

**Training command**:
```bash
bash scripts/train.sh main.py configs/ltx_model/policy_model_lerobot.yaml
```

**What happens**:
- With `train_mode: 'action_full'`: Trains ALL parameters (video backbone + action expert jointly)
- With `train_mode: 'action_only'`: Freezes video weights, only trains action expert parameters
- Model learns visuomotor policy for manipulation
- Output: GE-Act checkpoint (saved as folder with `diffusion_pytorch_model.safetensors` inside)

**Note**: Despite the name, `action_full` does NOT freeze video weights. Use `action_only` if you want frozen video backbone.

**Validation**: After training, check open-loop validation plots to ensure the model fits training data well. See `experiments/RUN.md` for example.

### Training Parameters Explained

**Data Parameters**:
- `chunk`: Number of video frames to predict (typically 9)
- `action_chunk`: Number of actions to predict (must be multiple of `chunk`)
  - Example: `chunk=9, action_chunk=54` means 6 actions per frame (`video_temporal_stride = action_chunk / chunk`)
- `n_previous`: Number of memory/context frames (typically 4)
- `previous_pick_mode`: How to sample memory frames:
  - `'random'`: Random temporal positions (more augmentation, recommended for training)
  - `'uniform'`: Evenly spaced (more consistent, use for validation)
- `sample_n_frames`: Window size to sample frames from each episode
- `sample_size`: Frame resolution `[height, width]` (typically `[192, 256]` or `[256, 256]` for Calvin)
- `preprocess`: Image preprocessing (`'resize'` or `'center_crop_resize'`)
- `random_crop`: Apply random cropping augmentation (set `False` for validation)
- `n_repeat`: Number of times to repeat the dataset (for small datasets)
- `ignore_seek`: Only load first future frame for efficiency (set `True` for action-only training)

**Action Parameters**:
- `action_type`:
  - `'absolute'`: Raw joint positions/velocities
  - `'delta'`: Change from previous timestep (more stable for most tasks)
  - `'relative'`: Difference from current state (for closed-loop)
- `action_space`:
  - `'joint'`: Joint space control
  - `'eef'`: End-effector space control
- `action_key`: Key in parquet files for actions (default: `"action"`, Calvin uses `"relative_action"`)
- `state_key`: Key for proprioception (default: `"observation.state"`)
- `add_state`: Include proprioceptive state in action predictions (set `True` for Calvin)
- `action_in_channels`: Action feature dimension in model config (varies by dataset: 14 for AgiBotWorld, 22 for Calvin)
- `action_out_channels`: Action output dimension (usually same as `action_in_channels`)

**Optimizer Parameters**:
- `optimizer`: `'adamw'`, `'prodigy'`, or `'came'`
- `lr`: Learning rate (typically `3e-5` for finetuning, `1e-4` for pretraining)
- `lr_scheduler`: `'constant_with_warmup'`, `'cosine'`, `'linear'`
- `lr_warmup_steps`: Steps for warmup (typically 1000)
- `weight_decay`: L2 regularization (typically `1e-5`)
- `max_grad_norm`: Gradient clipping (typically `1.0`)
- `gradient_accumulation_steps`: Simulate larger batch size

**Training Control**:
- `batch_size`: Per-GPU batch size
- `mixed_precision`: `'bf16'` (recommended), `'fp16'`, or `'no'`
- `gradient_checkpointing`: Enable to reduce memory (slight speed penalty)
- `train_steps` / `train_epochs`: Training duration (whichever reached first)
- `steps_to_save`: Checkpoint frequency
- `steps_to_val`: Validation frequency (set very high like `1000000000` to disable)
- `steps_to_log`: Logging frequency to tensorboard
- `load_weights`: **Must be `True`** to load pretrained checkpoint (set `False` only for debugging)

**Loss and Noise Parameters**:
- `action_loss_scale`: Weight for action loss relative to video loss (default: `1.0`)
- `caption_dropout_p`: Probability of dropping text prompt for classifier-free guidance (typically `0.06`, set `0.0` for Calvin)
- `noise_to_first_frame`: Amount of noise added to conditioning frames (typically `0.1`, use `0.01` for Calvin)
- `noisy_video`: When `True` with `return_action=True`, video latents are fully noised (`sigma=1.0`), focusing training on action prediction
- `use_color_jitter`: Apply color jitter augmentation to video frames
- `num_inference_step`: Denoising steps during validation (typically 5-10)

**Flow Matching Parameters**:
- `flow_weighting_scheme`: Loss weighting (`'none'`, `'sigma_sqrt'`, `'logit_normal'`, `'mode'`)
- `flow_logit_mean`, `flow_logit_std`, `flow_mode_scale`: Parameters for specific weighting schemes
- `pixel_wise_timestep`: Use different timesteps per pixel vs per frame (typically `True`)

**DeepSpeed ZeRO**:
```yaml
use_deepspeed: true
deepspeed:
  zero_optimization:
    stage: 2  # Stage 2 for multi-GPU, Stage 3 for very large models
  bf16:
    enabled: true
  gradient_clipping: 1.0
```

### Training Modes Summary

| Mode | `train_mode` | `return_action` | `return_video` | `action_expert` | What Gets Trained | Use Case |
|------|-------------|----------------|---------------|----------------|-------------------|----------|
| **Video Pretraining** | `'video_only'` | `false` | `true` | `false` | All non-action params | GE-Base pretraining |
| **Video Adaptation** | `'video_only'` | `false` | `true` | `false` | All non-action params | Task-specific finetuning |
| **Action Only** | `'action_only'` | `true` | `false` | `true` | Only action_* params | Freeze video, train action head |
| **Action Full** | `'action_full'` | `true` | `false` | `true` | **All params** (video + action) | Joint video+action finetuning |
| **All** | `'all'` | `true` | `true` | `true` | All params, both losses | Full joint training |

**Important**: `action_full` trains ALL parameters including video backbone. If you want to freeze video weights and only train the action expert, use `action_only` instead.

### Dataset Statistics Format

Action statistics JSON structure (includes percentiles for potential clipping):
```json
{
  "DATASETNAME_joint": {
    "mean": [0.1, 0.2, ...],
    "std": [1.0, 1.2, ...],
    "q99": [2.5, 3.0, ...],
    "q01": [-2.5, -3.0, ...]
  },
  "DATASETNAME_delta_joint": {
    "mean": [0.0, 0.0, ...],
    "std": [0.5, 0.6, ...],
    "q99": [...],
    "q01": [...]
  },
  "DATASETNAME_state_joint": {
    "mean": [0.1, 0.2, ...],
    "std": [1.0, 1.2, ...],
    "q99": [...],
    "q01": [...]
  }
}
```

**Note**: The script samples up to 50,000 episodes and filters outliers using 1st/99th percentiles before computing mean/std.

### Dataset-Specific Configurations

**Calvin** (simulation benchmark):
```yaml
sample_size: [256, 256]
valid_cam: ['observation.images.top', 'observation.images.wrist']  # 2 cameras
action_in_channels: 22
action_type: "absolute"
action_key: "relative_action"  # Note: uses relative_action despite absolute type
add_state: True
caption_dropout_p: 0.0
noise_to_first_frame: 0.01
ignore_seek: True
```

**LIBERO** (simulation benchmark):
```yaml
sample_size: [192, 256]
valid_cam: ['observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb']
action_in_channels: 7  # 7-DOF end-effector
action_type: "delta"
action_space: "eef"
```

**AgiBotWorld** (real robot):
```yaml
sample_size: [192, 256]
valid_cam: ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right']  # 3 cameras
action_in_channels: 14
action_type: "delta"
action_space: "joint"
```

### Best Practices

1. **Always start from pretrained weights**: Don't train from scratch unless you have massive data
2. **Use GE-Base-fast for action prediction**: Better suited for policy learning than slow version
3. **Validate action predictions**: Check open-loop plots - model should fit training data well before deployment
4. **Action space choice**: `delta` actions are generally more stable than `absolute` for most robots
5. **Memory requirements**:
   - Video training: ~40GB VRAM per GPU (batch_size=1)
   - Action training: ~30GB VRAM per GPU (batch_size=1)
   - Use gradient checkpointing if OOM
6. **Dataset size**: Minimum ~1000 episodes for video adaptation, ~5000+ for robust policy learning
7. **Checkpoint format**: Models are saved as folders containing `diffusion_pytorch_model.safetensors` (or sharded `.safetensors.index.json` for large models)

## Dataset Format

### LeRobot Format (for GE-Act training)

```
DATASET_ROOT/
├── DATASETNAME/
│   ├── data/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── meta/
│   │   ├── episodes_stats.jsonl
│   │   ├── episodes.jsonl
│   │   ├── tasks.json
│   │   └── info.json
│   └── videos/
│       └── chunk-000/
│           ├── observation.images.top_head/
│           │   ├── episode_000000.mp4
│           │   └── ...
│           ├── observation.images.hand_left/
│           └── observation.images.hand_right/
```

Key requirements:
- Multi-view camera support (configure in `valid_cam` list)
- Action normalization via statistics files (JSON with mean/std per domain and data type)
- Supports absolute, delta, and relative action spaces

### GE-Sim Input Format

- Images: `.npy` files or image files
- Extrinsics: `.npy` file with shape `[T, 4, 4]` (camera-to-base transform matrices)
- Intrinsics: `.npy` file with camera intrinsic matrices
- Actions: `.npy` file with action sequence

### Action Space Handling

- Actions and states are normalized using pre-computed statistics
- Statistics files contain mean/std for: `{DATASET}_{joint|eef}`, `{DATASET}_delta_{joint|eef}`, `{DATASET}_state_{joint|eef}`
- Action dimensions vary by domain (e.g., Calvin: 22 dims, BEHAVIOR-1K: 23 dims)
- The action expert module in the transformer outputs action predictions in the latent space

## Common Workflows

### Training GE-Act for New Robot/Task

1. Collect demonstrations in LeRobot format
2. Compute action statistics with `scripts/get_statistics.py`
3. Update config files with data paths and camera names
4. Run video adaptation (improves performance significantly)
5. Run action post-training with video checkpoint
6. Validate with `scripts/infer.sh`
7. Deploy with `web_infer_scripts/run_server.sh`

### Evaluating GE-Act on BEHAVIOR-1K

1. Deploy GE-Act as WebSocket server on port 8000
2. Use BEHAVIOR-1K evaluation client:
   ```bash
   conda activate behavior
   python BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py \
       policy=websocket \
       task.name=turning_on_radio \
       log_path=$LOG_PATH
   ```

### Generating Videos with GE-Sim

1. Prepare input: images, extrinsics, intrinsics, actions
2. Update config file paths in `configs/cosmos_model/acwm_cosmos.yaml`
3. Run inference script (single or multi-GPU)
4. Videos saved to output directory

## BEHAVIOR-1K Evaluation & Inference System

### BEHAVIOR-1K Environment Details

#### Robot Configuration
- **Default Robot**: R1Pro (mobile humanoid with dual arms)
- **Action Space**: 23 dimensions
  - Base: 3 dims (x, y, theta velocities)
  - Torso: 4 dims (joint positions)
  - Left arm: 7 dims
  - Left gripper: 1 dim
  - Right gripper: 1 dim
  - Right arm: 7 dims

#### Camera Configuration
Three cameras provide multi-view observations:
- **Head camera**: `robot_r1::robot_r1:zed_link:Camera:0::rgb` (720x720)
- **Left wrist**: `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` (480x480)
- **Right wrist**: `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` (480x480)

**Camera Intrinsics**:
```python
Head (720x720):     [[306.0, 0, 360],    [0, 306.0, 360],    [0, 0, 1]]
Wrist (480x480):    [[388.66, 0, 240],   [0, 388.66, 240],   [0, 0, 1]]
```

#### Proprioception
Full state vector is 256 dimensions. For GE-Act, relevant indices extracted:
- Base velocity: indices [253:256] - 3 dims
- Trunk position: indices [236:240] - 4 dims
- Left arm position: indices [158:165] - 7 dims
- Right arm position: indices [197:204] - 7 dims
- Left gripper: indices [193:195] - 2 dims (summed to 1)
- Right gripper: indices [232:234] - 2 dims (summed to 1)
- **Total extracted: 23 dims**

#### Available Tasks
BEHAVIOR-1K contains 50 household manipulation tasks:
- `turning_on_radio`, `picking_up_trash`, `putting_away_Halloween_decorations`
- `cleaning_up_plates_and_food`, `organizing_file_folders`, `serving_food`
- And 44 more tasks covering cooking, cleaning, organizing, and maintenance

#### Joint Limits (for safe operation)
- Base: [-0.75, 0.75] for x/y, [-1.0, 1.0] for theta
- Torso: [-1.13 to 1.83] (varies by joint)
- Arms: Shoulder [-4.45 to 1.31], Elbow [-0.17 to 3.14], etc.
- Grippers: [0.0, 0.05] width

### Inference Protocol & WebSocket Communication

#### Message Format (msgpack serialization)

**Client → Server (Observation)**:
```python
{
    "state": np.array([...]),              # Proprioception (22 or 23 dims)
    "obs": np.array([3, 192, 256, 3]),    # Multi-view images (V, H, W, C) uint8
    "prompt": str,                         # Task description
    "execution_step": int,                 # Temporal step indicator
}
```

**Server → Client (Action)**:
```python
{
    "actions": np.array([...]),           # Predicted action (T, action_dim)
    "server_timing": {
        "infer_ms": float,                # Inference time in ms
        "prev_total_ms": float,           # Previous cycle total time
    },
    "policy_timing": {
        ...                               # Optional policy-specific timings
    }
}
```

**Reset Signal**: Client sends `{"prompt": "<reset>..."}` to trigger policy reset

#### WebSocket Server Architecture

**Server Implementation**: `web_infer_utils/server.py` (MVActorServer)

```
┌─────────────────────────────────────────┐
│     WebSocket Async Handler             │
├─────────────────────────────────────────┤
│ 1. Accept connection                    │
│ 2. Send server metadata (msgpack)      │
│ 3. Infinite loop:                       │
│    - Receive obs from client (msgpack)  │
│    - Call policy.play() for inference   │
│    - Send actions back (msgpack)        │
│    - Track timing                       │
└─────────────────────────────────────────┘
```

**Key Features**:
- Async WebSocket using `websockets.asyncio.server`
- msgpack serialization with NumPy array support
- Health check endpoint at `/healthz`
- Compression disabled for large observations
- Connection-level error handling with traceback return

#### Client Implementation

**WebsocketClientPolicy Class**: `web_infer_utils/openpi_client/websocket_client_policy.py`

**Features**:
- Connects to server at `ws://host:port`
- Automatic server discovery with health check polling (/healthz)
- Connection retry with 5s backoff
- Support for API key authentication via headers
- Maintains persistent WebSocket connection

**Methods**:
- `infer(obs_dict)` - Send observation, receive action
- `reset()` - Trigger policy reset
- `get_server_metadata()` - Retrieve server info

#### Health Check Protocol
- Endpoint: `GET http://host:port/healthz`
- Response: `200 OK` with body `"OK\n"`
- Client polls before WebSocket connection (5s backoff on failure)

#### GE-Act Observation Processing

**Wrapper**: `web_infer_utils/ge_act_b1k_wrapper.py` (GEActB1KWrapper)

**Processing Pipeline**:
```
BEHAVIOR-1K obs dict
    ↓
_process_observation()
    ├─→ Extract 3 camera images from full resolution
    ├─→ Resize with padding to 256x256 (or 224x224 for B1K-trained)
    ├─→ Stack: (3, H, W, 3) format
    ├─→ Extract proprioception: 256→23 dims
    ├─→ Normalize state using statistics (mean/std or min/max)
    └─→ Return (images: np.uint8, state: np.float32)
         ↓
    MVActor.play() [GE-ACT model inference]
         ↓
_pad_action_to_b1k()
    ├─→ Take first 22 dims (Calvin action space)
    ├─→ Pad to 23 dims (B1K action space)
    └─→ Return torch.Tensor (1, 23)
```

### Evaluation Workflow

#### End-to-End Evaluation Setup

**Terminal 1 - Start GE-Act WebSocket Server**:
```bash
CUDA_VISIBLE_DEVICES=1 python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight /path/to/ge_act.safetensors \
    --task_name turning_on_radio \
    --port 8000 \
    --save_debug_images \
    --debug_image_dir /tmp/ge_act_debug
```

**Terminal 2 - Run BEHAVIOR-1K Evaluation**:
```bash
cd BEHAVIOR-1K/OmniGibson
CUDA_VISIBLE_DEVICES=0 python omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=/path/to/logs \
    eval_instance_ids=[0,1,2] \
    write_video=true \
    headless=true
```

#### Evaluation Script Options

**Command**: `python omnigibson/learning/eval.py`

**Key Parameters**:
- `policy`: Policy type (`websocket` or `local`)
- `task.name`: BEHAVIOR-1K task name (50 available)
- `log_path`: Output directory for metrics and videos
- `eval_on_train_instances`: If true, test on training instances; else test set
- `eval_instance_ids`: Specific instance IDs to run (list or null for all)
- `headless`: Run without GUI (true for server evaluation)
- `write_video`: Record video rollouts (resolution 448x672)
- `max_steps`: Episode timeout (default: 2x average human demo)
- `partial_scene_load`: Only load task-relevant rooms (faster)

#### Evaluation Lifecycle

1. **Initialization**:
   - Load environment config
   - Create OmniGibson simulation
   - Connect to WebSocket policy server
   - Load metrics trackers

2. **Episode Loop** (for each evaluation instance):
   - Load task instance (set objects, robot pose)
   - Reset policy and environment
   - Execute rollout

3. **Rollout Loop** (until done or max_steps):
   - Flatten observation dict
   - Compute camera relative poses
   - Send obs to WebSocket server
   - Receive action from policy
   - Execute action in simulator
   - Update metrics
   - Record video frame (if enabled)
   - Check termination/truncation

4. **Metrics Collection**:
   - **AgentMetric**: Success rate, human performance comparison
   - **TaskMetric**: Task-specific metrics
   - Written to `{log_path}/metrics/{task}_{instance}_{episode}.json`

5. **Output**:
   - Metrics JSON files
   - Video files: `{log_path}/videos/{task}_{instance}_{episode}.mp4`

#### Server Deployment Script

**Command**: `web_infer_scripts/serve_ge_act_b1k.py`

**Required Arguments**:
- `-c, --config`: Path to GE-ACT config YAML
- `-w, --weight`: Path to model weights (.safetensors)

**Task Configuration**:
- `--task_name`: BEHAVIOR-1K task (default: turning_on_radio)
- `--prompt`: Manual override of task prompt
- `--dataset_root`: Path to BEHAVIOR-1K dataset for metadata lookup

**Server Configuration**:
- `--host`: Bind address (default: 0.0.0.0 = accept from any IP)
- `-p, --port`: WebSocket port (default: 8000)

**Model Parameters**:
- `--domain_name`: Statistics domain (default: behavior1k)
- `--action_dim`: Model action dimension (default: 22 for Calvin)
- `--denoise_steps`: Diffusion denoising steps (default: 5)
- `--image_height/width`: Input resolution (default: 256x256)

**Debug Options**:
- `--save_debug_images`: Save camera frames per 50 steps
- `--debug_image_dir`: Output directory for images
- `--test_mode`: Return zero actions without model inference (for testing)

### GE-Act Config Structure (YAML)

**Example**: `configs/ltx_model/b1k/action_model_b1k.yaml`

```yaml
data:
  train:
    action_type: "delta"/"absolute"/"relative"
    action_space: "joint"
    chunk: 8                  # Context window
    action_chunk: 4           # Output action sequence length
    sample_size: [256, 256]   # Input resolution

  val:
    stat_file: "path/to/stats.json"
    valid_act_dim: 22         # Number of action dims to use
    valid_sta_dim: 22         # Number of state dims to use

diffusion_model:
  model_path: "path/to/checkpoint.safetensors"  # Video adaptation checkpoint
  config:
    action_in_channels: 22    # For add_state=True

pretrained_model_name_or_path: "../LTX-Video"  # Base model components
vae_path: "path/to/vae"       # Or embedded in pretrained model
num_inference_steps: 5        # Override via CLI --denoise_steps
```

**Important Paths**:
- **Tokenizer**: `pretrained_model_name_or_path` (usually LTX-Video path)
- **VAE**: `vae_path` or embedded in pretrained model
- **Weights**: Specified via CLI `--weight`, not in config
- **Statistics**: `data.val.stat_file` for normalization

#### Action Type Support

- **absolute**: Direct joint positions (denormalized to real values)
- **delta**: Relative changes (cumsum-accumulated over time)
- **relative**: Relative to current state (offset-based)

#### Normalization Types

- **meanstd**: Normalize using mean/std statistics from training data
- **minmax**: Normalize to [-1, 1] using min/max percentiles

#### MVActor Initialization Parameters

**Core Policy Engine**: `web_infer_utils/MVActor.py`

```python
MVActor(
    config_file="path/to/config.yaml",
    transformer_file="path/to/weights.safetensors",
    threshold=None,              # Steps before updating observation history
    n_prev=4,                    # Number of previous frames for temporal context
    action_dim=22,               # Output action dimension
    gripper_dim=1,               # Gripper dimension
    domain_name="behavior1k",    # For statistics lookup
    num_inference_steps=5,       # Diffusion steps (more = slower but better)
)
```

**Components Loaded**:
1. Text Encoder - Tokenizes task description
2. VAE - Video encoding/decoding
3. Diffusion Model - Transformer for action prediction
4. Scheduler - Diffusion process controller
5. Pipeline - Coordinates inference

## WebSocket Deployment Architecture

For real-world deployment, GE-Act uses a client-server model:

1. **Server** (`web_infer_scripts/main_server.py` or `serve_ge_act_b1k.py`):
   - Loads model weights and creates inference wrapper
   - Starts WebSocket server listening for observations
   - Processes observations through MVActor and returns actions

2. **Client** (in robot control code):
   - Connects to server via WebSocket
   - Sends observations (images + proprioception) as msgpack
   - Receives actions as tensors
   - Executes actions on robot

3. **Adapters** (e.g., `ge_act_b1k_wrapper.py`):
   - Transform domain-specific observations to GE-Act format
   - Handle action space conversions (e.g., padding, clipping)

See `docs/BEHAVIOR1K_INTEGRATION.md` for detailed BEHAVIOR-1K integration example.

## Troubleshooting

### Training Issues

**"You are not loading the pretrained weights"** warning:
- Ensure `load_weights: true` in your config file

**NaN loss detected**:
- Check action statistics file has correct dimensions matching your dataset
- Verify `action_in_channels` matches your action dimension
- Reduce learning rate

**OOM (Out of Memory)**:
- Enable `gradient_checkpointing: True`
- Reduce `batch_size`
- Use DeepSpeed ZeRO Stage 2 or 3

**Poor action predictions / high validation loss**:
- Verify statistics file path is correct in config
- Check `action_type` matches how actions are stored in your dataset
- Ensure `domains` name in config matches key prefix in statistics file
- For Calvin: use `action_key: "relative_action"`, `add_state: True`

**Training seems stuck / no progress**:
- Check `steps_to_log` to see tensorboard updates
- Verify dataset is loading correctly (check printed episode counts)
- Ensure `load_weights: true` to start from pretrained checkpoint

**Checkpoint loading errors (missing/unexpected keys)**:
- Mismatched model architecture: verify `action_expert`, `action_in_channels` match checkpoint
- Shape mismatches are printed but ignored by default (`ignore_mismatched_sizes=True`)

### Deployment Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Connection refused** | Server not running | Start server first, verify port with `netstat -tuln \| grep 8000` |
| **"Still waiting for server"** | Health check failing | Check server logs, verify `/healthz` endpoint accessible |
| **Action dimension mismatch** | Model expects different dim | Check `action_dim` matches model's `valid_act_dim` in config |
| **Image resizing errors** | PIL import missing | Install Pillow: `pip install Pillow` |
| **Memory OOM** | Batch processing too large | Reduce `num_inference_steps` or use smaller `sample_size` |
| **Prompt not found** | Dataset metadata unavailable | Provide `--prompt` directly via CLI to override |
| **Videos not saving** | Permissions or codec issues | Check `log_path` exists and is writable, install ffmpeg |
| **Slow inference** | Too many diffusion steps | Reduce `--denoise_steps` to 5 or lower (trade quality for speed) |
| **Action jitter/instability** | Wrong normalization stats | Verify `stat_file` matches the domain and action_type |
| **Robot collision** | Joint limits exceeded | Check action clipping in wrapper, verify statistics file |
| **WebSocket disconnect** | Inference timeout | Increase timeout in client, reduce model complexity |

### Debug Workflow

1. **Test server without client**:
   ```bash
   curl http://localhost:8000/healthz
   # Should return: OK
   ```

2. **Test with simple client**:
   ```bash
   python web_infer_scripts/simple_client.py --host localhost --port 8000
   ```

3. **Enable debug image saving**:
   ```bash
   --save_debug_images --debug_image_dir /tmp/debug
   # Check images to verify observation processing
   ```

4. **Use test mode (no model)**:
   ```bash
   --test_mode
   # Returns zero actions, tests infrastructure only
   ```

5. **Check server logs**:
   - Look for model loading errors
   - Verify statistics file loaded correctly
   - Check for CUDA errors or OOM

### Performance Optimization

- **Reduce latency**: Lower `num_inference_steps` (5 is good default)
- **Reduce memory**: Use smaller `sample_size` (224x224 vs 256x256)
- **Batch evaluation**: Run multiple instances in parallel (separate GPUs)
- **Faster simulation**: Use `partial_scene_load=true` in eval config

## Important Notes

- Model checkpoints use `.safetensors` format
- Training uses mixed precision (bf16) by default
- VAE encoding/decoding uses tiling and slicing to reduce memory
- The system supports gradient checkpointing for large models
- Multi-node training requires setting distributed environment variables
- Pretrained weights are hosted on HuggingFace (agibot-world/Genie-Envisioner) and ModelScope
- LTX-Video base weights required: text_encoder, tokenizer, vae, and optionally transformer
- Model outputs are in latent space, decoded by VAE during inference

## File Paths to Modify

When setting up for training or inference, update these paths in configs:

1. `pretrained_model_name_or_path`: Directory containing VAE, tokenizer, text_encoder
2. `diffusion_model.model_path`: Path to .safetensors checkpoint
3. `data.train/val.data_roots`: List of dataset root directories
4. `data.train/val.stat_file`: Path to action statistics JSON
5. `output_dir`: Where to save checkpoints and logs
6. `data.train/val.dataset_info_cache_path`: Cache for dataset metadata (AgiBotWorld only)

## Key Configuration Files

- `configs/ltx_model/video_model_lerobot.yaml`: Video adaptation config
- `configs/ltx_model/policy_model_lerobot.yaml`: Action training config
- `configs/ltx_model/video_model_infer_slow.yaml`: Video generation config
- `configs/cosmos_model/acwm_cosmos.yaml`: GE-Sim config

## Model Weights Locations

- **GE-Base weights**: Download to local, specify in config `diffusion_model.model_path`
- **LTX-Video components**: `../LTX-Video/` (text_encoder, tokenizer, vae, model_index.json)
- **Cosmos components**: `../Cosmos-Predict2-2B-Video2World/` (scheduler, text_encoder, tokenizer, vae)
- Specify pretrained model path in config: `pretrained_model_name_or_path`

## File Structure Reference

```
Genie-Envisioner/
├── web_infer_scripts/
│   ├── serve_ge_act_b1k.py        # B1K-specific deployment script
│   ├── simple_client.py            # Basic test client
│   └── run_server.sh               # Shell wrapper for server startup
│
├── web_infer_utils/
│   ├── server.py                   # MVActorServer (async WebSocket)
│   ├── MVActor.py                  # Core policy inference engine
│   ├── ge_act_b1k_wrapper.py       # BEHAVIOR-1K observation adapter
│   └── openpi_client/
│       ├── websocket_client_policy.py    # Client implementation
│       └── msgpack_numpy.py              # Serialization utils

BEHAVIOR-1K/OmniGibson/
└── omnigibson/learning/
    ├── eval.py                      # Main evaluation script
    ├── policies.py                  # LocalPolicy, WebsocketPolicy
    └── configs/
        ├── base_config.yaml        # Default evaluation config
        ├── policy/websocket.yaml   # WebSocket policy config
        └── task/behavior.yaml      # BEHAVIOR-1K task config
```

## Related Documentation

- Main README: `README.md`
- Simulation benchmark evaluation: `experiments/RUN.md`
- BEHAVIOR-1K integration: `docs/BEHAVIOR1K_INTEGRATION.md`
- Example results: `video_gen_examples/`, `gesim_video_gen_examples/`

## Reference Directories

- **BEHAVIOR-1K/**: Simulation benchmark - refer to for evaluation environment details if needed
- **b1k-baselines/**: Baseline implementations - refer to for comparison or examples if needed
- **calvin/**: CALVIN benchmark - refer to if needed
