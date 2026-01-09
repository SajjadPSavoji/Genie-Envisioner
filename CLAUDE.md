# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genie Envisioner is a unified world foundation platform for robotic manipulation that consists of two main components:

1. **GE-Sim**: World model for video generation conditioned on actions (using Cosmos2 or LTX-Video backbones)
2. **GE-Act**: Action prediction model for robotic control (policy model)

The system uses diffusion-based transformers and supports multiple robotic datasets including AgiBotWorld, LIBERO, Calvin, and BEHAVIOR-1K.

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
python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos_multigpu.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=results/
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

Calculate with:
```bash
# data_root should point to the folder containing parquet files (the data/ subfolder)
python scripts/get_statistics.py \
    --data_root /path/to/dataset/data \
    --data_name DATASETNAME \
    --data_type joint \
    --action_key action \
    --state_key observation.state \
    --save_path /path/to/stats.json
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

### Dataset Format

The system supports LeRobot-format datasets with this structure:

```
DATASETNAME/
├── data/
│   └── episode_XXXXXX.parquet
├── meta/
│   ├── episodes_stats.jsonl
│   ├── episodes.jsonl
│   ├── tasks.json
│   └── info.json
└── videos/
    └── chunk-XXX/
        └── observation.images.CAMERA_NAME/
            └── episode_XXXXXX.mp4
```

Key requirements:
- Multi-view camera support (configure in `valid_cam` list)
- Action normalization via statistics files (JSON with mean/std per domain and data type)
- Supports absolute, delta, and relative action spaces

### Action Space Handling

- Actions and states are normalized using pre-computed statistics
- Statistics files contain mean/std for: `{DATASET}_{joint|eef}`, `{DATASET}_delta_{joint|eef}`, `{DATASET}_state_{joint|eef}`
- Action dimensions vary by domain (e.g., Calvin: 22 dims, BEHAVIOR-1K: 23 dims)
- The action expert module in the transformer outputs action predictions in the latent space

### GE-Sim (World Model) Architecture

GE-Sim generates future video frames conditioned on:
- Initial frames (sparse memory frames, typically 4 frames)
- Camera extrinsics (4x4 camera-to-base transformation matrices)
- Camera intrinsics (fx, fy, cx, cy)
- Action trajectories (robot joint/end-effector actions)

Input format for `infer_gesim.py`:
- Images: Multi-view frames as PNG/JPG
- Extrinsics: `.npy` files with shape `(num_frames, 4, 4)`
- Intrinsics: `.npy` files with shape `(num_cameras, 4)` or `(num_frames, num_cameras, 4)`
- Actions: `.npy` file with shape `(num_frames, action_dim)`

### WebSocket Deployment Architecture

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

## Troubleshooting Training Issues

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

## Related Documentation

- Main README: `README.md`
- Simulation benchmark evaluation: `experiments/RUN.md`
- BEHAVIOR-1K integration: `docs/BEHAVIOR1K_INTEGRATION.md`
- Example results: `video_gen_examples/`, `gesim_video_gen_examples/`
