# CLAUDE.md

## Overview

**Genie-Envisioner**: Unified world foundation platform for robotic manipulation.
- **GE-Act**: Action prediction model (policy) based on LTX-Video
- **GE-Sim**: Video generation world model based on Cosmos2 or LTX-Video
- **GE-Base**: Foundation video generation model

Supports: AgiBotWorld, LIBERO, Calvin, BEHAVIOR-1K datasets.

**Related directories**: `BEHAVIOR-1K/`, `calvin/`, `LTX-Video/`, `Cosmos-Predict2-2B-Video2World/`

## Code Protocol

- Verify code correctness before committing (training runs are expensive)
- Test with temp files: `test_temp.py` → run → delete
- Ask clarifying questions when requirements are ambiguous
- Activate env: `conda activate genie_envisioner`

## Directory Structure

```
configs/           # YAML configs (ltx_model/, cosmos_model/)
data/              # Dataset loaders
models/            # Architectures (ltx_models/, cosmos_models/, pipeline/)
runner/            # ge_trainer.py, ge_inferencer.py
scripts/           # Training scripts
web_infer_scripts/ # Deployment (serve_ge_act_b1k.py)
web_infer_utils/   # MVActor.py, server.py, ge_act_b1k_wrapper.py
```

## Commands

### Training
```bash
bash scripts/train.sh main.py configs/ltx_model/video_model.yaml          # Video pretraining
bash scripts/train.sh main.py configs/ltx_model/policy_model_lerobot.yaml # Action training
NGPU=8 bash scripts/train.sh main.py CONFIG.yaml                          # Multi-GPU
```

### Inference
```bash
bash scripts/infer.sh main.py CONFIG.yaml CHECKPOINT.safetensors OUTPUT_PATH DATASETNAME
```

### Dataset Statistics
```bash
python scripts/get_statistics.py --data_root PATH/data --data_name NAME --data_type joint \
    --action_key action --state_key observation.state --save_path stats.json
```

### BEHAVIOR-1K Server
```bash
CUDA_VISIBLE_DEVICES=1 python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight checkpoint.safetensors --task_name "task_name" --port 8000
```

## Training Stages

1. **GE-Base Pretraining**: `video_model.yaml` → train on AgiBotWorld
2. **Video Adaptation**: `video_model_lerobot.yaml` → finetune on target domain
3. **Action Post-Training**: `policy_model_lerobot.yaml` → train action expert

### Training Modes

| Mode | `train_mode` | `return_action` | `return_video` | `action_expert` | Trains |
|------|-------------|----------------|---------------|----------------|--------|
| Video | `video_only` | false | true | false | All non-action |
| Action Only | `action_only` | true | false | true | Only action params |
| Action Full | `action_full` | true | false | true | **All params** |

**Note**: `action_full` trains ALL parameters. Use `action_only` to freeze video backbone.

## Key Config Parameters

```yaml
pretrained_model_name_or_path: /path/to/ltx-video  # VAE, tokenizer, text_encoder
diffusion_model:
  model_path: /path/to/checkpoint.safetensors
  config:
    action_expert: true/false
    action_in_channels: 14  # 14=AgiBotWorld, 22=Calvin, 7=LIBERO
train_mode: 'video_only'/'action_only'/'action_full'
return_action: true/false
return_video: true/false
data:
  train:
    chunk: 9                    # Video frames to predict
    action_chunk: 54            # Actions to predict (multiple of chunk)
    n_previous: 4               # Memory/context frames
    action_type: "delta"        # absolute/delta/relative
    action_space: "joint"       # joint/eef
    stat_file: /path/stats.json
load_weights: true              # MUST be true to load pretrained
```

## Dataset Format (LeRobot)

```
DATASET/
├── data/episode_*.parquet
├── meta/{episodes_stats.jsonl, tasks.json, info.json}
└── videos/chunk-000/observation.images.*/episode_*.mp4
```

**Statistics JSON**:
```json
{"DATASET_delta_joint": {"mean": [...], "std": [...], "q99": [...], "q01": [...]}}
```

## BEHAVIOR-1K

**Robot**: R1Pro, 23-dim action (base:3, torso:4, arms:14, grippers:2)

**Cameras**: Head (720x720), Left/Right wrist (480x480)

**Proprioception extraction** (256→23 dims):
- Base: [253:256], Trunk: [236:240], Arms: [158:165]+[197:204], Grippers: [193:195]+[232:234]

### WebSocket Protocol (msgpack)

**Client→Server**: `{state, obs, prompt, execution_step}`
**Server→Client**: `{actions, server_timing, policy_timing}`
**Reset**: `{"prompt": "<reset>..."}`

### Evaluation
```bash
# Terminal 1: Server
python web_infer_scripts/serve_ge_act_b1k.py --config CONFIG --weight WEIGHTS --port 8000

# Terminal 2: Client
python BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=TASK log_path=LOGS
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Not loading weights | Set `load_weights: true` |
| NaN loss | Check stats file dims, reduce LR |
| OOM | Enable `gradient_checkpointing`, reduce `batch_size` |
| Connection refused | Start server, check port |
| Action mismatch | Verify `action_in_channels`, `action_dim` |
| Slow inference | Reduce `num_inference_steps` to 5 |

## Memory Frames

Sparse historical frames for temporal coherence. Configured via `n_previous` (typically 4).
- Selected uniformly or randomly from history
- Stay fixed during diffusion (only future frames predicted)
- Encoded separately by VAE

## Key Files

- Training: `runner/ge_trainer.py`
- Inference: `runner/ge_inferencer.py`, `models/pipeline/custom_pipeline.py`
- Deployment: `web_infer_utils/MVActor.py`, `web_infer_utils/server.py`
- B1K adapter: `web_infer_utils/ge_act_b1k_wrapper.py`
- Datasets: `data/agibotworld_dataset.py`, `data/lerobot_like_dataset.py`
