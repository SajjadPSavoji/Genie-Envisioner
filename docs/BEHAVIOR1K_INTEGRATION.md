# GE-ACT + BEHAVIOR-1K Integration Walkthrough

## Architecture Overview

The integration uses a **client-server architecture** where BEHAVIOR-1K (simulation) sends observations to a GE-ACT server and receives actions back.

```
┌─────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ BEHAVIOR-1K │         │ WebSocket Client │         │ WebSocket Server │
│ (Simulator) │         │   (eval.py)      │         │(serve_ge_act.py) │
└──────┬──────┘         └────────┬─────────┘         └────────┬─────────┘
       │                         │                            │
       │ obs (images + proprio)  │                            │
       ├────────────────────────>│                            │
       │                         │                            │
       │                         │ WebSocket msg (msgpack)    │
       │                         ├───────────────────────────>│
       │                         │                            │
       │                         │                    ┌───────▼────────┐
       │                         │                    │ GEActB1KWrapper│
       │                         │                    └───────┬────────┘
       │                         │                            │
       │                         │                    Process obs →
       │                         │                    (3, H, W, 3)
       │                         │                            │
       │                         │                    ┌───────▼────────┐
       │                         │                    │    MVActor     │
       │                         │                    │  (GE-ACT Model)│
       │                         │                    └───────┬────────┘
       │                         │                            │
       │                         │                    action (22 dims)
       │                         │                            │
       │                         │                    ┌───────▼────────┐
       │                         │                    │ Extract → Pad  │
       │                         │                    │  16 → 23 dims  │
       │                         │                    └───────┬────────┘
       │                         │                            │
       │                         │ action tensor (1, 23)      │
       │                         │<───────────────────────────┤
       │                         │                            │
       │ action tensor (1, 23)   │                            │
       │<────────────────────────┤                            │
       │                         │                            │
  Execute action                 │                            │
  in simulator                   │                            │
       │                         │                            │
```


---

## How It Works

### Terminal 1: GE-ACT Server (GPU 1)

**Script:** [serve_ge_act_b1k.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_scripts/serve_ge_act_b1k.py)

1. Loads GE-ACT model weights
2. Creates `GEActB1KWrapper` with the model
3. Starts `WebsocketPolicyServer` on port 8000
4. Waits for connections from BEHAVIOR-1K

### Terminal 2: BEHAVIOR-1K Evaluation (GPU 0)

**Script:** `OmniGibson/omnigibson/learning/eval.py`

1. Loads simulation environment and robot
2. Creates `WebsocketClientPolicy` → connects to server
3. **Each step:**
   - Get observation from simulator (3 cameras + proprio)
   - Send to server via WebSocket
   - Receive action (23 dims)
   - Execute action in simulator

---

## Initialization Call Chain

When you run `serve_ge_act_b1k.py`, here's what gets initialized:

```
serve_ge_act_b1k.py:main() (Line 168)
    │
    ├── GEActB1KWrapper(config, weights, prompt, ...)  → Line 181
    │       │
    │       └── ge_act_b1k_wrapper.py:__init__() (Line 58)
    │               │
    │               └── MVActor(config, weights, ...)  → Line 95  ← MODEL LOADED HERE
    │                       │
    │                       └── MVActor.py:__init__() 
    │                               │
    │                               └── Loads transformer weights from .safetensors
    │
    └── WebsocketPolicyServer(policy=wrapper)  → Line 201
            │
            └── network_utils.py (from BEHAVIOR-1K)
                    │
                    └── serve_forever()  → Starts listening on port 8000
```

**Key files involved:**

| File | Location | Role |
|------|----------|------|
| [serve_ge_act_b1k.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_scripts/serve_ge_act_b1k.py) | `web_infer_scripts/` | Entry point, creates wrapper + server |
| [ge_act_b1k_wrapper.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_utils/ge_act_b1k_wrapper.py) | `web_infer_utils/` | Adapts B1K obs → GE-ACT format |
| [MVActor.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_utils/MVActor.py) | `web_infer_utils/` | Core GE-ACT inference logic |
| [network_utils.py](file:///shared_work/physical_intelligence/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/network_utils.py) | `BEHAVIOR-1K/` | WebSocket server implementation |

---

## Data Flow Detail

### Observation (B1K → GE-ACT)

| Field | Source | Shape | Description |
|-------|--------|-------|-------------|
| Head camera | `robot_r1::robot_r1:zed_link:Camera:0::rgb` | (224, 224, 3) | RGB image (resized by RGBLowResWrapper) |
| Left wrist | `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | (224, 224, 3) | RGB image (resized by RGBLowResWrapper) |
| Right wrist | `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | (224, 224, 3) | RGB image (resized by RGBLowResWrapper) |
| Proprio | `robot_r1::proprio` | (256,) | Joint states |

> [!NOTE]
> Native camera resolutions are 720x720 (head) and 480x480 (wrists), but BEHAVIOR-1K's `RGBLowResWrapper` resizes all images to 224x224 before sending via WebSocket.

**Wrapper processing:** Resize images → stack to (3, 192, 256, 3) → normalize

### Action (GE-ACT → B1K)

| Step | Shape | Description |
|------|-------|-------------|
| GE-ACT output | (22,) | 16 action dims + 6 state dims |
| Extract actions | (16,) | First 16 dims only |
| Pad for B1K | (23,) | Zero-pad to match R1Pro robot |

---

## Files Created

| File | Purpose |
|------|---------|
| [ge_act_b1k_wrapper.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_utils/ge_act_b1k_wrapper.py) | Processes B1K observations → GE-ACT format |
| [serve_ge_act_b1k.py](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/web_infer_scripts/serve_ge_act_b1k.py) | Starts WebSocket server with GE-ACT policy |
| [action_model_b1k.yaml](file:///shared_work/physical_intelligence/ruiheng/Genie-Envisioner/configs/ltx_model/b1k/action_model_b1k.yaml) | Model config (action dims, image size) |

---

## Verification Steps

### Step 0: Verify BEHAVIOR-1K Installation
```bash
cd /shared_work/physical_intelligence/BEHAVIOR-1K/OmniGibson
conda activate behavior
python -m omnigibson.examples.robots.robot_control_example --quickstart
```

### Step 1: Start GE-ACT Server (Test Mode)

**Terminal 1** - GPU 1:
```bash
# 1. Navigate to GE-ACT directory
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner

# 2. Activate conda environment
conda activate genie_envisioner

# 3. Start server
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

The server auto-creates a timestamped directory and a `ge_act_latest` symlink.

**Wait for the server to print:**
```
================================================================================
OUTPUT DIRECTORY:
  Actual: /shared_work/physical_intelligence/ruiheng/tmp/ge_act_20260109_120841
  Symlink: /shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest

For BEHAVIOR-1K, use:
  log_path=/shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest
================================================================================
Server ready! Listening on 0.0.0.0:8000
```


### Step 2: Run BEHAVIOR-1K Evaluation

**Terminal 2** - GPU 0:

**IMPORTANT:** Start the server (Step 1) first and wait for "Server ready!" before running this.

Use `ge_act_latest` as `log_path` (symlink always points to the latest run):

```bash
# 1. Navigate to BEHAVIOR-1K directory
cd /shared_work/physical_intelligence/BEHAVIOR-1K

# 2. Activate conda environment
conda activate behavior

# 3. Run evaluation (with GUI visible)
CUDA_VISIBLE_DEVICES=0 python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    log_path=/shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest \
    headless=false
```

**Note:** Remove `headless=false` to run without GUI (faster, less resource intensive).

**Expected behavior:**
- Simulation window opens showing the robot and environment
- Robot receives zero actions (stays still in test mode)
- Episode completes
- All outputs saved to the same directory

### Step 3: Check Outputs

```bash
# Check the output directory
ls -lh /shared_work/physical_intelligence/ruiheng/tmp/ge_act_latest/
```

**Expected structure:**
```
ge_act_latest/ -> ge_act_20260109_115808/
├── images/                     # From GE-ACT server
│   ├── head/
│   │   ├── step_0000_raw.png
│   │   └── ...
│   ├── left_wrist/
│   └── right_wrist/
├── videos/                     # From BEHAVIOR-1K
│   └── turning_on_radio_181_0.mp4
├── metrics/                    # From BEHAVIOR-1K
│   └── turning_on_radio_181_0.json
└── ge_act_metadata.json        # From GE-ACT server
```




### Step 4: Run with Actual Model (Remove Test Mode)

Once test mode works, remove `--test_mode` flag to run actual GE-ACT inference:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n genie_envisioner python web_infer_scripts/serve_ge_act_b1k.py \
    --config configs/ltx_model/b1k/action_model_b1k.yaml \
    --weight /shared_work/physical_intelligence/ge_weights/ge_act_calvin.safetensors \
    --task_name "assembling_gift_baskets" \
    --host 0.0.0.0 \
    --port 8000 \
    --save_debug_images \
    --debug_image_dir /shared_work/physical_intelligence/ruiheng/tmp/ge_act_debug_images
```


---

## Known Limitations

> [!WARNING]
> **Model not trained on B1K:** Calvin model outputs may not produce meaningful actions for B1K tasks.

> [!NOTE]
> **Action space mismatch:** Calvin uses 22-dim actions, B1K R1Pro uses 23-dim joint actions. Actions are padded with zeros.
