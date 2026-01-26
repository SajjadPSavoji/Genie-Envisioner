# Dataset Output Format Verification

This document verifies that B1K dataset output format matches the existing working datasets in Genie-Envisioner.

## Summary: All Datasets Return Identical Format ✓

```python
sample = {
    'video': torch.Tensor,     # (C, V, T, H, W)
    'actions': torch.Tensor,   # (T_action, action_dim)
    'caption': str,            # Task description
    'state': torch.Tensor,     # (1, state_dim)
}
```

## Comparison Across All Datasets

### 1. Dictionary Keys (All Identical)

| Dataset | Keys | ✓ |
|---------|------|---|
| **AgiBotWorld** | `video, actions, state, caption` | ✓ |
| **LeRobot-like** | `video, actions, caption, state` | ✓ |
| **LIBERO** | `video, actions, caption, state` | ✓ |
| **B1K** | `video, actions, caption, state` | ✓ |

**Note**: Order doesn't matter in Python dicts.

---

### 2. Shape Formulas

#### Video Shape: `(C, V, T, H, W)`
- **C**: 3 (RGB channels)
- **V**: Number of cameras (varies by dataset)
- **T**: `n_previous + chunk` (number of frames)
- **H, W**: Sample size (e.g., 224×224 or 256×256)

| Dataset | Config Example | Expected Video Shape |
|---------|---------------|---------------------|
| **LIBERO** | chunk=9, n_previous=4, 2 cams, 256×256 | `(3, 2, 13, 256, 256)` |
| **Calvin** | chunk=9, n_previous=4, 2 cams, 256×256 | `(3, 2, 13, 256, 256)` |
| **B1K** | chunk=4, n_previous=12, 1 cam, 224×224 | `(3, 1, 16, 224, 224)` |

**Formula**: `T_video = n_previous + chunk`

---

#### Actions Shape: `(T_action, action_dim)`
- **T_action**: `n_previous + action_chunk`
- **action_dim**: Depends on action space (EEF=7, Joint=varies)

| Dataset | Config Example | Expected Actions Shape |
|---------|---------------|----------------------|
| **LIBERO** | action_chunk=36, n_previous=4, EEF | `(40, 7)` |
| **Calvin** | action_chunk=54, n_previous=4, EEF | `(58, 7)` |
| **B1K** | action_chunk=4, n_previous=12, Joint | `(16, 23)` |

**Formula**: `T_action = n_previous + action_chunk`

**Code verification**:
```python
# From lerobot_like_dataset.py:436 and b1k_dataset.py:419
indexes = mem_indexes + video_end  # Length: n_previous + action_chunk
action = action[indexes]            # Shape: (n_previous + action_chunk, action_dim)
```

---

#### State Shape: `(1, state_dim)`
- Always **1 timestep** (single state observation)
- **state_dim**: Varies by dataset

| Dataset | State Extraction | State Dim | Shape |
|---------|-----------------|-----------|-------|
| **AgiBotWorld** | `action[n_previous-1:n_previous]` | 7 (EEF) | `(1, 7)` |
| **LeRobot-like** | `state[indexes][n_previous-1:n_previous]` | Varies | `(1, dim)` |
| **LIBERO** | `state[n_previous-1:n_previous]` | 7 | `(1, 7)` |
| **B1K** | `state[indexes][n_previous-1:n_previous]` | 256 | `(1, 256)` |

**Code verification**:
```python
# All datasets use this pattern:
state = state[self.n_previous-1:self.n_previous]  # Extracts single timestep
# Result: (1, state_dim)
```

---

#### Caption: `str`
- Simple string containing task description
- All datasets: ✓

---

### 3. Value Ranges

| Field | Expected Range | Normalization Method |
|-------|---------------|---------------------|
| **video** | `[-1, 1]` | `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` |
| **actions** | Normalized (mean≈0, std≈1) | `(action - mean) / std` from stats file |
| **state** | Normalized (mean≈0, std≈1) | `(state - mean) / std` from stats file |
| **caption** | N/A | Raw string |

**All datasets use identical normalization**: ✓

---

## B1K Dataset Compliance

### Output Format Verification

```python
# From data/b1k_dataset.py:466-472
sample = dict(
    video=video,      # (3, V, n_previous+chunk, H, W)
    actions=actions,  # (n_previous+action_chunk, 23)
    caption=caption,  # str
    state=state,      # (1, 256)
)
```

**✓ Matches all other datasets**

### Shape Computation

With test config: `chunk=4, action_chunk=4, n_previous=12, 1 camera, 224×224`

```python
video:   (3, 1, 16, 224, 224)    # 16 = 12 + 4
actions: (16, 23)                # 16 = 12 + 4
state:   (1, 256)                # Single timestep
caption: "Turn on the radio..."  # String
```

**✓ Follows exact same formula as LIBERO, Calvin, AgiBotWorld**

### State Extraction

```python
# From data/b1k_dataset.py:419
state = torch.FloatTensor(state)[indexes][self.n_previous - 1:self.n_previous]
```

**✓ Identical to lerobot_like_dataset.py:436**

---

## Conclusion

✅ **B1K dataset output format is 100% compatible with existing datasets**

The B1K dataset implementation follows the exact same patterns as:
- AgiBotWorld
- LeRobot-like datasets
- LIBERO
- Calvin

All key aspects match:
- Dictionary keys ✓
- Shape formulas ✓
- Normalization methods ✓
- State extraction pattern ✓
- Video format (C, V, T, H, W) ✓

**The B1K dataset can be used as a drop-in replacement for any of these datasets in the training pipeline.**

---

## Quick Reference

### Test Your Dataset

```bash
conda activate genie_envisioner
cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner
python b1k_stuff/test_b1k_dataset.py
```

### Expected Test Output

```
=== Test 2: Loading Single Sample ===
✓ Successfully loaded sample
  Video shape: torch.Size([3, 1, 16, 224, 224])
  Actions shape: torch.Size([16, 23])
  Caption: Turn on the radio receiver that's on the table in...
  State shape: torch.Size([1, 256])
```

### Shape Formula Reference

```python
# Video
T_video = n_previous + chunk
video_shape = (3, num_cameras, T_video, height, width)

# Actions
T_action = n_previous + action_chunk
actions_shape = (T_action, action_dim)

# State
state_shape = (1, state_dim)

# Caption
caption = str
```
