"""
BEHAVIOR-1K Dataset for Genie-Envisioner Training.

Adapts the LeRoBot-style dataset interface for B1K's specific directory structure:
- data/task-XXXX/episode_XXXXXXXX.parquet
- videos/task-XXXX/camera_name/episode_XXXXXXXX.mp4  
- annotations/task-XXXX/episode_XXXXXXXX.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

from utils import zero_rank_print


def load_task_descriptions(tasks_jsonl_path: str) -> dict:
    """Load task descriptions from tasks.jsonl.

    Returns:
        dict mapping task_index to task description string
    """
    task_descriptions = {}
    with open(tasks_jsonl_path, 'r') as f:
        for line in f:
            task_data = json.loads(line.strip())
            task_idx = task_data['task_index']
            task_desc = task_data['task']
            task_descriptions[task_idx] = task_desc
    return task_descriptions


class B1KDataset(Dataset):
    """BEHAVIOR-1K dataset for Genie-Envisioner training.
    
    Compatible with the CustomLeRobotDataset interface used in GE training.
    """
    
    # B1K camera names
    B1K_CAMERAS = [
        'observation.images.rgb.head',
        'observation.images.rgb.left_wrist', 
        'observation.images.rgb.right_wrist',
    ]
    
    # Proprioceptive feature indices from 256-dim state to 23-dim filtered state
    # Maps original indices in 256-dim state to filtered 23-dim state positions
    PROPRIO_INDICES_256 = {
        'base_qvel': list(range(253, 256)),           # indices 0-2 in filtered state (3 dims)
        'trunk_qpos': list(range(236, 240)),          # indices 3-6 in filtered state (4 dims)
        'left_arm_qpos': list(range(158, 165)),       # indices 7-13 in filtered state (7 dims)
        'left_gripper_qpos': list(range(193, 195)),   # indices 14 in filtered state (2 -> 1 summed)
        'right_arm_qpos': list(range(197, 204)),      # indices 15-21 in filtered state (7 dims)
        'right_gripper_qpos': list(range(232, 234)),  # index 22 in filtered state (2 -> 1 summed)
    }
    
    # Selective delta indices (which dimensions compute delta vs absolute)
    # Delta indices (17 dims): torso, left_arm, right_arm
    DELTA_ACTION_INDICES = list(range(3, 7)) + list(range(7, 14)) + list(range(15, 22))
    # Absolute indices (5 dims): base, left_gripper, right_gripper
    ABSOLUTE_ACTION_INDICES = [0, 1, 2, 14, 22]
    # Same for state
    DELTA_STATE_INDICES = list(range(3, 7)) + list(range(7, 14)) + list(range(15, 22))
    ABSOLUTE_STATE_INDICES = [0, 1, 2, 14, 22]
    
    def __init__(
        self,
        data_roots,
        domains,
        task_recap_file=None,
        step_recap_file=None,
        sample_size=(224, 224),  # B1K uses 224x224 (matches vision transformer input)
        sample_n_frames=64,
        preprocess='resize',
        valid_cam=['observation.images.rgb.head', 'observation.images.rgb.left_wrist', 'observation.images.rgb.right_wrist'],  # B1K-specific cameras
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path=None,
        action_type="absolute",
        action_space="joint",
        ignore_seek=False,
        train_dataset=True,
        action_key="action",
        state_key="observation.state",
        use_unified_prompt=False,
        unified_prompt="best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx=None,
        fix_sidx=None,
        fix_mem_idx=None,
        stat_file=None,
        task_ids=None,
    ):
        """
        Args:
            data_roots: List containing B1K root directory
                       (e.g., ["/shared_work/physical_intelligence/2025-challenge-demos"])
            domains: List of domain names (e.g., ["behavior1k"])
            sample_size: Output frame size (H, W)
            valid_cam: Camera names to use. If None, uses all 3 RGB cameras.
            chunk: Number of video frames to predict
            action_chunk: Number of actions to predict
            n_previous: Number of memory/context frames
            stat_file: Path to normalization statistics JSON file
            task_ids: Optional list of task IDs to load (e.g., [0, 1, 5]). If None, loads all tasks.
        """
        zero_rank_print(f"Loading BEHAVIOR-1K dataset...")
        
        assert action_type in ["delta", "absolute", "relative"]
        self.action_type = action_type
        assert action_space in ["eef", "joint"]
        self.action_space = action_space
        
        self.action_key = action_key
        self.state_key = state_key
        self.random_crop = random_crop
        
        # Camera configuration - use B1K cameras if not specified
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam]
        self.valid_cam = valid_cam
        
        self.data_roots = data_roots if isinstance(data_roots, list) else [data_roots]
        self.domains = domains if isinstance(domains, list) else [domains]
        self.task_ids = task_ids

        # Load task descriptions from tasks.jsonl
        self.task_descriptions = {}
        for data_root in self.data_roots:
            tasks_jsonl_path = Path(data_root) / "meta" / "tasks.jsonl"
            if tasks_jsonl_path.exists():
                self.task_descriptions.update(load_task_descriptions(str(tasks_jsonl_path)))
                zero_rank_print(f"Loaded task descriptions from {tasks_jsonl_path}")
            else:
                zero_rank_print(f"Warning: tasks.jsonl not found at {tasks_jsonl_path}")

        # Build dataset index
        self.dataset = []
        self._build_dataset_index(dataset_info_cache_path)
        
        self.length = len(self.dataset)
        zero_rank_print(f"B1K dataset size: {self.length} episodes")
        
        # Frame/action chunk configuration
        self.chunk = chunk
        self.action_chunk = action_chunk if action_chunk else chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert self.chunk * self.video_temporal_stride == self.action_chunk
        
        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size
        
        # Transforms
        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        else:  # 'resize'
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess
        
        # Memory frame configuration
        if n_previous > 1:
            self.n_previous = int(n_previous)
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = int(self.sample_n_frames - self.chunk)
            self.previous_pick_mode = 'uniform'
        
        # Load statistics
        self.StatisticInfo = {}
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)
            zero_rank_print(f"Loaded statistics from {stat_file}")
        
        # Caption augmentation
        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None
        
        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None
        
        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt
        
        self.ignore_seek = ignore_seek
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx
    
    def _build_dataset_index(self, cache_path=None):
        """Build index of all episodes with their metadata."""
        
        if cache_path and os.path.exists(cache_path):
            zero_rank_print(f"Loading cached dataset index from {cache_path}")
            with open(cache_path, 'r') as f:
                self.dataset = json.load(f)
            return
        
        for data_root, domain_name in zip(self.data_roots, self.domains):
            data_root = Path(data_root)
            
            zero_rank_print(f"Scanning B1K episodes in {data_root}...")
            
            parquet_dir = data_root / "data"
            video_dir = data_root / "videos"
            annotation_dir = data_root / "annotations"
            
            # Iterate through task folders
            for task_dir in sorted(parquet_dir.iterdir()):
                if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
                    continue

                # Filter by task_ids if specified
                task_num = int(task_dir.name.split('-')[1])
                if self.task_ids is not None and task_num not in self.task_ids:
                    continue

                task_name = task_dir.name  # e.g., "task-0000"
                
                for parquet_file in sorted(task_dir.glob("*.parquet")):
                    episode_name = parquet_file.stem  # e.g., "episode_00000010"
                    
                    # Count frames in episode
                    try:
                        df = pd.read_parquet(parquet_file)
                        total_frames = len(df)
                    except Exception as e:
                        zero_rank_print(f"Warning: Could not read {parquet_file}: {e}")
                        continue
                    
                    # Build paths
                    # Video path template: videos/task-0000/{cam_name}/episode_00000010.mp4
                    video_path_template = str(video_dir / task_name / "{}" / f"{episode_name}.mp4")

                    # Get task caption from tasks.jsonl using task_num
                    caption = self.task_descriptions.get(task_num, f"B1K task {task_num}")
                    
                    info = [
                        video_path_template,  # 0: video path template
                        None,                  # 1: camera_info (unused)
                        str(parquet_file),     # 2: parquet path
                        domain_name,           # 3: domain name
                        "",                    # 4: domain_id (unused)
                        None,                  # 5: task_info (unused)
                        caption,               # 6: caption
                        total_frames,          # 7: total frames
                    ]
                    self.dataset.append(info)
        
        # Save cache if path provided
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.dataset, f)
            zero_rank_print(f"Saved dataset cache to {cache_path}")
    
    def get_frame_indexes(self, total_frames):
        """Select memory frames and prediction frames.
        
        Same logic as CustomLeRobotDataset for compatibility.
        """
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            action_indexes = np.clip(action_indexes, 0, total_frames - 1)
            frame_indexes = np.clip(frame_indexes, 0, total_frames - 1)
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes
        
        chunk_end = random.randint(self.action_chunk, total_frames + self.action_chunk)
        
        indexes_start = max(-self.n_previous, chunk_end - self.sample_n_frames)
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, 1, total_frames - 1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = indexes[:-self.action_chunk]
        
        if len(mem_candidates) < self.n_previous - 1:
            mem_candidates = [1] * (self.n_previous - 1) + mem_candidates
        
        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates) - 1, self.n_previous).tolist()]
        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(len(mem_candidates) - 1)), size=self.n_previous - 1, replace=False).tolist())] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(f"Unsupported previous_pick_mode: {self.previous_pick_mode}")
        
        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride - 1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]
        
        action_indexes = mem_indexes + video_end
        
        return frame_indexes, action_indexes
    
    def get_action_bias_std(self, domain_name, expected_dim=None):
        """Get normalization mean and std for actions or states.

        Args:
            domain_name: Name key for statistics (e.g., "behavior1k" or "behavior1k_state")
            expected_dim: Expected dimension for fallback identity normalization
        """
        key = f"{domain_name}_{self.action_space}"
        # if key not in self.StatisticInfo:
        #     if expected_dim is None:
        #         expected_dim = 23  # Default for action
        #     zero_rank_print(f"Warning: Stats key '{key}' not found, using identity normalization with dim={expected_dim}")
        #     return torch.zeros(1, expected_dim), torch.ones(1, expected_dim)
        return (
            torch.tensor(self.StatisticInfo[key]['mean']).unsqueeze(0),
            torch.tensor(self.StatisticInfo[key]['std']).unsqueeze(0) + 1e-6
        )
    
    def extract_proprioceptive_stats(self, mean_full, std_full, indices):
        """Extract proprioceptive indices from full-dimensional statistics.
        
        Args:
            mean_full: Full statistics mean tensor of shape (1, full_dim)
            std_full: Full statistics std tensor of shape (1, full_dim)
            indices: List of indices to extract. If None, extracts first 23 dims.
        
        Returns:
            Tuple of (mean_extracted, std_extracted) with shape (1, len(indices))
        """
        return mean_full[:, indices], std_full[:, indices]

    def seek_mp4(self, video_path_template, cam_name_list, slices):
        """Load video frames from mp4 files."""
        video_list = []
        for cam_name in cam_name_list:
            video_path = video_path_template.format(cam_name)
            video_reader = VideoFileClip(video_path)
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx) / fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float() / 255.
            video_reader.close()
            video_list.append(video)
        return video_list
    
    def transform_video(self, videos, specific_transforms_resize, sample_size):
        """Apply transforms to video frames."""
        v = len(videos)
        new_videos = []
        for iv in range(v):
            video = videos[iv]
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        return new_videos, None
    
    def normalize_video(self, video, specific_transforms_norm):
        """Normalize video to [-1, 1] range."""
        c, v, t, h, w = video.shape
        video = specific_transforms_norm(video.permute(1, 2, 0, 3, 4).reshape(-1, c, h, w))
        video = video.reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
        return video
    
    def get_transform(self):
        return self.sample_size, self.pixel_transforms_resize, self.pixel_transforms_norm
    
    def get_long_recaption(self, step_captions, task_caption):
        """Generate augmented caption from task and step descriptions.
        
        This matches the original CustomLeRobotDataset behavior for caption augmentation.
        """
        newcap = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption, [])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list, 1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)
        
        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the ", " ")
        
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption, [])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list, 1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        
        cap_type = random.randint(0, 2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap
    
    def get_batch(self, idx):
        """Load a single batch item."""
        video_path_template = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames)
        
        # Load parquet data
        data = pd.read_parquet(parquet_path)

        # Extract actions and states (vectorized for performance)
        action = np.stack(data[self.action_key].values).astype(np.float32)
        state = np.stack(data[self.state_key].values).astype(np.float32)

        # Extract and process proprioceptive features with EEF (end-effector) gripper logic
        # EEF grippers: sum finger positions to get single width value
        proprio_features = []
        
        # base_qvel (3)
        proprio_features.append(state[:, list(range(253, 256))])  # Shape: (num_frames, 3)
        
        # trunk_qpos (4)
        proprio_features.append(state[:, list(range(236, 240))])  # Shape: (num_frames, 4)
        
        # left_arm_qpos (7)
        proprio_features.append(state[:, list(range(158, 165))])  # Shape: (num_frames, 7)
        
        # left_gripper_qpos (2 -> 1, summed as EEF)
        left_gripper = state[:, list(range(193, 195))].sum(axis=-1, keepdims=True)  # Shape: (num_frames, 1)
        proprio_features.append(left_gripper)
        
        # right_arm_qpos (7)
        proprio_features.append(state[:, list(range(197, 204))])  # Shape: (num_frames, 7)
        
        # right_gripper_qpos (2 -> 1, summed as EEF)
        right_gripper = state[:, list(range(232, 234))].sum(axis=-1, keepdims=True)  # Shape: (num_frames, 1)
        proprio_features.append(right_gripper)
        
        # Concatenate all features: 3 + 4 + 7 + 1 + 7 + 1 = 23 dims
        state = np.concatenate(proprio_features, axis=-1)

        # Get normalization statistics - may be full 256-dim or already 23-dim from JSON
        action_mean_full, action_std_full = self.get_action_bias_std(domain_name, expected_dim=23)
        state_mean_full, state_std_full = self.get_action_bias_std(domain_name + "_state", expected_dim=23)
        
        # Extract proprioceptive indices consistently
        action_mean, action_std = self.extract_proprioceptive_stats(action_mean_full, action_std_full, indices=list(range(23)))
        state_mean, state_std = self.extract_proprioceptive_stats(state_mean_full, state_std_full, indices=PROPRIO_INDICES_256)
        
        if self.action_type == "absolute":
            # All 23 dimensions use standard stats with consistent index extraction
            action = torch.FloatTensor(action[indexes])
            action = (action - action_mean) / action_std
            state = torch.FloatTensor(state)[indexes][self.n_previous - 1:self.n_previous]
            state = (state - state_mean) / state_std
            
        elif self.action_type == "delta":
            # Use class constants for consistent index mapping
            delta_action_indices = self.DELTA_ACTION_INDICES
            absolute_action_indices = self.ABSOLUTE_ACTION_INDICES
            delta_state_indices = self.DELTA_STATE_INDICES
            absolute_state_indices = self.ABSOLUTE_STATE_INDICES
            
            # Get full delta stats (may be full 256-dim or already 23-dim from JSON)
            delta_mean_full, delta_std_full = self.get_action_bias_std(domain_name + "_delta", expected_dim=23)
            
            # Extract proprioceptive indices consistently for delta stats
            delta_mean_full_23, delta_std_full_23 = self.extract_proprioceptive_stats(delta_mean_full, delta_std_full, indices=list(range(23)))
            
            # Extract only delta indices from extracted stats for proper dimensionality
            delta_mean_subset = delta_mean_full_23[:, delta_action_indices]  # (1, 17)
            delta_std_subset = delta_std_full_23[:, delta_action_indices]    # (1, 17)
            action_mean_absolute = action_mean[:, absolute_action_indices]  # (1, 5)
            action_std_absolute = action_std[:, absolute_action_indices]    # (1, 5)
            
            # Process actions
            action_curr = torch.FloatTensor(action[indexes])
            action_last = torch.FloatTensor(action[[i - 1 for i in indexes]])
            
            delta_action = action_curr.clone()
            delta_action[:, delta_action_indices] = action_curr[:, delta_action_indices] - action_last[:, delta_action_indices]
            
            # Normalize with subset stats
            delta_action[:, delta_action_indices] = (delta_action[:, delta_action_indices] - delta_mean_subset) / delta_std_subset
            delta_action[:, absolute_action_indices] = (action_curr[:, absolute_action_indices] - action_mean_absolute) / action_std_absolute
            action = delta_action
            
            # Extract proprioceptive indices consistently for state stats
            delta_mean_state_full_23, delta_std_state_full_23 = self.extract_proprioceptive_stats(delta_mean_full, delta_std_full, indices=list(range(23)))
            state_delta_mean_subset = delta_mean_state_full_23[:, delta_state_indices]  # (1, 17)
            state_delta_std_subset = delta_std_state_full_23[:, delta_state_indices]    # (1, 17)
            
            # Process states with same logic
            state_mean_absolute = state_mean[:, absolute_state_indices]
            state_std_absolute = state_std[:, absolute_state_indices]
            
            state_curr = torch.FloatTensor(state)[indexes]
            state_last = torch.FloatTensor(state)[[i - 1 for i in indexes]]
            
            delta_state = state_curr.clone()
            delta_state[:, delta_state_indices] = state_curr[:, delta_state_indices] - state_last[:, delta_state_indices]
            
            # Normalize with subset stats
            delta_state[:, delta_state_indices] = (delta_state[:, delta_state_indices] - state_delta_mean_subset) / state_delta_std_subset
            delta_state[:, absolute_state_indices] = (state_curr[:, absolute_state_indices] - state_mean_absolute) / state_std_absolute
            state = delta_state[:, self.n_previous - 1:self.n_previous]
            
        elif self.action_type == "relative":
            raise ValueError(
                f"Action type 'relative' is not supported for B1K dataset. "
                f"Please use 'absolute' or 'delta' action types instead."
            )
        
        # Load videos
        videos = self.seek_mp4(video_path_template, self.valid_cam, vid_indexes)
        videos, _ = self.transform_video(videos, specific_transforms_resize, sample_size)
        videos = self.normalize_video(videos, specific_transforms_norm)
        
        # Apply unified prompt if enabled
        if self.use_unified_prompt:
            caption = self.unified_prompt
        
        return videos, action, caption, state
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.fix_epiidx is not None:
            video, actions, caption, state = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state = self.get_batch(idx)
                    break
                except ValueError:
                    # Re-raise ValueErrors (e.g., unsupported action_type)
                    # These are configuration errors, not data loading errors
                    raise
                except Exception:
                    traceback.print_exc()
                    idx = random.randint(0, self.length - 1)
        
        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        return sample


# # Alias for config compatibility
# CustomLeRobotDataset = B1KDataset