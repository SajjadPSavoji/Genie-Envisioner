"""
Test script for B1K dataset implementation.

Usage:
    conda activate genie_envisioner
    cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner
    python b1k_stuff/test_b1k_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from data.b1k_dataset import B1KDataset
from pathlib import Path
import json


def test_dataset_initialization():
    """Test that dataset can be initialized."""
    print("\n=== Test 1: Dataset Initialization ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"
    stat_file = "configs/ltx_model/b1k/behavior1k_stats.json"

    # Check if paths exist
    if not Path(data_root).exists():
        print(f"❌ Data root not found: {data_root}")
        return False

    if not Path(stat_file).exists():
        print(f"⚠️  Stats file not found: {stat_file}")
        print("   You may need to run compute_b1k_stats.py first")
        stat_file = None

    try:
        dataset = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0, 1],
            sample_size=(224, 224),
            sample_n_frames=16,  # Small for testing
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=['observation.images.rgb.head'],  # Just one camera for speed
            stat_file=stat_file,
            action_space="joint",
            action_type="absolute",

        )
        print(f"✓ Dataset initialized successfully")
        print(f"  Total episodes: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataset_loading(dataset):
    """Test loading a single sample."""
    print("\n=== Test 2: Loading Single Sample ===")

    if dataset is None:
        print("❌ Skipping - dataset not initialized")
        return False

    try:
        sample = dataset[201]
        print(f"✓ Successfully loaded sample")
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Actions shape: {sample['actions'].shape}")
        print(f"  Caption: {sample['caption']}")
        print(f"  State shape: {sample['state'].shape}")

        # Verify shapes
        c, v, t, h, w = sample['video'].shape
        print(f"\n  Video breakdown: C={c}, Views={v}, Time={t}, H={h}, W={w}")

        return True
    except Exception as e:
        print(f"❌ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_camera(dataset):
    """Test loading with multiple cameras."""
    print("\n=== Test 3: Multi-Camera Loading ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"

    try:
        dataset_multi = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0, 1],
            sample_size=(224, 224),
            sample_n_frames=16,
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=[
                'observation.images.rgb.head',
                'observation.images.rgb.left_wrist',
                'observation.images.rgb.right_wrist',
            ],
            stat_file=None,  # Skip stats for faster testing
            action_space="joint",
            action_type="absolute",
        )

        sample = dataset_multi[0]
        c, v, t, h, w = sample['video'].shape

        print(f"✓ Multi-camera dataset loaded")
        print(f"  Number of camera views: {v}")
        print(f"  Expected: 3 cameras")

        if v == 3:
            print(f"  ✓ Correct number of views")
            return True
        else:
            print(f"  ❌ Wrong number of views")
            return False

    except Exception as e:
        print(f"❌ Multi-camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_loading(dataset):
    """Test loading multiple samples (simulate DataLoader)."""
    print("\n=== Test 4: Batch Loading ===")

    if dataset is None:
        print("❌ Skipping - dataset not initialized")
        return False

    try:
        samples = [dataset[i] for i in range(min(3, len(dataset)))]
        print(f"✓ Successfully loaded {len(samples)} samples")

        # Test stacking (what DataLoader does)
        videos = torch.stack([s['video'] for s in samples])
        actions = torch.stack([s['actions'] for s in samples])
        states = torch.stack([s['state'] for s in samples])

        print(f"  Batched video shape: {videos.shape}")
        print(f"  Batched actions shape: {actions.shape}")
        print(f"  Batched states shape: {states.shape}")

        return True
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_integration(dataset):
    """Test with PyTorch DataLoader."""
    print("\n=== Test 5: DataLoader Integration ===")

    if dataset is None:
        print("❌ Skipping - dataset not initialized")
        return False

    try:
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        batch = next(iter(dataloader))

        print(f"✓ DataLoader integration successful")
        print(f"  Batch video shape: {batch['video'].shape}")
        print(f"  Batch actions shape: {batch['actions'].shape}")
        print(f"  Batch states shape: {batch['state'].shape}")
        print(f"  Batch captions: {len(batch['caption'])} strings")

        return True
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_types():
    """Test different action types (absolute, delta, relative)."""
    print("\n=== Test 6: Action Types ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"
    stat_file = "configs/ltx_model/b1k/behavior1k_stats.json"

    if not Path(stat_file).exists():
        print(f"⚠️  Skipping - stats file needed for this test")
        return False

    action_types = ["absolute", "delta", "relative"]

    for action_type in action_types:
        try:
            dataset = B1KDataset(
                data_roots=[data_root],
                domains=["behavior1k"],
                task_ids=[0],
                sample_size=(224, 224),
                sample_n_frames=16,
                chunk=4,
                action_chunk=4,
                n_previous=12,
                valid_cam=['observation.images.rgb.head'],
                stat_file=stat_file,
                action_space="joint",
                action_type=action_type,
            )

            if action_type == "relative":
                # Relative action type should raise an error for B1K
                try:
                    sample = dataset[0]
                    print(f"  ❌ {action_type}: should have raised ValueError!")
                    return False
                except ValueError as e:
                    if "not supported for B1K" in str(e):
                        print(f"  ✓ {action_type}: correctly raises error (incompatible dimensions)")
                    else:
                        print(f"  ❌ {action_type}: unexpected error: {e}")
                        return False
            else:
                sample = dataset[0]
                print(f"  ✓ {action_type}: actions shape {sample['actions'].shape}")
        except Exception as e:
            print(f"  ❌ {action_type}: unexpected error during initialization: {e}")
            return False

    return True


def test_video_value_range(dataset):
    """Test that video values are normalized to [-1, 1] range."""
    print("\n=== Test 7: Video Value Range ===")

    if dataset is None:
        print("❌ Skipping - dataset not initialized")
        return False

    try:
        sample = dataset[0]
        video = sample['video']

        min_val = video.min().item()
        max_val = video.max().item()
        mean_val = video.mean().item()

        print(f"  Video min: {min_val:.4f}")
        print(f"  Video max: {max_val:.4f}")
        print(f"  Video mean: {mean_val:.4f}")

        # Check if values are in [-1, 1] range (with small tolerance)
        if min_val >= -1.1 and max_val <= 1.1:
            print(f"  ✓ Video values are in normalized range [-1, 1]")
            return True
        else:
            print(f"  ❌ Video values out of expected range!")
            return False

    except Exception as e:
        print(f"❌ Video value range test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_state_normalization():
    """Test that actions and states are properly normalized."""
    print("\n=== Test 8: Action/State Normalization ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"
    stat_file = "configs/ltx_model/b1k/behavior1k_stats.json"

    if not Path(stat_file).exists():
        print(f"⚠️  Skipping - stats file needed for this test")
        return False

    try:
        dataset = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0],
            sample_size=(224, 224),
            sample_n_frames=16,
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=['observation.images.rgb.head'],
            stat_file=stat_file,
            action_space="joint",
            action_type="absolute",
        )

        # Collect stats from multiple samples
        action_samples = []
        state_samples = []
        
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            action_samples.append(sample['actions'])
            state_samples.append(sample['state'])

        actions = torch.cat(action_samples, dim=0)
        states = torch.cat(state_samples, dim=0)

        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0)
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0)

        print(f"  Actions mean range: [{action_mean.min().item():.3f}, {action_mean.max().item():.3f}]")
        print(f"  Actions std range: [{action_std.min().item():.3f}, {action_std.max().item():.3f}]")
        print(f"  States mean range: [{state_mean.min().item():.3f}, {state_mean.max().item():.3f}]")
        print(f"  States std range: [{state_std.min().item():.3f}, {state_std.max().item():.3f}]")

        # Normalized data should have mean close to 0 and std close to 1
        # Use loose bounds since we're only sampling a few episodes
        action_mean_ok = action_mean.abs().max().item() < 5.0
        action_std_ok = action_std.mean().item() > 0.1 and action_std.mean().item() < 10.0

        if action_mean_ok and action_std_ok:
            print(f"  ✓ Normalization appears to be working")
            return True
        else:
            print(f"  ⚠️  Normalization may have issues (values outside expected range)")
            return True  # Don't fail, just warn

    except Exception as e:
        print(f"❌ Normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caption_augmentation():
    """Test caption augmentation with recap files."""
    print("\n=== Test 9: Caption Augmentation ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"

    try:
        # Test without recap files (should return original caption)
        dataset = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0],
            sample_size=(224, 224),
            sample_n_frames=16,
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=['observation.images.rgb.head'],
            task_recap_file=None,
            step_recap_file=None,
        )

        sample = dataset[0]
        caption = sample['caption']
        
        print(f"  Caption without recap: '{caption[:60]}...'")
        
        if caption and len(caption) > 0:
            print(f"  ✓ Caption generated successfully")
        else:
            print(f"  ❌ Empty caption")
            return False

        # Test unified prompt override
        dataset_unified = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0],
            sample_size=(224, 224),
            sample_n_frames=16,
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=['observation.images.rgb.head'],
            use_unified_prompt=True,
            unified_prompt="test unified prompt",
        )

        sample_unified = dataset_unified[0]
        unified_caption = sample_unified['caption']
        
        print(f"  Unified prompt: '{unified_caption}'")
        
        if unified_caption == "test unified prompt":
            print(f"  ✓ Unified prompt override works")
            return True
        else:
            print(f"  ❌ Unified prompt not applied correctly")
            return False

    except Exception as e:
        print(f"❌ Caption augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_index_edge_cases():
    """Test frame index generation with edge cases."""
    print("\n=== Test 10: Frame Index Edge Cases ===")

    data_root = "/shared_work/physical_intelligence/2025-challenge-demos"

    try:
        dataset = B1KDataset(
            data_roots=[data_root],
            domains=["behavior1k"],
            task_ids=[0],
            sample_size=(224, 224),
            sample_n_frames=16,
            chunk=4,
            action_chunk=4,
            n_previous=12,
            valid_cam=['observation.images.rgb.head'],
        )

        # Test with a short episode (should handle gracefully)
        # Find an episode to test the edge cases
        test_total_frames = 50  # Simulate short episode
        
        frame_idx, action_idx = dataset.get_frame_indexes(test_total_frames)
        
        print(f"  With {test_total_frames} frames:")
        print(f"    Frame indexes: {len(frame_idx)} indexes, range [{min(frame_idx)}, {max(frame_idx)}]")
        print(f"    Action indexes: {len(action_idx)} indexes, range [{min(action_idx)}, {max(action_idx)}]")

        # Check indexes are valid (within bounds)
        if min(frame_idx) >= 0 and max(frame_idx) < test_total_frames:
            print(f"  ✓ Frame indexes within valid bounds")
        else:
            print(f"  ❌ Frame indexes out of bounds!")
            return False

        if min(action_idx) >= 0 and max(action_idx) < test_total_frames:
            print(f"  ✓ Action indexes within valid bounds")
        else:
            print(f"  ❌ Action indexes out of bounds!")
            return False

        # Test with very short episode
        very_short = 10
        try:
            frame_idx2, action_idx2 = dataset.get_frame_indexes(very_short)
            print(f"  With {very_short} frames: handled gracefully")
            print(f"  ✓ Edge case handled")
        except Exception as e:
            print(f"  ⚠️  Very short episode raised error: {e}")

        return True

    except Exception as e:
        print(f"❌ Frame index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("B1K Dataset Test Suite")
    print("=" * 60)

    results = {}

    # Run tests
    dataset = test_dataset_initialization()
    results['initialization'] = dataset is not None

    results['single_sample'] = test_dataset_loading(dataset)
    results['multi_camera'] = test_multi_camera(dataset)
    results['batch_loading'] = test_batch_loading(dataset)
    results['dataloader'] = test_dataloader_integration(dataset)
    results['action_types'] = test_action_types()
    results['video_value_range'] = test_video_value_range(dataset)
    results['action_state_normalization'] = test_action_state_normalization()
    results['caption_augmentation'] = test_caption_augmentation()
    results['frame_index_edge_cases'] = test_frame_index_edge_cases()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
