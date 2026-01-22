"""
Compute normalization statistics for BEHAVIOR-1K dataset.

This script adapts scripts/get_statistics.py for B1K's nested directory structure:
  data/task-XXXX/episode_XXXXXXXX.parquet

Usage:
    conda activate genie_envisioner
    cd /shared_work/physical_intelligence/ruiheng/Genie-Envisioner
    python b1k_stuff/compute_b1k_stats.py \
        --data_root /shared_work/physical_intelligence/2025-challenge-demos \
        --save_path configs/ltx_model/b1k/behavior1k_stats.json
"""

import os
import numpy as np
import pandas as pd
import tqdm
import json
import argparse
from pathlib import Path


def load_data(data_path, key="action"):
    """Load a column from parquet file and stack into array."""
    data = pd.read_parquet(data_path)
    data = np.stack([data[key][i] for i in range(data[key].shape[0])])
    return data 


def cal_statistic(data, _filter=True):
    """Compute statistics with optional outlier filtering."""
    q99 = np.percentile(data, 99, axis=0)
    q01 = np.percentile(data,  1, axis=0)
    if _filter:
        data_mask = (data >= q01) & (data <= q99)
        data_mask = data_mask.min(axis=1)
        data = data[data_mask, :]
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    # Avoid zero std
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds, q99, q01


def update_running_stats(existing_stats, batch_data):
    """Update running statistics with new batch using Welford's algorithm.

    Args:
        existing_stats: dict with 'count', 'mean', 'M2' (for variance), 'min', 'max'
        batch_data: new data batch to incorporate

    Returns:
        Updated stats dict
    """
    batch_count = len(batch_data)
    batch_mean = np.mean(batch_data, axis=0)
    batch_var = np.var(batch_data, axis=0)
    batch_M2 = batch_var * batch_count

    if existing_stats is None:
        return {
            'count': batch_count,
            'mean': batch_mean,
            'M2': batch_M2,
            'min': np.min(batch_data, axis=0),
            'max': np.max(batch_data, axis=0),
        }

    # Combine statistics using parallel algorithm
    total_count = existing_stats['count'] + batch_count
    delta = batch_mean - existing_stats['mean']
    new_mean = existing_stats['mean'] + delta * batch_count / total_count
    new_M2 = existing_stats['M2'] + batch_M2 + delta**2 * existing_stats['count'] * batch_count / total_count

    return {
        'count': total_count,
        'mean': new_mean,
        'M2': new_M2,
        'min': np.minimum(existing_stats['min'], np.min(batch_data, axis=0)),
        'max': np.maximum(existing_stats['max'], np.max(batch_data, axis=0)),
    }


def compute_percentiles_batched(data_path_list, action_key, state_key, batch_size, percentiles=[1, 99]):
    """Compute exact percentiles by processing all data in batches.

    Collects all data points across batches to compute exact percentiles.
    Memory usage is controlled by batch_size.
    """
    print("  Collecting data for percentile computation...")

    all_data_batches = []
    all_delta_batches = []
    all_state_batches = []

    num_batches = (len(data_path_list) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data_path_list))
        batch_paths = data_path_list[start_idx:end_idx]

        data_list = []
        delta_data_list = []
        state_list = []

        for data_path in tqdm.tqdm(batch_paths, desc=f"  Percentiles batch {batch_idx + 1}/{num_batches}", leave=False):
            try:
                data = load_data(data_path, action_key)
                delta_data = data[1:] - data[:-1]
                state = load_data(data_path, state_key)

                data_list.append(data)
                delta_data_list.append(delta_data)
                state_list.append(state)
            except Exception as e:
                pass

        if data_list:
            batch_data = np.concatenate(data_list, axis=0)
            batch_delta = np.concatenate(delta_data_list, axis=0)
            batch_state = np.concatenate(state_list, axis=0)

            all_data_batches.append(batch_data)
            all_delta_batches.append(batch_delta)
            all_state_batches.append(batch_state)

            del data_list, delta_data_list, state_list, batch_data, batch_delta, batch_state

    print("  Computing percentiles from all data...")
    all_data = np.concatenate(all_data_batches, axis=0)
    all_delta = np.concatenate(all_delta_batches, axis=0)
    all_state = np.concatenate(all_state_batches, axis=0)

    del all_data_batches, all_delta_batches, all_state_batches

    result = {}
    for p in percentiles:
        result[f'q{p:02d}'] = {
            'data': np.percentile(all_data, p, axis=0),
            'delta': np.percentile(all_delta, p, axis=0),
            'state': np.percentile(all_state, p, axis=0),
        }

    del all_data, all_delta, all_state

    return result


def update_running_stats_filtered(existing_stats, batch_data, q01, q99):
    """Update running statistics with new batch, applying outlier filtering.

    Args:
        existing_stats: dict with 'count', 'mean', 'M2' (for variance)
        batch_data: new data batch to incorporate
        q01: 1st percentile for filtering
        q99: 99th percentile for filtering

    Returns:
        Updated stats dict
    """
    # Apply filtering
    data_mask = (batch_data >= q01) & (batch_data <= q99)
    data_mask = data_mask.min(axis=1)
    filtered_data = batch_data[data_mask, :]

    if len(filtered_data) == 0:
        return existing_stats

    batch_count = len(filtered_data)
    batch_mean = np.mean(filtered_data, axis=0)
    batch_var = np.var(filtered_data, axis=0)
    batch_M2 = batch_var * batch_count

    if existing_stats is None:
        return {
            'count': batch_count,
            'mean': batch_mean,
            'M2': batch_M2,
        }

    # Combine statistics using parallel algorithm
    total_count = existing_stats['count'] + batch_count
    delta = batch_mean - existing_stats['mean']
    new_mean = existing_stats['mean'] + delta * batch_count / total_count
    new_M2 = existing_stats['M2'] + batch_M2 + delta**2 * existing_stats['count'] * batch_count / total_count

    return {
        'count': total_count,
        'mean': new_mean,
        'M2': new_M2,
    }


def find_b1k_episodes(data_root: str, task_ids: list = None) -> list[str]:
    """Find all episode parquet files in B1K's nested structure.
    
    B1K structure: data/task-XXXX/episode_XXXXXXXX.parquet
    
    Args:
        data_root: Root directory of B1K dataset
        task_ids: Optional list of task IDs to include (e.g., [0, 1, 5]). If None, includes all tasks.
    """
    data_root = Path(data_root)
    parquet_dir = data_root / "data"
    
    episodes = []
    for task_dir in sorted(parquet_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        
        # Extract task number and filter if task_ids specified
        task_num = int(task_dir.name.split('-')[1])
        if task_ids is not None and task_num not in task_ids:
            continue
        
        for parquet_file in sorted(task_dir.glob("*.parquet")):
            episodes.append(str(parquet_file))
    
    return episodes


def get_statistics(data_root, data_name, data_type, save_path,
                   action_key="action", state_key="observation.state",
                   nrnd=2000, _filter=True, task_ids=None, batch_size=None):
    """Compute and save statistics for B1K dataset.

    Args:
        task_ids: Optional list of task IDs to include (e.g., [0, 1, 5]). If None, uses all tasks.
        batch_size: Number of episodes to process at once to limit memory usage. If None, processes all at once.
    """

    assert data_type in ["joint", "eef"], f"data_type must be 'joint' or 'eef', got {data_type}"

    # Find all B1K episodes (optionally filtered by task_ids)
    print(f"Scanning for episodes in {data_root}...")
    data_path_list = find_b1k_episodes(data_root, task_ids=task_ids)
    print(f"Found {len(data_path_list)} episodes")

    # Sample if needed
    if nrnd < len(data_path_list):
        data_path_list = np.random.choice(data_path_list, nrnd, replace=False).tolist()
        print(f"Sampled {nrnd} episodes")

    # Determine if using batch processing
    use_batching = batch_size is not None and batch_size < len(data_path_list)

    if use_batching:
        # 3-PASS APPROACH: Exact statistics with memory-efficient batch processing
        num_batches = (len(data_path_list) + batch_size - 1) // batch_size
        print(f"Processing in {num_batches} batches of ~{batch_size} episodes (3-pass mode for exact results)")

        # Pass 1: Compute percentiles from all data (processed in batches)
        print("\nPass 1: Computing percentiles...")
        percentile_results = compute_percentiles_batched(
            data_path_list, action_key, state_key, batch_size, percentiles=[1, 99]
        )
        q01 = percentile_results['q01']['data']
        q99 = percentile_results['q99']['data']
        delta_q01 = percentile_results['q01']['delta']
        delta_q99 = percentile_results['q99']['delta']
        state_q01 = percentile_results['q01']['state']
        state_q99 = percentile_results['q99']['state']

        # Pass 2: Compute mean and std incrementally with outlier filtering
        print("\nPass 2: Computing mean and std with outlier filtering...")
        data_stats = None
        delta_stats = None
        state_stats = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_path_list))
            batch_paths = data_path_list[start_idx:end_idx]

            data_list = []
            state_list = []
            delta_data_list = []

            for data_path in tqdm.tqdm(batch_paths, desc=f"  Batch {batch_idx + 1}/{num_batches}", leave=False):
                try:
                    data = load_data(data_path, action_key)
                    data_list.append(data)
                    delta_data = data[1:] - data[:-1]
                    delta_data_list.append(delta_data)
                    state = load_data(data_path, state_key)
                    state_list.append(state)
                except Exception as e:
                    print(f"Warning: Failed to load {data_path}: {e}")

            # Concatenate batch and update running stats with filtering
            if data_list:
                batch_data = np.concatenate(data_list, axis=0)
                batch_delta = np.concatenate(delta_data_list, axis=0)
                batch_state = np.concatenate(state_list, axis=0)

                if _filter:
                    data_stats = update_running_stats_filtered(data_stats, batch_data, q01, q99)
                    delta_stats = update_running_stats_filtered(delta_stats, batch_delta, delta_q01, delta_q99)
                    state_stats = update_running_stats_filtered(state_stats, batch_state, state_q01, state_q99)
                else:
                    data_stats = update_running_stats(data_stats, batch_data)
                    delta_stats = update_running_stats(delta_stats, batch_delta)
                    state_stats = update_running_stats(state_stats, batch_state)

                # Free memory immediately
                del data_list, state_list, delta_data_list, batch_data, batch_delta, batch_state

        # Extract mean and std
        means = data_stats['mean']
        stds = np.sqrt(data_stats['M2'] / data_stats['count'])
        stds = np.where(stds < 1e-6, 1.0, stds)

        delta_means = delta_stats['mean']
        delta_stds = np.sqrt(delta_stats['M2'] / delta_stats['count'])
        delta_stds = np.where(delta_stds < 1e-6, 1.0, delta_stds)

        state_means = state_stats['mean']
        state_stds = np.sqrt(state_stats['M2'] / state_stats['count'])
        state_stds = np.where(state_stds < 1e-6, 1.0, state_stds)

        print(f"\nTotal samples processed (after filtering): {data_stats['count']:,}")
        print(f"Action dims: {len(means)}, State dims: {len(state_means)}")

    else:
        # Original implementation: load all at once
        print("Loading all episodes at once...")
        data_list = []
        state_list = []
        delta_data_list = []

        for data_path in tqdm.tqdm(data_path_list, desc="Loading episodes"):
            try:
                data = load_data(data_path, action_key)
                data_list.append(data)
                delta_data = data[1:] - data[:-1]
                delta_data_list.append(delta_data)
                state = load_data(data_path, state_key)
                state_list.append(state)
            except Exception as e:
                print(f"Warning: Failed to load {data_path}: {e}")

        print("Concatenating data...")
        data_list = np.concatenate(data_list, axis=0)
        delta_data_list = np.concatenate(delta_data_list, axis=0)
        state_list = np.concatenate(state_list, axis=0)

        assert len(data_list.shape) == 2
        assert len(delta_data_list.shape) == 2
        assert len(state_list.shape) == 2

        print(f"Total samples: {len(data_list)}")
        print(f"Action dims: {data_list.shape[1]}, State dims: {state_list.shape[1]}")

        print("Computing statistics...")
        means, stds, q99, q01 = cal_statistic(data_list, _filter=_filter)
        delta_means, delta_stds, delta_q99, delta_q01 = cal_statistic(delta_data_list, _filter=_filter)
        state_means, state_stds, state_q99, state_q01 = cal_statistic(state_list, _filter=_filter)

    statistics_info = {
        f"{data_name}_{data_type}": {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "q99": q99.tolist(),
            "q01": q01.tolist(),
        },
        f"{data_name}_delta_{data_type}": {
            "mean": delta_means.tolist(),
            "std": delta_stds.tolist(),
            "q99": delta_q99.tolist(),
            "q01": delta_q01.tolist(),
        },
        f"{data_name}_state_{data_type}": {
            "mean": state_means.tolist(),
            "std": state_stds.tolist(),
            "q99": state_q99.tolist(),
            "q01": state_q01.tolist(),
        },
    }

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(statistics_info, f, indent=4)
    
    print(f"\n✓ Saved statistics to {save_path}")
    
    # Summary
    print("\n=== Statistics Summary ===")
    for key, stats in statistics_info.items():
        print(f"\n{key}:")
        print(f"  Dims: {len(stats['mean'])}")
        print(f"  Mean: [{min(stats['mean']):.4f}, {max(stats['mean']):.4f}]")
        print(f"  Std:  [{min(stats['std']):.4f}, {max(stats['std']):.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute B1K normalization statistics")
    parser.add_argument('--data_root', default="/shared_work/physical_intelligence/2025-challenge-demos",
                        help="Root directory of B1K dataset")
    parser.add_argument('--data_name', default="behavior1k",
                        help="Dataset name prefix for output keys")
    parser.add_argument('--data_type', default="joint",
                        help="Data type: 'joint' or 'eef'")
    parser.add_argument('--action_key', default="action",
                        help="Column name for actions in parquet")
    parser.add_argument('--state_key', default="observation.state",
                        help="Column name for state in parquet")
    parser.add_argument('--save_path', default="configs/ltx_model/b1k/behavior1k_stats.json",
                        help="Output JSON file path")
    parser.add_argument('--nrnd', type=int, default=2000,
                        help="Number of random episodes to sample")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--task_ids', type=int, nargs='+', default=None,
                        help="List of task IDs to include (e.g., --task_ids 0 1 5). If not specified, uses all tasks.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Number of episodes to process per batch to limit memory usage. If not specified, processes all at once.")

    args = parser.parse_args()
    np.random.seed(args.seed)

    get_statistics(
        args.data_root, args.data_name, args.data_type, args.save_path,
        action_key=args.action_key, state_key=args.state_key, nrnd=args.nrnd,
        task_ids=args.task_ids, batch_size=args.batch_size
    )
