"""
Simple script to verify B1K data dimensions directly from source files.
"""
import pandas as pd
import numpy as np

# Load a sample episode from B1K
parquet_file = '/shared_work/physical_intelligence/2025-challenge-demos/data/task-0000/episode_00000010.parquet'

print("=" * 60)
print("B1K Data Dimension Verification")
print("=" * 60)
print(f"\nReading: {parquet_file}\n")

# Read the parquet file
df = pd.read_parquet(parquet_file)
breakpoint()
print(f"Number of timesteps in episode: {len(df)}")
print(f"\nColumns in parquet file:")
for col in df.columns:
    print(f"  - {col}")

# Check action dimension
if 'action' in df.columns:
    action_sample = df['action'].iloc[0]
    action_array = np.array(action_sample)
    print(f"\n{'='*60}")
    print("ACTION DIMENSIONS")
    print(f"{'='*60}")
    print(f"  Type: {type(action_sample)}")
    print(f"  Shape: {action_array.shape}")
    print(f"  Dimension: {len(action_sample)}")
    print(f"  First 5 values: {action_sample[:5]}")
else:
    print("\n⚠️  'action' column not found!")

# Check observation.state dimension
if 'observation.state' in df.columns:
    state_sample = df['observation.state'].iloc[0]
    state_array = np.array(state_sample)
    print(f"\n{'='*60}")
    print("OBSERVATION.STATE DIMENSIONS")
    print(f"{'='*60}")
    print(f"  Type: {type(state_sample)}")
    print(f"  Shape: {state_array.shape}")
    print(f"  Dimension: {len(state_sample)}")
    print(f"  First 10 values: {state_sample[:10]}")
    print(f"  Last 5 values: {state_sample[-5:]}")
else:
    print("\n⚠️  'observation.state' column not found!")

print(f"\n{'='*60}")
print("VERIFICATION")
print(f"{'='*60}")
if 'action' in df.columns and 'observation.state' in df.columns:
    print(f"✓ Action dimension: {len(df['action'].iloc[0])}")
    print(f"✓ State dimension: {len(df['observation.state'].iloc[0])}")
    print(f"\nThese are the ACTUAL dimensions in the B1K dataset!")
