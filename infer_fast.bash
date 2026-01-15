#!/bin/bash

# Array of chunk values to test
CHUNKS=(1 2 4 6 8 10)

# Base paths
IMAGE_ROOT="video_gen_examples/sample_794073_frame184_task_prompt"
CONFIG_FILE="configs/ltx_model/video_model_infer_fast.yaml"
OUTPUT_BASE="evaluation_results/video_gen"

# Build prompt path from image root
PROMPT_FILE="${IMAGE_ROOT}/prompt.txt"

# Extract sample name from image root for output naming
SAMPLE_NAME=$(basename ${IMAGE_ROOT})

# Iterate over chunk values
for CHUNK in "${CHUNKS[@]}"; do
    echo "========================================="
    echo "Running inference with n_chunk=${CHUNK}"
    echo "========================================="
    
    # Build output path with sample name and chunk
    OUTPUT_PATH="${OUTPUT_BASE}/fast_${SAMPLE_NAME}_chunk${CHUNK}"
    
    CUDA_VISIBLE_DEVICES=0 python video_gen_examples/infer.py \
        --config_file ${CONFIG_FILE} \
        --image_root ${IMAGE_ROOT} \
        --prompt_txt_file ${PROMPT_FILE} \
        --output_path ${OUTPUT_PATH} \
        --n_chunk ${CHUNK}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed n_chunk=${CHUNK}"
    else
        echo "✗ Failed for n_chunk=${CHUNK}"
        exit 1
    fi
    
    echo ""
done

echo "All inference runs completed!"