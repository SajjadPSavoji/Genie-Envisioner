#!/bin/bash

# Array of chunk values to test
CHUNKS=(1 2 4 6 8 10)

# Array of prompt types to test
PROMPT_TYPES=(
    "complete_prompt"
    # "first_prompt"
    # "second_prompt"
    # "no_prompt"
    # "task_prompt"
)

# Base paths
IMAGE_ROOT_BASE="video_gen_examples/sample_684445_frame500"
CONFIG_FILE="configs/ltx_model/video_model_infer_slow.yaml"
OUTPUT_BASE="evaluation_results/video_gen"

# Iterate over prompt types
for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    echo "========================================="
    echo "Processing prompt type: ${PROMPT_TYPE}"
    echo "========================================="

    # Build image root path with prompt type
    IMAGE_ROOT="${IMAGE_ROOT_BASE}_${PROMPT_TYPE}"

    # Build prompt path from image root
    PROMPT_FILE="${IMAGE_ROOT}/prompt.txt"

    # Extract sample name from image root for output naming
    SAMPLE_NAME=$(basename ${IMAGE_ROOT})

    # Iterate over chunk values
    for CHUNK in "${CHUNKS[@]}"; do
        echo "-----------------------------------------"
        echo "Running inference with n_chunk=${CHUNK}"
        echo "-----------------------------------------"

        # Build output path with sample name and chunk
        OUTPUT_PATH="${OUTPUT_BASE}/slow_${SAMPLE_NAME}_chunk${CHUNK}"

        CUDA_VISIBLE_DEVICES=1 python video_gen_examples/infer.py \
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

    echo "✓ Completed all chunks for ${PROMPT_TYPE}"
    echo ""
done

echo "All inference runs completed!"