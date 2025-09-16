#!/bin/bash

# Define sequences to process
# SEQUENCES=("night" "weather" "season" "nordland" "uacampus")
# SEQUENCES=("night")
SEQUENCES=("nordland" "uacampus")

# Define paths
DATA_ROOT="data/GV-Bench"
OUTPUT_BASE="output/gvbench"
PRETRAINED_MODEL="checkpoints/checkpoint-dg%2Bvisym.pth"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_BASE

echo "Starting GV-Bench evaluation across all sequences..."

# Process each sequence
for seq in "${SEQUENCES[@]}"; do
    echo "===================================================="
    echo "Processing sequence: $seq"
    echo "===================================================="
    
    # Run the Python script for this sequence
    python gvbench_usage.py \
        --data_root $DATA_ROOT \
        --output_path $OUTPUT_BASE \
        --seq $seq \
        --pretrained $PRETRAINED_MODEL
    
    echo "Completed processing sequence: $seq"
    echo ""
done

echo "All sequences processed successfully!"
echo "Results saved to: $OUTPUT_BASE"