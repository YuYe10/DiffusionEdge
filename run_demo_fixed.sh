#!/bin/bash
# Test script to verify the fix works

set -e

echo "=================================="
echo "DiffusionEdge Demo - Testing Fix"
echo "=================================="
echo

# Step 1: Check if trained models exist
echo "Step 1: Looking for trained models..."
if [ -d "./outputs/disloss" ]; then
    model_count=$(find ./outputs/disloss -name "model-*.pt" | wc -l)
    if [ $model_count -gt 0 ]; then
        latest_model=$(find ./outputs/disloss -name "model-*.pt" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        echo "✓ Found $model_count trained models"
        echo "  Latest: $latest_model"
    else
        echo "✗ No model files found in ./outputs/disloss"
        exit 1
    fi
else
    echo "✗ ./outputs/disloss directory not found"
    exit 1
fi
echo

# Step 2: Show recommended command
echo "Step 2: Recommended command:"
echo "---"
echo "python3 demo.py \\"
echo "    --cfg ./configs/BSDS_sample.yaml \\"
echo "    --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \\"
echo "    --pre_weight $latest_model \\"
echo "    --out_dir ./outputs/demo_split_3_test \\"
echo "    --bs 8"
echo "---"
echo

# Step 3: Ask if user wants to run
read -p "Run this command? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running inference..."
    python3 demo.py \
        --cfg ./configs/BSDS_sample.yaml \
        --input_dir ./data/Multicue_split/MDBD_split3/test/imgs \
        --pre_weight "$latest_model" \
        --out_dir ./outputs/demo_split_3_test \
        --bs 8
    echo
    echo "✓ Inference complete!"
    echo "Results saved to: ./outputs/demo_split_3_test"
else
    echo "Cancelled."
    exit 0
fi
