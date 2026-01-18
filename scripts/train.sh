#!/bin/bash
# Training script for MWK-FLAIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Training with HarMA adapter
python examples/train_with_harma.py \
    --data_root ./data \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --adapter_dim 64 \
    --use_harma \
    --save_dir ./checkpoints \
    --log_dir ./logs

echo "Training completed!"
