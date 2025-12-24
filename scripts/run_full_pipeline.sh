#!/usr/bin/env bash
set -euo pipefail

# Phase 1: 1 epoch on small data (memmap expected at data/phase1_small)
python3 scripts/train_phase1.py --epochs 1 --data-dir data/phase1_small --num-workers 0

# Phase 2: 1 iteration using Phase 1 checkpoint
python3 scripts/train_phase2.py --device cpu --checkpoint checkpoints/phase1/best.pt --num-games 1 --mcts-sims 1 --training-steps 1 --batch-size 2 --use-batched --batched-games 1 --max-game-length 4

echo "Launch dashboard with: python3 scripts/visual_dashboard.py --checkpoint checkpoints/phase2/phase2_iter_0.pt"
