# TinyAlphaZero

Minimal AlphaZero-style chess engine with supervised Phase 1 training and self-play Phase 2.

## Quick Start

## Data Generation

Generate random self-play games (JSON):

```bash
python3 scripts/generate_data.py --num-games 100 --output data/sample_games.json
```

## Convert to Memmap

Convert JSON dataset to memmap files for fast loading:

```bash
python3 scripts/convert_to_memmap.py --input data/sample_games.json --output-dir data/phase1
```

## Phase 1 Training

Train supervised model (example uses memmap data):

```bash
python3 scripts/train_phase1.py --data-dir data/phase1 --epochs 10 --num-workers 0
```

To use JSON directly:

```bash
python3 scripts/train_phase1.py --format json --json-path data/sample_games.json --epochs 1
```

## Phase 2 Training

Run self-play iteration (batched MCTS enabled):

```bash
python3 scripts/train_phase2.py --device cpu --num-games 2 --mcts-sims 10 --training-steps 5 --use-batched --batched-games 2 --batch-size 16
```

## Configuration Notes

On high-memory systems (e.g., 96GB Strix Halo), tune these for throughput:

- `--batch-size`: Larger batches improve GPU utilization but require more VRAM/RAM. Increase until you hit memory limits.
- `--num-workers`: Increase for parallel self-play if multiprocessing is available; otherwise use batched mode (`--use-batched`).
- `--batched-games`: Number of parallel games in virtual batching. Increase to raise positions/sec.

## Dashboard

Launch the visual dashboard (optionally load a checkpoint):

```bash
python3 scripts/visual_dashboard.py --checkpoint checkpoints/phase2/phase2_iter_0.pt
```
# Tiny-AlphaZero
