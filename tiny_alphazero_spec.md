# TinyAlphaZero: A Minimal Chess Engine via Self-Play

## Project Overview

**Goal**: Build a minimal but complete AlphaZero-style chess engine that learns to play chess through self-play reinforcement learning.

**Hardware Target**: AMD Strix Halo i395 with 96GB unified VRAM

**Philosophy**: Smallest possible implementation that captures the core AlphaZero algorithm. Optimise for clarity and correctness over performance. Scale up only after validating each component works.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        TinyAlphaZero                            │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Supervised Learning (Learn the Rules)                 │
│  ├── Data: Random self-play games                               │
│  ├── Task: Predict legal moves from board state                 │
│  └── Goal: >95% legal move accuracy                             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Self-Play RL (Learn to Win)                           │
│  ├── MCTS: Tree search guided by policy + value networks        │
│  ├── Self-Play: Generate training games using MCTS              │
│  ├── Training: Policy → MCTS policy, Value → game outcome       │
│  └── Goal: Beat Phase 1 model, achieve positive Elo growth      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Supervised Learning

### 1.1 Objective

Train a transformer to predict chess moves from board positions. The model should learn:
- Legal move generation (piece movement rules)
- Board state encoding/decoding
- Basic positional understanding

**Success Criteria**:
- Legal move accuracy > 95% (model's top-1 prediction is a legal move)
- Exact match accuracy > 15% (predicts the actual move played)
- Model generates only legal moves when sampled with temperature

### 1.2 Board Representation

#### 1.2.1 Input Encoding

Fixed-length sequence of 69 tokens representing the board state:

| Position | Content | Token Count |
|----------|---------|-------------|
| 0-63 | Square contents (a1, b1, ..., h8) | 64 |
| 64 | Turn indicator | 1 |
| 65 | White kingside castling | 1 |
| 66 | White queenside castling | 1 |
| 67 | Black kingside castling | 1 |
| 68 | Black queenside castling | 1 |

#### 1.2.2 Token Vocabulary

```
Piece tokens (13):
  ES    = Empty square
  WP    = White pawn
  WN    = White knight
  WB    = White bishop
  WR    = White rook
  WQ    = White queen
  WK    = White king
  BP-BK = Black pieces (same pattern)

Flag tokens (10):
  TURN_W, TURN_B           = Turn indicator
  CASTLE_WK_Y, CASTLE_WK_N = White kingside castling rights
  CASTLE_WQ_Y, CASTLE_WQ_N = White queenside castling rights
  CASTLE_BK_Y, CASTLE_BK_N = Black kingside castling rights
  CASTLE_BQ_Y, CASTLE_BQ_N = Black queenside castling rights

Total vocabulary size: 23 tokens
```

#### 1.2.3 Move Encoding

Moves encoded as a single integer in range [0, 4095]:

```
move_index = from_square * 64 + to_square
```

Where squares are indexed 0-63 (a1=0, b1=1, ..., h8=63).

**Promotion Handling**: For Phase 1, promotions default to queen. Phase 2 may extend the move space to 4672 (64×64 + 64×4×3 for underpromotions) if needed.

**Note on Move Legality**: The 4096 move space contains many illegal moves for any given position. The model must learn to assign near-zero probability to illegal moves. During inference, illegal moves are masked before sampling.

### 1.3 Data Generation

#### 1.3.1 Random Play Dataset

Generate training data by playing random legal moves:

```python
def generate_game():
    board = chess.Board()
    positions = []
    
    while not board.is_game_over() and len(positions) < MAX_GAME_LENGTH:
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        
        positions.append({
            'board': encode_board(board),
            'move': encode_move(move),
            'legal_moves': [encode_move(m) for m in legal_moves],
            'outcome': None  # Filled after game ends
        })
        
        board.push(move)
    
    # Backfill outcomes
    outcome = get_outcome(board)  # +1 white wins, -1 black wins, 0 draw
    for i, pos in enumerate(positions):
        # Outcome from perspective of side to move
        pos['outcome'] = outcome if (i % 2 == 0) else -outcome
    
    return positions
```

#### 1.3.2 Dataset Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Number of games | 50,000 | ~2M positions, sufficient for rule learning |
| Max game length | 200 moves | Prevent infinite games |
| Min game length | 10 moves | Skip trivial games |
| Train/Val split | 90/10 | Standard split |

#### 1.3.3 Data Storage Format

```json
{
  "metadata": {
    "version": "1.0",
    "generated": "2024-01-15T10:30:00Z",
    "num_games": 50000,
    "num_positions": 2150000
  },
  "samples": [
    {
      "board": [0, 1, 2, ...],  // 69 integers
      "move": 1234,             // Single integer [0, 4095]
      "legal_moves": [52, 116, 180, ...],
      "outcome": 1              // +1, 0, -1
    }
  ]
}
```

**Storage Consideration**: For 2M positions, JSON will be ~2GB. Consider using memory-mapped numpy arrays or HDF5 for faster loading:

```python
# Recommended: numpy memmap
boards = np.memmap('boards.dat', dtype=np.int8, mode='w+', shape=(N, 69))
moves = np.memmap('moves.dat', dtype=np.int16, mode='w+', shape=(N,))
```

### 1.4 Model Architecture

#### 1.4.1 Network Design

Encoder-only transformer with dual heads (policy + value):

```
Input: (batch, 69) board tokens
    ↓
Token Embedding: (batch, 69, d_model)
    +
Position Embedding: (batch, 69, d_model)
    ↓
Transformer Encoder: N layers, full attention (no causal mask)
    ↓
Mean Pool: (batch, d_model)
    ↓
    ├── Policy Head → (batch, 4096) move logits
    └── Value Head  → (batch, 1) position evaluation
```

#### 1.4.2 Hyperparameters

| Parameter | Phase 1 Value | Notes |
|-----------|---------------|-------|
| d_model | 256 | Embedding dimension |
| n_layers | 6 | Transformer depth |
| n_heads | 8 | Attention heads |
| d_ff | 1024 | Feed-forward dimension |
| dropout | 0.1 | Regularisation |
| max_seq_len | 69 | Fixed board encoding |

**Estimated Parameters**: ~4M (small enough to iterate quickly)

#### 1.4.3 Position Encoding Strategy

**Problem**: Flat `nn.Embedding(69, d_model)` forces the model to learn that square 0 (a1) is adjacent to square 8 (a2). This wastes capacity and training time.

**Solution**: Factored Rank + File embeddings that hard-code the 8×8 grid structure.

```python
class ChessPositionEncoding(nn.Module):
    """
    Factored position encoding for chess board.
    
    Squares 0-63: rank_embed[sq // 8] + file_embed[sq % 8]
    Flags 64-68:  flag_embed[idx]
    
    This encodes the 2D grid structure directly - the model knows
    a1 and a2 share a file embedding without learning it.
    """
    def __init__(self, d_model):
        super().__init__()
        self.rank_embed = nn.Embedding(8, d_model)  # Ranks 1-8
        self.file_embed = nn.Embedding(8, d_model)  # Files a-h
        self.flag_embed = nn.Embedding(5, d_model)  # 5 flag positions
        
    def forward(self, seq_len, device):
        # Squares 0-63: factored rank + file
        squares = torch.arange(64, device=device)
        ranks = squares // 8
        files = squares % 8
        square_pos = self.rank_embed(ranks) + self.file_embed(files)
        
        # Flags 64-68: separate embeddings
        flag_indices = torch.arange(5, device=device)
        flag_pos = self.flag_embed(flag_indices)
        
        # Concatenate: (69, d_model)
        return torch.cat([square_pos, flag_pos], dim=0)
```

**Fallback** (if Solution A insufficient): Relative position encoding (T5/Swin style) that adds attention bias based on Chebyshev distance between squares. This modifies the attention mechanism itself:

```python
def chebyshev_distance(sq1, sq2):
    """King distance between two squares."""
    r1, f1 = sq1 // 8, sq1 % 8
    r2, f2 = sq2 // 8, sq2 % 8
    return max(abs(r1 - r2), abs(f1 - f2))

# Pre-compute 64×64 distance matrix for attention bias
# Only implement if factored embeddings fail
```

#### 1.4.4 Value Head Design

**Problem**: Value head collapse. In random play, draws are common and wins/losses are noisy. The model learns "predict 0 everywhere" to minimise MSE - a local minimum it never escapes.

**Solution**: Discrete outcome classification instead of scalar regression.

```python
# Instead of:
# self.value_head = nn.Sequential(..., nn.Linear(d_model, 1), nn.Tanh())

# Use 3-class classification:
self.value_head = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 3)  # [Loss, Draw, Win]
)

# Training target: one-hot [1,0,0] for loss, [0,1,0] for draw, [0,0,1] for win
# Loss: CrossEntropyLoss (sharper gradients than MSE)

# Inference: convert back to scalar
def value_from_logits(logits):
    probs = F.softmax(logits, dim=-1)  # (batch, 3)
    # value = P(Win) - P(Loss)
    return probs[:, 2] - probs[:, 0]
```

**Why this works**: Cross-entropy gradients are much sharper than MSE when predictions are "kinda wrong". A prediction of [0.4, 0.5, 0.1] for a winning position gets strong gradient signal, whereas MSE on 0.3 vs 1.0 is relatively weak.

**Additional Mitigation - Mate Puzzles**: Include "Mate in 1" positions in Phase 1 training data. These provide unambiguous +1/-1 labels that anchor the value head to extreme positions. Generate ~10K mate-in-1 puzzles or download from Lichess puzzle database.

#### 1.4.5 Model Implementation

```python
class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=23, num_moves=4096, d_model=256, 
                 n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = ChessPositionEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # Policy head: predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_moves)
        )
        
        # Value head: 3-class outcome prediction [Loss, Draw, Win]
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
    
    def forward(self, x):
        B, S = x.shape
        
        # Factored position encoding
        pos = self.pos_encode(S, x.device)  # (69, d_model)
        x = self.tok_embed(x) + pos.unsqueeze(0)  # (B, 69, d_model)
        
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))  # Mean pool
        
        policy = self.policy_head(x)
        value_logits = self.value_head(x)
        
        return policy, value_logits
    
    def get_value(self, value_logits):
        """Convert 3-class logits to scalar value [-1, +1]."""
        probs = F.softmax(value_logits, dim=-1)
        return probs[:, 2] - probs[:, 0]  # P(Win) - P(Loss)
```

**Gradient Monitoring**: During training, log the gradient norms for policy vs value heads separately. They should be within an order of magnitude. If value gradients are 10× smaller, increase `value_weight`.

### 1.5 Training Procedure

#### 1.5.1 Loss Function

Combined policy and value loss with 3-class value head:

```python
def compute_loss(policy_logits, value_logits, target_move, target_outcome, 
                 legal_mask=None, policy_weight=1.0, value_weight=1.0):
    """
    Args:
        policy_logits: (batch, 4096) raw move logits
        value_logits: (batch, 3) outcome class logits [Loss, Draw, Win]
        target_move: (batch,) move index
        target_outcome: (batch,) outcome as class index: 0=Loss, 1=Draw, 2=Win
        legal_mask: (batch, 4096) boolean mask of legal moves
    """
    # Policy loss: cross-entropy on the move played
    policy_loss = F.cross_entropy(policy_logits, target_move)
    
    # Value loss: cross-entropy on outcome class (sharper gradients than MSE)
    value_loss = F.cross_entropy(value_logits, target_outcome)
    
    # Optional: auxiliary loss for legal move prediction
    # Encourages model to assign low probability to illegal moves
    if legal_mask is not None:
        illegal_probs = F.softmax(policy_logits, dim=-1) * (~legal_mask).float()
        illegal_loss = illegal_probs.sum(dim=-1).mean()
    else:
        illegal_loss = 0
    
    return policy_weight * policy_loss + value_weight * value_loss + 0.1 * illegal_loss


def outcome_to_class(outcome):
    """Convert scalar outcome to class index.
    
    outcome: -1 (loss), 0 (draw), +1 (win)
    returns: 0 (loss), 1 (draw), 2 (win)
    """
    return outcome + 1  # Maps -1,0,1 → 0,1,2
```

#### 1.5.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 512 |
| Learning rate | 1e-3 |
| LR schedule | Cosine decay with 1000-step warmup |
| Weight decay | 0.01 |
| Epochs | 50 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW |

#### 1.5.3 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Legal Accuracy | Top-1 prediction is legal | > 95% |
| Exact Accuracy | Predicted the actual move | > 15% |
| Top-5 Accuracy | True move in top 5 predictions | > 40% |
| Value MSE | Mean squared error on outcome | < 0.5 |
| Illegal Probability | Mean prob mass on illegal moves | < 1% |

#### 1.5.4 Checkpointing

Save checkpoints every 5 epochs and on best validation loss:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_metrics': val_metrics,
    'config': config
}
torch.save(checkpoint, f'checkpoints/phase1_epoch{epoch}.pt')
```

### 1.6 Validation Protocol

#### 1.6.1 Move Generation Test

Verify model generates only legal moves when sampled:

```python
def test_legal_generation(model, n_games=100, n_moves=50):
    illegal_count = 0
    total_moves = 0
    
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(n_moves):
            if board.is_game_over():
                break
            
            policy, _ = model(encode_board(board))
            legal_mask = get_legal_mask(board)
            
            # Mask and sample
            policy[~legal_mask] = float('-inf')
            probs = F.softmax(policy, dim=-1)
            move_idx = torch.multinomial(probs, 1).item()
            
            # Verify
            move = decode_move(move_idx)
            if move not in board.legal_moves:
                illegal_count += 1
            
            total_moves += 1
            board.push(move)
    
    return illegal_count / total_moves  # Should be 0%
```

#### 1.6.2 Position Understanding Spot Checks

Test specific scenarios to verify rule learning:

| Scenario | Test |
|----------|------|
| Checkmate recognition | Model should not move into checkmate |
| Pin awareness | Pinned pieces should have reduced move probs |
| Castling | Model should castle when rights exist |
| En passant | Model should recognise en passant captures |
| Promotion | Pawns on 7th rank should promote |

---

## Phase 2: Self-Play Reinforcement Learning

### 2.1 Objective

Train the model to play strong chess through self-play, using Monte Carlo Tree Search (MCTS) to generate improved training targets.

**Success Criteria**:
- Elo improvement over Phase 1 baseline: > 500 Elo
- Win rate vs Phase 1 model: > 80%
- Positive Elo trend over training iterations

### 2.2 MCTS Implementation

#### 2.2.1 Core Algorithm

```
function MCTS(root_state, model, num_simulations):
    root = Node(root_state)
    
    for i in 1 to num_simulations:
        node = root
        state = root_state.copy()
        
        # Selection: traverse tree using UCB
        while node.is_expanded and not state.is_terminal:
            node = select_child(node)
            state.apply_move(node.move)
        
        # Expansion: add children for unexpanded node
        if not state.is_terminal:
            policy, value = model(state)
            expand_node(node, policy, state.legal_moves)
        else:
            value = get_terminal_value(state)
        
        # Backpropagation: update visit counts and values
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent
    
    return root.get_policy()  # Visit count distribution
```

#### 2.2.2 Node Structure

```python
@dataclass
class MCTSNode:
    state: BoardState
    move: Optional[int]  # Move that led to this node
    parent: Optional['MCTSNode']
    children: Dict[int, 'MCTSNode']  # move -> child node
    
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy prior from neural net
    
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def ucb_score(self, parent_visits, c_puct=1.5):
        exploration = c_puct * self.prior * sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration
```

#### 2.2.3 MCTS Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_simulations | 800 | Simulations per move |
| c_puct | 1.5 | Exploration constant |
| temperature | 1.0 → 0.1 | Move selection temperature |
| temp_threshold | 30 | Switch to low temp after N moves |
| dirichlet_alpha | 0.3 | Noise for root exploration |
| dirichlet_frac | 0.25 | Fraction of noise to add |

#### 2.2.4 Temperature Schedule

```python
def get_temperature(move_number, temp_threshold=30):
    """
    High temperature early game (exploration) → Low temperature late game (exploitation)
    """
    if move_number < temp_threshold:
        return 1.0
    else:
        return 0.1  # Near-greedy
```

### 2.3 Self-Play Game Generation

#### 2.3.1 Game Generation Loop

```python
def generate_self_play_game(model, mcts_config):
    board = chess.Board()
    game_data = []
    
    while not board.is_game_over() and len(game_data) < MAX_GAME_LENGTH:
        # Run MCTS from current position
        mcts_policy = run_mcts(board, model, mcts_config)
        
        # Sample move using temperature
        temp = get_temperature(len(game_data))
        if temp < 0.2:
            move = argmax(mcts_policy)
        else:
            move = sample(mcts_policy, temperature=temp)
        
        # Store training data
        game_data.append({
            'board': encode_board(board),
            'mcts_policy': mcts_policy,  # Target for policy head
            'move_played': move
        })
        
        board.push(decode_move(move))
    
    # Determine outcome
    outcome = get_outcome(board)
    
    # Annotate each position with outcome (from side-to-move perspective)
    for i, data in enumerate(game_data):
        data['outcome'] = outcome if (i % 2 == 0) else -outcome
    
    return game_data
```

#### 2.3.2 Parallelisation Strategy

With 96GB unified memory, parallelise self-play across multiple workers:

```
┌─────────────────────────────────────────────┐
│            Self-Play Coordinator            │
├─────────────────────────────────────────────┤
│  Worker 1  │  Worker 2  │  ...  │ Worker N  │
│  (Game 1)  │  (Game 2)  │  ...  │ (Game N)  │
│     ↓           ↓                    ↓      │
│  Shared Model (read-only during generation) │
│     ↓           ↓                    ↓      │
│         Replay Buffer (shared)              │
└─────────────────────────────────────────────┘
```

**Recommended Workers**: 8-16 parallel games (limited by MCTS GPU batching efficiency)

### 2.4 Training Loop

#### 2.4.1 Training Pipeline

```python
def training_iteration(model, replay_buffer, config):
    # Generate self-play games
    games = parallel_self_play(model, config.games_per_iteration)
    replay_buffer.add(games)
    
    # Sample training batch
    for step in range(config.training_steps):
        batch = replay_buffer.sample(config.batch_size)
        
        # Forward pass
        policy_logits, value_pred = model(batch['boards'])
        
        # Policy loss: KL divergence from MCTS policy
        policy_target = batch['mcts_policy']
        policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        
        # Value loss: MSE from game outcome
        value_loss = F.mse_loss(value_pred.squeeze(), batch['outcome'])
        
        # Combined loss
        loss = policy_loss + config.value_weight * value_loss
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
    
    return model
```

#### 2.4.2 Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, games):
        for game in games:
            for position in game:
                self.buffer.append(position)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return collate(batch)
    
    def __len__(self):
        return len(self.buffer)
```

#### 2.4.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Games per iteration | 500 |
| Training steps per iteration | 1000 |
| Batch size | 256 |
| Learning rate | 1e-4 |
| LR schedule | Cosine decay |
| Weight decay | 1e-4 |
| Value loss weight | 1.0 |
| Gradient clipping | 1.0 |
| Replay buffer size | 500,000 positions |
| Minimum buffer size | 10,000 (before training starts) |

### 2.5 Evaluation & Elo Tracking

#### 2.5.1 Evaluation Protocol

Every N iterations, evaluate current model against:
1. Previous best checkpoint
2. Phase 1 baseline model
3. Pool of historical checkpoints

```python
def evaluate_model(challenger, opponent, n_games=100):
    wins = draws = losses = 0
    
    for i in range(n_games):
        # Alternate colours
        white = challenger if i % 2 == 0 else opponent
        black = opponent if i % 2 == 0 else challenger
        
        result = play_game(white, black, mcts_simulations=400)
        
        if result == (1 if i % 2 == 0 else -1):
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    
    return wins, draws, losses
```

#### 2.5.2 Elo Calculation

```python
def update_elo(rating_a, rating_b, score_a, k=32):
    """
    score_a: 1 for win, 0.5 for draw, 0 for loss
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + k * (score_a - expected_a)
    return new_rating_a
```

#### 2.5.3 Checkpoint Selection

Only update "best model" if new model beats current best:
- Minimum 55% win rate (accounting for variance)
- Or significant Elo gain (>30 points)

### 2.6 Preventing Collapse

#### 2.6.1 Root Dirichlet Noise

Add noise to root node priors to ensure exploration:

```python
def add_dirichlet_noise(priors, legal_moves, alpha=0.3, frac=0.25):
    noise = np.random.dirichlet([alpha] * len(legal_moves))
    for i, move in enumerate(legal_moves):
        priors[move] = (1 - frac) * priors[move] + frac * noise[i]
    return priors
```

#### 2.6.2 Opening Randomisation

For first N moves, sample from uniform distribution over legal moves:

```python
def should_randomise(move_number, randomise_moves=8):
    return move_number < randomise_moves and random.random() < 0.5
```

#### 2.6.3 Resign Threshold

Don't waste compute on clearly lost games:

```python
def should_resign(value, threshold=-0.9, consecutive_required=3):
    """Resign if value below threshold for N consecutive moves"""
    # Track in game state
    pass
```

### 2.7 Memory & Compute Considerations

#### 2.7.1 MCTS Memory Usage

Per MCTS tree:
- ~1000 nodes typical
- ~500 bytes per node
- ~500KB per tree
- 16 parallel games: ~8MB

#### 2.7.2 Batch Inference Optimisation

Batch neural network evaluations across MCTS simulations:

```python
class BatchedMCTS:
    """
    Collect leaf nodes from multiple trees, batch evaluate, then continue.
    Significantly faster than sequential evaluation.
    """
    def __init__(self, model, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self.pending = []  # (tree_id, node, state)
    
    def add_evaluation_request(self, tree_id, node, state):
        self.pending.append((tree_id, node, state))
        if len(self.pending) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.pending:
            return
        
        states = torch.stack([encode(s) for _, _, s in self.pending])
        policies, values = self.model(states)
        
        for (tree_id, node, _), policy, value in zip(self.pending, policies, values):
            node.expand(policy)
            node.backprop(value)
        
        self.pending = []
```

#### 2.7.3 Training Memory Budget

| Component | Memory |
|-----------|--------|
| Model (4M params, fp32) | ~16MB |
| Model (training, gradients) | ~48MB |
| Replay buffer (500K positions) | ~2GB |
| MCTS trees (16 parallel) | ~8MB |
| Batch data | ~100MB |
| **Total** | ~3GB |

Plenty of headroom on 96GB system. Consider:
- Larger model (16M params)
- Larger replay buffer (2M positions)
- More parallel self-play workers

---

## Project Structure

```
tiny_alphazero/
├── README.md
├── pyproject.toml
├── config/
│   ├── phase1.yaml
│   └── phase2.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py      # ChessTransformer
│   │   └── utils.py            # Weight init, loading
│   ├── data/
│   │   ├── __init__.py
│   │   ├── encoding.py         # Board/move encoding
│   │   ├── generation.py       # Random game generation
│   │   └── dataset.py          # PyTorch Dataset
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── node.py             # MCTSNode
│   │   ├── tree.py             # MCTS search
│   │   └── batched.py          # Batched inference
│   ├── training/
│   │   ├── __init__.py
│   │   ├── phase1.py           # Supervised training
│   │   ├── phase2.py           # Self-play training
│   │   └── evaluation.py       # Elo tracking
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Config loading
│       └── logging.py          # Metrics logging
├── scripts/
│   ├── generate_data.py        # Phase 1 data generation
│   ├── train_phase1.py         # Phase 1 training
│   ├── train_phase2.py         # Phase 2 self-play loop
│   ├── evaluate.py             # Model evaluation
│   └── play.py                 # Interactive play
├── tests/
│   ├── test_encoding.py
│   ├── test_mcts.py
│   └── test_model.py
└── checkpoints/
    └── .gitkeep
```

---

## Development Milestones

### Milestone 1: Data Pipeline (Week 1)
- [ ] Board encoding/decoding with round-trip tests
- [ ] Move encoding/decoding with legality verification
- [ ] Random game generator
- [ ] Dataset class with efficient loading
- [ ] Generate 50K games (~2M positions)

### Milestone 2: Phase 1 Model (Week 2)
- [ ] Transformer architecture implementation
- [ ] Training loop with logging
- [ ] Validation metrics (legal acc, exact acc, top-5, value MSE)
- [ ] Checkpoint saving/loading
- [ ] Achieve >95% legal move accuracy

### Milestone 3: MCTS (Week 3)
- [ ] Node structure and tree operations
- [ ] UCB selection
- [ ] Expansion with policy priors
- [ ] Backpropagation
- [ ] Batched inference for efficiency
- [ ] Unit tests for MCTS correctness

### Milestone 4: Self-Play Pipeline (Week 4)
- [ ] Single-threaded self-play game generation
- [ ] Replay buffer implementation
- [ ] Phase 2 training loop
- [ ] Elo evaluation system
- [ ] Parallelised self-play

### Milestone 5: Training & Iteration (Weeks 5-8)
- [ ] Run Phase 2 training
- [ ] Monitor Elo progression
- [ ] Tune hyperparameters
- [ ] Iterate on architecture if needed
- [ ] Achieve target Elo improvement

---

## Technical Risks & Mitigations

### Risk 1: Python MCTS Speed Trap

**The Problem**: Pure Python MCTS is slow. Each simulation involves tree traversal, dictionary lookups, and object creation. Phase 2 requires millions of simulations - at 800 sims/move × 100 moves/game × 500 games/iteration, that's 40M simulations per iteration. Python overhead will bottleneck training.

**Severity**: High - could make Phase 2 impractically slow.

**Mitigation Strategy** (progressive):

1. **Start with pure Python** (Week 3)
   - Prioritise correctness and clarity
   - Profile to identify actual bottlenecks
   - Establish baseline: target >50 sims/second/game

2. **Numba JIT compilation** (if needed)
   - Rewrite hot path (node selection, UCB calculation) with `@numba.jit`
   - Use typed data structures instead of classes
   ```python
   import numba
   from numba import int32, float32
   from numba.experimental import jitclass
   
   node_spec = [
       ('visit_count', int32),
       ('value_sum', float32),
       ('prior', float32),
       ('children_indices', int32[:]),
   ]
   
   @jitclass(node_spec)
   class FastNode:
       ...
   ```
   - Expected speedup: 10-50×

3. **Rust/C++ extension** (if Numba insufficient)
   - Rewrite tree search as Python extension
   - Keep neural network calls in Python
   - Use PyO3 (Rust) or pybind11 (C++)
   - Expected speedup: 50-100×

4. **Batched MCTS** (orthogonal optimisation)
   - Run multiple trees in parallel
   - Batch leaf evaluations across trees
   - Already specified in §2.7.2 - implement early

**Monitoring**: Track sims/second during development. If <100 sims/sec after batching, escalate to Numba.

### Risk 2: Random Play Data Quality

**The Problem**: Purely random chess produces low-quality training signal:
- Games often end in drawn-out shuffling
- Tactical patterns (forks, pins, skewers) rarely occur
- "Don't hang your queen" is never demonstrated
- Model may learn that any legal move is equally good

**Severity**: Medium - Phase 1 might plateau at legal move prediction without learning move quality.

**Mitigation Strategy** (progressive):

1. **Start with pure random** (baseline)
   - Establish what the model learns from noise
   - Legal move accuracy should still reach >90%

2. **Biased sampling** (if legal acc is fine but value head is poor)
   ```python
   def biased_move_selection(board, capture_weight=3.0, check_weight=2.0):
       """Weight moves by tactical interest."""
       legal_moves = list(board.legal_moves)
       weights = []
       
       for move in legal_moves:
           w = 1.0
           if board.is_capture(move):
               w *= capture_weight
           board.push(move)
           if board.is_check():
               w *= check_weight
           board.pop()
           weights.append(w)
       
       weights = np.array(weights)
       weights /= weights.sum()
       return np.random.choice(legal_moves, p=weights)
   ```

3. **MVV-LVA for captures** (more sophisticated)
   ```python
   # Most Valuable Victim - Least Valuable Attacker
   PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
   
   def mvv_lva_score(board, move):
       if not board.is_capture(move):
           return 0
       victim = board.piece_at(move.to_square)
       attacker = board.piece_at(move.from_square)
       return PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]
   ```

4. **Mix in engine games** (nuclear option)
   - Generate 10% of games using Stockfish at low depth (depth=1-3)
   - Provides examples of "sensible" play
   - Risk: model might just memorise engine patterns
   - Only use if biased random fails

**Monitoring**: Track value head MSE and correlation with actual outcomes. If value head predicts ~0 for everything, data quality is the issue.

**Data Quality Metrics**:
| Metric | Random | Target |
|--------|--------|--------|
| Avg game length | ~80 | ~60 |
| % games with captures | ~95% | ~98% |
| % decisive games | ~50% | ~70% |
| Avg material imbalance | low | medium |

### Risk 3: Illegal Move Mask Efficiency

**The Problem**: With 4096 move logits, masking illegal moves naively is expensive:
- Generating legal moves: ~1μs (python-chess is fast)
- Creating boolean mask: ~10μs
- Applying mask in training loop: adds up over millions of samples

**Severity**: Low-Medium - won't break anything, but can slow training 2-3×.

**Mitigation Strategy**:

1. **Pre-compute masks during data generation** (already in spec)
   ```python
   # In generate_game()
   legal_mask = torch.zeros(4096, dtype=torch.bool)
   for move in board.legal_moves:
       legal_mask[encode_move(move)] = True
   sample['legal_mask'] = legal_mask
   ```

2. **Store masks efficiently**
   ```python
   # Don't store full 4096 booleans - store indices
   sample['legal_move_indices'] = [encode_move(m) for m in board.legal_moves]
   
   # Reconstruct mask in DataLoader (on GPU)
   def collate_fn(batch):
       masks = torch.zeros(len(batch), 4096, dtype=torch.bool, device='cuda')
       for i, sample in enumerate(batch):
           masks[i, sample['legal_move_indices']] = True
       return ..., masks
   ```

3. **Use sparse representation for very large batches**
   ```python
   # For batch_size=512, full masks = 512 × 4096 × 1 byte = 2MB
   # Sparse: avg 30 legal moves × 2 bytes × 512 = 30KB
   ```

4. **Fused masking operation**
   ```python
   # Instead of:
   logits[~mask] = float('-inf')
   probs = F.softmax(logits, dim=-1)
   
   # Use:
   probs = F.softmax(logits.masked_fill(~mask, float('-inf')), dim=-1)
   # Single fused kernel, no intermediate tensor
   ```

**Training Loop Integration**:
```python
def compute_loss(policy_logits, legal_masks, target_moves, ...):
    # Mask BEFORE softmax for numerical stability
    masked_logits = policy_logits.masked_fill(~legal_masks, float('-inf'))
    
    # Cross-entropy with masked logits
    policy_loss = F.cross_entropy(masked_logits, target_moves)
    
    # Optional: penalise probability mass on illegal moves (regularisation)
    # This encourages the model to learn legality even without masking
    raw_probs = F.softmax(policy_logits, dim=-1)  # Unmasked
    illegal_prob = (raw_probs * (~legal_masks).float()).sum(dim=-1).mean()
    
    return policy_loss + 0.1 * illegal_prob
```

**Note on Inference**: During MCTS, masks must be computed on-the-fly since positions are novel. This is unavoidable but fast (~30 legal moves to encode per position).

### Risk Summary Table

| Risk | Severity | Phase Affected | Mitigation Cost | Fallback Ready |
|------|----------|----------------|-----------------|----------------|
| Python MCTS speed | High | Phase 2 | Medium (Numba) | Yes |
| Random data quality | Medium | Phase 1 | Low (biased sampling) | Yes |
| Mask efficiency | Low | Both | Low (pre-compute) | N/A |

### Additional Risks (Lower Priority)

**Risk 4: Transformer Position Encoding**
- Problem: Learned positional embeddings may not capture 8×8 grid structure
- Mitigation: If spatial reasoning is weak, try 2D sinusoidal encodings or factored (rank × file) embeddings

**Risk 5: Value Head Collapse**
- Problem: Value head outputs constant ~0 regardless of position
- Mitigation: Increase value loss weight, add auxiliary value targets (material count)

**Risk 6: Self-Play Mode Collapse**

Problem: The bot finds one trick (e.g., Scholar's Mate, specific pawn storm) that beats its previous self and plays it 100% of the time. The model overfits to itself and fails against diverse opponents.

**Mitigation A: Forced Random Openings (Critical)**

AlphaZero relied on massive batch sizes to get diversity. TinyAlphaZero doesn't have that luxury.

```python
def generate_self_play_game(model, mcts_config, random_opening_ply=6):
    board = chess.Board()
    game_data = []
    
    while not board.is_game_over() and len(game_data) < MAX_GAME_LENGTH:
        move_num = len(game_data)
        
        # Force random moves for first N ply
        if move_num < random_opening_ply and random.random() < 0.5:
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            mcts_policy = None  # Don't train on random moves
        else:
            mcts_policy = run_mcts(board, model, mcts_config)
            move = sample_move(mcts_policy, temperature=get_temperature(move_num))
        
        if mcts_policy is not None:
            game_data.append({
                'board': encode_board(board),
                'mcts_policy': mcts_policy,
                'move_played': move
            })
        
        board.push(decode_move(move))
    
    return game_data
```

**Mitigation B: Aggressive Temperature Schedule**

Never go fully deterministic during training data generation:

```python
def get_temperature(move_number, temp_threshold=15):
    """
    More aggressive than original spec:
    - First 10 moves: temp=1.5 (high exploration)
    - Moves 10-15: temp=1.0
    - After 15: temp=0.3 (near-greedy but not argmax)
    
    Key: Never use temp=0 (argmax) for training games.
    """
    if move_number < 10:
        return 1.5
    elif move_number < temp_threshold:
        return 1.0
    else:
        return 0.3  # NOT 0.0 - always some randomness
```

**Mitigation C: Checkpoint Pool (Latest Window)**

Don't just play against "best" checkpoint - maintain diversity:

```python
class CheckpointPool:
    """
    Maintain sliding window of recent checkpoints.
    Self-play picks random opponent from pool, not just current/best.
    """
    def __init__(self, max_size=20):
        self.checkpoints = deque(maxlen=max_size)
    
    def add(self, checkpoint_path, elo=None):
        self.checkpoints.append({'path': checkpoint_path, 'elo': elo})
    
    def sample_opponent(self):
        """Random checkpoint from pool - prevents A beats B, B beats C, C beats A cycles."""
        return random.choice(self.checkpoints)['path']
    
    def get_latest(self):
        return self.checkpoints[-1]['path'] if self.checkpoints else None


# In training loop:
def self_play_iteration(current_model, checkpoint_pool, games_per_iteration):
    games = []
    for _ in range(games_per_iteration):
        # 50% self-play, 50% vs random historical checkpoint
        if random.random() < 0.5 and len(checkpoint_pool) > 1:
            opponent_path = checkpoint_pool.sample_opponent()
            opponent = load_model(opponent_path)
        else:
            opponent = current_model
        
        game = play_game(current_model, opponent)
        games.append(game)
    
    return games
```

**Mitigation D: Elo Stagnation Detection**

If Elo stops improving, trigger diversity interventions:

```python
class EloTracker:
    def __init__(self, stagnation_threshold=50, stagnation_iters=10):
        self.history = []
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_iters = stagnation_iters
    
    def update(self, elo):
        self.history.append(elo)
    
    def is_stagnating(self):
        if len(self.history) < self.stagnation_iters:
            return False
        recent = self.history[-self.stagnation_iters:]
        return max(recent) - min(recent) < self.stagnation_threshold
    
    def get_intervention(self):
        """Returns config changes to break stagnation."""
        return {
            'random_opening_ply': 10,  # Increase from 6
            'temperature_multiplier': 1.5,  # More exploration
            'dirichlet_frac': 0.4,  # More root noise (up from 0.25)
        }
```

---

## Configuration Files

### config/phase1.yaml

```yaml
# Phase 1: Supervised Learning Configuration

data:
  num_games: 50000
  max_game_length: 200
  min_game_length: 10
  val_split: 0.1
  output_dir: data/phase1
  # Biased sampling for richer positions
  capture_weight: 3.0
  check_weight: 2.0
  # Include mate puzzles to anchor value head
  mate_puzzles:
    enabled: true
    count: 10000
    source: "lichess"  # or "generate"

model:
  vocab_size: 23
  num_moves: 4096
  num_value_classes: 3  # [Loss, Draw, Win]
  d_model: 256
  n_layers: 6
  n_heads: 8
  dropout: 0.1
  position_encoding: "factored"  # "factored" or "learned"

training:
  batch_size: 512
  epochs: 50
  learning_rate: 1.0e-3
  weight_decay: 0.01
  warmup_steps: 1000
  grad_clip: 1.0
  policy_weight: 1.0
  value_weight: 2.0  # Increased to balance with policy loss

evaluation:
  eval_every: 1  # epochs
  save_every: 5  # epochs
  checkpoint_dir: checkpoints/phase1

logging:
  log_every: 100  # steps
  log_grad_norms: true  # Monitor policy vs value gradient balance
  wandb_project: tiny-alphazero
  wandb_run: phase1
```

### config/phase2.yaml

```yaml
# Phase 2: Self-Play RL Configuration

model:
  # Inherit from Phase 1 or specify
  checkpoint: checkpoints/phase1/best.pt

mcts:
  num_simulations: 800
  c_puct: 1.5
  dirichlet_alpha: 0.3
  dirichlet_frac: 0.25
  resign_threshold: -0.9

self_play:
  games_per_iteration: 500
  max_game_length: 200
  num_workers: 8
  # Anti-collapse settings
  random_opening_ply: 6          # Force random moves for first N ply
  random_opening_prob: 0.5       # Probability of random move in opening
  # Temperature: aggressive schedule to prevent collapse
  temperature:
    early_game_temp: 1.5         # Moves 0-10
    mid_game_temp: 1.0           # Moves 10-15
    late_game_temp: 0.3          # Moves 15+, NOT 0.0
    early_game_moves: 10
    mid_game_moves: 15
  # Opponent diversity
  self_play_ratio: 0.5           # 50% self-play, 50% vs checkpoint pool

training:
  steps_per_iteration: 1000
  batch_size: 256
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  value_weight: 1.0

replay_buffer:
  capacity: 500000
  min_size: 10000

evaluation:
  games_per_eval: 100
  eval_every: 5  # iterations
  mcts_simulations_eval: 400
  elo_k_factor: 32
  # Stagnation detection
  stagnation_threshold: 50       # Elo points
  stagnation_iters: 10           # Iterations before triggering intervention

checkpointing:
  save_every: 10  # iterations
  checkpoint_dir: checkpoints/phase2
  pool_size: 20                  # Sliding window of checkpoints for diversity

logging:
  log_every: 10  # steps
  wandb_project: tiny-alphazero
  wandb_run: phase2
```

---

## Testing Requirements

### Unit Tests

```python
# tests/test_encoding.py
def test_board_encoding_roundtrip():
    """Encode then decode should return identical board"""
    
def test_move_encoding_all_legal():
    """All legal moves should encode to valid indices"""
    
def test_starting_position():
    """Starting position should encode correctly"""

# tests/test_mcts.py
def test_mcts_legal_moves_only():
    """MCTS should only consider legal moves"""
    
def test_ucb_exploration():
    """Unvisited nodes should be prioritised"""
    
def test_backprop_flips_value():
    """Value should flip sign up the tree"""

# tests/test_model.py
def test_model_output_shapes():
    """Policy should be (B, 4096), value should be (B, 1)"""
    
def test_legal_masking():
    """Masked logits should produce valid probability distribution"""
```

### Integration Tests

```python
def test_full_game_generation():
    """Generate a complete game without errors"""
    
def test_self_play_game():
    """Run MCTS self-play for one game"""
    
def test_training_step():
    """Complete one training step without errors"""
```

---

## Dependencies

```toml
[project]
name = "tiny-alphazero"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.0",
    "chess>=1.10",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "wandb>=0.15",  # Optional: for logging
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]
```

---

## Appendix A: Extended Move Encoding (Optional)

For handling underpromotions properly, extend move space:

```
Standard moves: from_sq × to_sq = 64 × 64 = 4096
Promotions: from_sq × {Q, R, B, N} = 64 × 4 = 256 per promotion rank

Total: 4096 + 256 = 4352 (simplified)

Or AlphaZero-style:
- 56 queen moves (8 directions × 7 distances)
- 8 knight moves
- 9 underpromotions (3 directions × 3 piece types)
= 73 move types per square × 64 squares = 4672 total
```

For Phase 1, the simpler 4096 encoding with queen-only promotion is sufficient.

---

## Appendix B: Debugging Self-Play

Common failure modes and diagnostics:

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Elo not improving | Insufficient MCTS depth | Increase simulations |
| Games too short | Resign threshold too aggressive | Raise threshold |
| Policy collapse | Not enough exploration | Increase Dirichlet noise |
| Value always ~0 | Value head not training | Check value loss, increase weight |
| Illegal moves | Masking bug | Verify legal_mask generation |
| OOM during MCTS | Tree not pruned | Clear old trees between moves |

---

## Appendix C: Potential Enhancements

For future iterations beyond MVP:

1. **Pondering**: Continue MCTS during opponent's turn
2. **Opening book**: Load standard openings for variety
3. **Endgame tablebases**: Perfect play in simple endgames
4. **Larger model**: Scale to 16M-64M parameters
5. **Distributed training**: Multi-GPU self-play
6. **Time control**: Allocate search time dynamically
7. **Analysis mode**: Output principal variation and evaluation
