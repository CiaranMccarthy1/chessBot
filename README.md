sto# Go-Torch Chess Engine

An autonomous chess training system built on top of [go-torch](https://github.com/CiaranMccarthy1/go-torch), a custom deep learning framework written in Go. The bot now uses a two-phase pipeline: supervised pretraining from Stockfish self-play, then reinforcement learning with self-play plus weak/tutor/boss Stockfish tiers until it reaches a target ELO rating.

## Latest training result

- Reached **600 ELO** in about **30 minutes** (single local run).
- Run summary: 50 batches / 1000 games, target ELO 600 reached.

---

## Architecture

```
chessBot/
├── go.mod         Module config
├── go.sum         Dependency checksums
├── README.md      Project documentation
└── src/
    ├── board.go       Board representation, move generation, legal move filtering
    ├── model.go       AlphaNet — value head + policy head via Go-Torch
    ├── selfplay.go    Game generation: self-play and Stockfish opponent games
    ├── stockfish.go   UCI process bridge, FEN serialisation
    ├── elo.go         ELO estimation via tiered Stockfish test matches
    └── main.go        Training loop, worker pool, terminal UI
```

The chess engine is a standalone Go module. It imports go-torch as a library and contains no framework code of its own.

---

## Neural Network

**AlphaNet** is a two-headed network sharing a single hidden layer:

```
Input [1, 128]
    │
    MatMul W1 [128, 64]
    │
    ReLU
    │
    ├── MatMul Wv [64, 1]  →  Value head  (position score, float32)
    └── MatMul Wp [64, 64] →  Policy head (logit per destination square)
```

**Input encoding** — each position is packed into a 128-dimensional tensor:

| Dim | Content |
|-----|---------|
| 0–63 | Material value + piece-square table bonus per square. White = positive, Black = negative. |
| 64–127 | Mobility heat-map — legal move count per target square, weighted ±0.05 by colour. |

The full input is flipped to the side-to-move's perspective before the forward pass, so the network always reasons from the current player's point of view.

**Go-Torch operations used:**

| Op | Where |
|----|-------|
| `gt.NewTensor` | Board → tensor encoding |
| `gt.MatMul` | Three times per forward pass (W1, Wv, Wp) |
| `gt.ReLU` | After hidden layer |
| `tensor.Backward()` | Autograd through the value head on each training step |
| `AtomicAddFloat32` | Gradient accumulation inside MatMul backward (concurrent workers) |

**Loss:** MSE on the value head — `(predicted_score − game_result)²`. Gradient is injected manually into `vOut.Grad` then propagated via `vOut.Backward()`.

**Optimiser:** SGD with a fixed learning rate (default `0.001`).

---

## Training Pipeline

### RL opponent mix

| Tier | Share | Opponent | Purpose |
|------|-------|----------|---------|
| Self-play | 62.5% | Another instance of AlphaNet | Balanced improvement; training partner scales with the bot |
| Tutor | 20.8% | Stockfish, starting at depth 1 | Tactical baseline; depth increases on strong results |
| Boss | 4.2% | Stockfish depth 18 (~2500 ELO) | Stress-tests long-range threats; bakes deep calculation into static weights |
| Weak | 12.5% | Stockfish at half tutor depth (min 1) | Early win signal and confidence building |

### Loop

```
for each batch:
     1. If in pretraining mode:
         - Generate Stockfish vs Stockfish games at the current pretraining depth
         - Train with supervised policy targets (value weight low, policy weight high)
         - Increase pretraining depth as policy loss improves
         - Switch to RL after the configured number of batches or sustained low policy loss
     2. If in RL mode:
         - Spin up N worker goroutines
         - Each worker generates games according to the S/T/B/W split
         - Self-play: AlphaNet vs AlphaNet (exploration in openings)
         - Tutor/Boss/Weak: AlphaNet vs Stockfish via UCI, alternating colours
     3. Collect all (board, side_to_move, game_result) samples
     4. Shuffle samples across all games
     5. Run TrainMiniBatch updates on replay samples
     6. Every N batches: probe ELO against tiered Stockfish opponents
     7. Stop when estimated ELO >= target
```

### Concurrency

Game generation is parallelised across CPU cores with `sync.WaitGroup`. Gradient accumulation inside Go-Torch's `MatMul.Backward` uses `sync/atomic` CAS loops (`AtomicAddFloat32`) so workers can accumulate gradients without locking.

### ELO estimation

Every 5 batches the bot plays a short match against Stockfish at increasing depths:

| Stockfish depth | Approximate ELO |
|-----------------|-----------------|
| 1 | 600 |
| 2 | 900 |
| 3 | 1100 |
| 4 | 1300 |
| 5 | 1500 |
| 6 | 1700 |
| 7 | 1900 |
| 8 | 2100 |

Score against each tier is converted to an ELO difference via the standard logistic formula. The bot stops climbing tiers once it scores below 50% and reports its estimated ELO.

---

## Usage

### Prerequisites

- Go 1.22+
- Stockfish installed and on `PATH` (`apt install stockfish` on Debian/Ubuntu)
- [go-torch](https://github.com/CiaranMccarthy1/go-torch) cloned locally

### Module setup

In `go.mod`, point the replace directive at your local go-torch clone:

```
require github.com/CiaranMccarthy1/go-torch v0.0.0

replace github.com/CiaranMccarthy1/go-torch => ../go-torch
```

Adjust the path to match where go-torch lives relative to this module. If go-torch is ever published to the Go module proxy, remove the `replace` line and run `go get github.com/CiaranMccarthy1/go-torch@latest`.

### Run

```bash
# Train to default target (1200 ELO)
go run ./src

# Train to default target from scratch (ignore checkpoints)
go run ./src --new

# Pretrain then switch to RL (default 300 pretrain batches)
go run ./src --pretrain

# Pretrain for a custom number of batches
go run ./src --pretrain 500

# Pretrain for a custom number of batches and set target ELO
go run ./src --pretrain 300 1500

# Train to a custom target
go run ./src 1500

# Train to a custom target from scratch
go run ./src --new 1500

# Build a binary first
go build -o chess-engine ./src
./chess-engine 1400
```

On startup, the bot automatically tries to load `checkpoints/best.gob`.
Use `--new` to ignore saved weights and start from random initialization.

Training runs indefinitely, printing a live terminal dashboard. Press `Ctrl+C` to stop cleanly.

### Configuration

Edit the `defaultConfig()` function in `src/main.go`:

| Field | Default | Description |
|-------|---------|-------------|
| `TargetElo` | 1200 | ELO at which training stops |
| `LearningRate` | 0.0004 | Adam step size |
| `BatchSize` | 24 | Games generated per training batch |
| `EloCheckEvery` | 5 | Batches between ELO probes |
| `EloGamesPerTier` | 16 | Games played per Stockfish tier during probe |
| `Workers` | 8 | Concurrent game-generation goroutines |
| `SelfPlayRatio` | 0.625 | Fraction of batch that is self-play |
| `TutorRatio` | 0.2083 | Fraction of batch vs tutor Stockfish |
| `BossRatio` | 0.0417 | Fraction of batch vs boss Stockfish |
| `WeakRatio` | 0.125 | Fraction of batch vs weak Stockfish |
| `ValueLossWeight` | 1.0 | Value loss weight during RL |
| `PolicyLossWeight` | 0.3 | Policy loss weight during RL |
| `PretrainDepth` | 4 | Stockfish depth for pretraining (both sides) |
| `PretrainBatches` | 300 | Pretraining batches before switching to RL |
| `PretrainPolicyWeight` | 1.0 | Policy loss weight during pretraining |
| `PretrainValueWeight` | 0.2 | Value loss weight during pretraining |
| `PolicySwitchLoss` | 1.5 | Policy loss threshold to switch to RL |
| `PolicySwitchWindow` | 10 | Consecutive batches below threshold before switching |

Boss/weak ratios are explicit in config. Tutor depth starts at 1 and promotes upward; weak depth tracks half of tutor depth (min 1).

---

## Dependency

| Package | Role |
|---------|------|
| `github.com/CiaranMccarthy1/go-torch` | Tensor operations, autograd, parallel MatMul |

No other external dependencies. The Stockfish bridge uses Go's `os/exec` over stdin/stdout (UCI protocol).

---

## Limitations & next steps

- **Weight serialisation** — checkpoints are saved to disk via encoding/gob as checkpoints/latest.gob (every N batches) and checkpoints/best.gob (on ELO improvement or clean shutdown). Pass --new to ignore saved weights and start from random initialisation.
- **No castling or en-passant** — the move generator omits both for simplicity. Adding them improves opening and tactical play significantly.
- **FEN is simplified** — castling rights and en-passant square are omitted from generated FENs. This is fine for training position evaluation but means Stockfish may play slightly differently than it would in a full game.
- **Fixed learning rate** — a cosine or step decay schedule would improve convergence at higher ELOs.
- **Single GPU / CPU only** — Go-Torch currently uses CPU parallelism via goroutines. A CUDA backend would dramatically speed up the forward and backward passes.