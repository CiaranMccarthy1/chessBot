package main

import (
	"math/rand"
)

// GameResult from White's perspective: 1=white win, 0=draw, -1=black win.
type GameResult float32

const (
	WhiteWin GameResult = 1.0
	Draw     GameResult = -0.001
	BlackWin GameResult = -1.0
)

// PositionSample is one board snapshot + which side moved + what the game result was.
// Used as training data for the value head.
type PositionSample struct {
	Board       Board
	Turn        Color       // whose turn it was at this position
	Result      float32     // +1 if that side eventually won, -1 if lost, 0 draw
	Policy      [64]float32 // optional destination-square summary (not primary policy target)
	TargetIndex int         // full move target index (from/to/promo encoded)
}

// GameRecord holds all positions from one game.
type GameRecord struct {
	Samples []PositionSample
	Result  GameResult
}

// ── Self-play ─────────────────────────────────────────────────────────────────

const maxGameMoves = 200
const mctsSimulations = 150
const mctsRootDirichletAlpha = 0.30
const mctsRootDirichletEps = 0.1

type ExploreConfig struct {
	SelfExploreStart float32
	SelfExploreEnd   float32
	VsExploreStart   float32
	VsExploreEnd     float32
	OpeningPly       int
	OpeningTemp      float32
	CurrentBatch     int
	TotalBatches     int
}

var defaultExploreConfig = ExploreConfig{
	SelfExploreStart: 0.10,
	SelfExploreEnd:   0.02,
	VsExploreStart:   0.05,
	VsExploreEnd:     0.01,
	OpeningPly:       12,
	OpeningTemp:      1.15,
	CurrentBatch:     1,
	TotalBatches:     2000,
}

func scheduleValue(start, end float32, batch, total int) float32 {
	if total <= 1 {
		return end
	}
	if batch < 1 {
		batch = 1
	}
	if batch > total {
		batch = total
	}
	t := float32(batch-1) / float32(total-1)
	return start + (end-start)*t
}

func oneHotPolicy(idx int) (out [64]float32) {
	if idx >= 0 && idx < 64 {
		out[idx] = 1
	}
	return out
}

func updateHalfMoveClock(b Board, m Move, halfMoveClock int) int {
	moving := b[m.FromR][m.FromC]
	captured := b[m.ToR][m.ToC]
	if moving.Type == Pawn || captured.Type != None {
		return 0
	}
	return halfMoveClock + 1
}

// PlaySelf plays one game between two instances of the same net.
// Adds Dirichlet noise to prevent determinism (epsilon-greedy style).
func PlaySelf(net *AlphaNet, rng *rand.Rand, cfg ExploreConfig) GameRecord {
	b := StartPos()
	turn := White
	var record GameRecord
	selfExplore := scheduleValue(cfg.SelfExploreStart, cfg.SelfExploreEnd, cfg.CurrentBatch, cfg.TotalBatches)
	halfMoveClock := 0
	repetition := map[string]int{BoardToFEN(b, turn): 1}

	for move := 0; move < maxGameMoves; move++ {
		moves := LegalMoves(b, turn)
		if len(moves) == 0 {
			if IsInCheck(b, turn) {
				if turn == White {
					record.Result = BlackWin
				} else {
					record.Result = WhiteWin
				}
			} else {
				record.Result = Draw
			}
			break
		}

		var m Move
		var policy [64]float32
		if rng.Float32() < selfExplore {
			m = moves[rng.Intn(len(moves))]
			policy = oneHotPolicy(m.ToR*8 + m.ToC)
		} else {
			temp := float32(0)
			if move < cfg.OpeningPly {
				temp = cfg.OpeningTemp
			}
			m, policy = net.SelectMoveMCTSWithPolicy(
				b,
				turn,
				mctsSimulations,
				temp,
				mctsRootDirichletAlpha,
				mctsRootDirichletEps,
				rng,
			)
		}

		targetIdx := movePolicyIndex(m)
		record.Samples = append(record.Samples, PositionSample{
			Board:       b,
			Turn:        turn,
			Policy:      policy,
			TargetIndex: targetIdx,
		})

		halfMoveClock = updateHalfMoveClock(b, m, halfMoveClock)
		b = ApplyMove(b, m)
		turn = turn.Flip()

		if halfMoveClock >= 100 {
			record.Result = Draw
			break
		}

		posKey := BoardToFEN(b, turn)
		repetition[posKey]++
		if repetition[posKey] >= 3 {
			record.Result = Draw
			break
		}

		if move == maxGameMoves-1 {
			record.Result = Draw
		}
	}

	// Back-fill results into samples
	for i := range record.Samples {
		s := &record.Samples[i]
		r := float32(record.Result)
		// Flip sign if this sample was from Black's perspective
		if s.Turn == Black {
			r = -r
		}
		s.Result = r
	}

	return record
}

// ── Stockfish game ────────────────────────────────────────────────────────────

// PlayVsStockfish plays one game where our net plays White, Stockfish plays Black.
// opponentColor controls which colour Stockfish takes.
func PlayVsStockfish(net *AlphaNet, sf *StockfishEngine, opponentColor Color, rng *rand.Rand, cfg ExploreConfig) GameRecord {
	b := StartPos()
	turn := White
	var record GameRecord
	ourColor := opponentColor.Flip()
	vsExplore := scheduleValue(cfg.VsExploreStart, cfg.VsExploreEnd, cfg.CurrentBatch, cfg.TotalBatches)
	halfMoveClock := 0
	repetition := map[string]int{BoardToFEN(b, turn): 1}

	for move := 0; move < maxGameMoves; move++ {
		moves := LegalMoves(b, turn)
		if len(moves) == 0 {
			if IsInCheck(b, turn) {
				if turn == White {
					record.Result = BlackWin
				} else {
					record.Result = WhiteWin
				}
			} else {
				record.Result = Draw
			}
			break
		}

		var m Move
		var policy [64]float32
		if turn == ourColor {
			// Our net picks the move
			if rng.Float32() < vsExplore {
				m = moves[rng.Intn(len(moves))]
				policy = oneHotPolicy(m.ToR*8 + m.ToC)
			} else {
				temp := float32(0)
				if move < cfg.OpeningPly {
					temp = cfg.OpeningTemp
				}
				m, policy = net.SelectMoveMCTSWithPolicy(
					b,
					turn,
					mctsSimulations,
					temp,
					0,
					0,
					rng,
				)
			}
		} else {
			// Stockfish picks the move
			fen := BoardToFEN(b, turn)
			uciMove, err := sf.BestMove(fen)
			if err != nil || uciMove == "" {
				// Stockfish failed — pick random
				m = moves[rng.Intn(len(moves))]
			} else {
				var ok bool
				m, ok = ParseUCI(uciMove, b, turn)
				if !ok {
					m = moves[rng.Intn(len(moves))]
				}
			}
			policy = oneHotPolicy(m.ToR*8 + m.ToC)
		}

		targetIdx := movePolicyIndex(m)
		record.Samples = append(record.Samples, PositionSample{
			Board:       b,
			Turn:        turn,
			Policy:      policy,
			TargetIndex: targetIdx,
		})

		halfMoveClock = updateHalfMoveClock(b, m, halfMoveClock)
		b = ApplyMove(b, m)
		turn = turn.Flip()

		if halfMoveClock >= 100 {
			record.Result = Draw
			break
		}

		posKey := BoardToFEN(b, turn)
		repetition[posKey]++
		if repetition[posKey] >= 3 {
			record.Result = Draw
			break
		}

		if move == maxGameMoves-1 {
			record.Result = Draw
		}
	}

	for i := range record.Samples {
		s := &record.Samples[i]
		r := float32(record.Result)
		if s.Turn == Black {
			r = -r
		}
		s.Result = r
	}

	return record
}
