package main

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// ── RNG ───────────────────────────────────────────────────────────────────────

func newRNG() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// ── Training config ───────────────────────────────────────────────────────────

type TrainConfig struct {
	TargetElo       int
	LearningRate    float32
	BatchSize       int     // games per training batch
	EloCheckEvery   int     // batches between ELO probes
	EloGamesPerTier int     // games per Stockfish tier during ELO check
	Workers         int     // concurrent game-generation goroutines
	SelfPlayRatio   float64 // base ratios (used when adaptive schedule disabled)
	TutorRatio      float64
	BossRatio       float64
	WeakRatio       float64
	AdaptiveMix     bool

	MiniBatchSize    int
	UpdatesPerBatch  int
	ReplayCapacity   int
	MinReplayToTrain int

	ValueLossWeight  float32
	PolicyLossWeight float32
	WeightDecay      float32
	GradClipNorm     float32

	ScheduleBatches int

	CheckpointPath  string
	CheckpointEvery int
	// Ratios should sum to 1.0
}

func defaultConfig() TrainConfig {
	return TrainConfig{
		TargetElo:       1200,
		LearningRate:    0.0004,
		BatchSize:       24,
		EloCheckEvery:   5,
		EloGamesPerTier: 16,
		Workers:         8,
		SelfPlayRatio:   0.40,
		TutorRatio:      0.40,
		BossRatio:       0.05,
		WeakRatio:       0.15,
		AdaptiveMix:     false,

		MiniBatchSize:    128,
		UpdatesPerBatch:  8,
		ReplayCapacity:   50000,
		MinReplayToTrain: 2000,

		ValueLossWeight:  1.0,
		PolicyLossWeight: 1.0,
		WeightDecay:      1e-5,
		GradClipNorm:     2.0,

		ScheduleBatches: 1500,

		CheckpointPath:  "checkpoints/latest.gob",
		CheckpointEvery: 5,
	}
}

// ── Display ───────────────────────────────────────────────────────────────────

const (
	clrScreen = "\033[H\033[2J"
	bold      = "\033[1m"
	dim       = "\033[2m"
	reset     = "\033[0m"
	cyan      = "\033[36m"
	green     = "\033[32m"
	yellow    = "\033[33m"
	red       = "\033[31m"
	gray      = "\033[90m"
)

type TrainingState struct {
	mu            sync.Mutex
	Batch         int
	TotalGames    int
	CurrentElo    int
	TargetElo     int
	AvgLoss       float32
	AvgValueLoss  float32
	AvgPolicyLoss float32
	ReplaySize    int
	SelfWins      int
	TutorWins     int
	BossWins      int
	WeakWins      int
	SelfLosses    int
	TutorLosses   int
	BossLosses    int
	WeakLosses    int
	Log           []string
	StartTime     time.Time
	LastEloCheck  time.Time
}

func (s *TrainingState) AddLog(msg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Log = append(s.Log, msg)
	if len(s.Log) > 12 {
		s.Log = s.Log[len(s.Log)-12:]
	}
}

func progressBar(current, target, width int) string {
	if target == 0 {
		return ""
	}
	pct := float64(current) / float64(target)
	if pct > 1 {
		pct = 1
	}
	filled := int(pct * float64(width))
	bar := ""
	for i := 0; i < width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	return bar
}

func render(s *TrainingState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	fmt.Print(clrScreen)
	elapsed := time.Since(s.StartTime).Round(time.Second)
	nextCheck := s.LastEloCheck.Add(time.Duration(s.Batch%5) * time.Minute)
	_ = nextCheck

	fmt.Printf("%s%sGo-Torch Chess — Autonomous Training%s\n", bold, cyan, reset)
	fmt.Printf("%s%s%s\n\n", gray, "────────────────────────────────────────────────────────", reset)

	// ELO progress
	eloBar := progressBar(s.CurrentElo, s.TargetElo, 36)
	pct := 0.0
	if s.TargetElo > 0 {
		pct = float64(s.CurrentElo) / float64(s.TargetElo) * 100
	}
	fmt.Printf("  %sELO%s  %s%s%s  %s%d%s → %s%d%s  (%.0f%%)\n\n",
		gray, reset,
		green, eloBar, reset,
		bold, s.CurrentElo, reset,
		gray, s.TargetElo, reset,
		pct)

	// Stats row
	fmt.Printf("  %sBatch%s  %-6d  %sGames%s  %-6d  %sReplay%s  %-6d  %sUptime%s  %s\n",
		gray, reset, s.Batch,
		gray, reset, s.TotalGames,
		gray, reset, s.ReplaySize,
		gray, reset, elapsed)

	fmt.Printf("  %sLoss%s total=%.4f value=%.4f policy=%.4f\n\n",
		gray, reset, s.AvgLoss, s.AvgValueLoss, s.AvgPolicyLoss)

	// Opponent breakdown
	fmt.Printf("  %s%-20s  W     L%s\n", gray, "Opponent", reset)
	fmt.Printf("  %-20s  %s%-5d%s %s%-5d%s\n", "Self-play",
		green, s.SelfWins, reset, red, s.SelfLosses, reset)
	fmt.Printf("  %-20s  %s%-5d%s %s%-5d%s\n", "Tutor SF",
		green, s.TutorWins, reset, red, s.TutorLosses, reset)
	fmt.Printf("  %-20s  %s%-5d%s %s%-5d%s\n", "Boss SF",
		green, s.BossWins, reset, red, s.BossLosses, reset)
	fmt.Printf("  %-20s  %s%-5d%s %s%-5d%s\n\n", "Weak SF",
		green, s.WeakWins, reset, red, s.WeakLosses, reset)

	// Log
	fmt.Printf("  %sTraining log:%s\n", gray, reset)
	for _, l := range s.Log {
		fmt.Printf("  %s%s%s\n", dim, l, reset)
	}

	fmt.Printf("\n  %s[Ctrl+C to stop and save weights]%s\n", gray, reset)
}

type ReplayBuffer struct {
	data     []PositionSample
	capacity int
}

func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{capacity: max1(capacity, 1)}
}

func (r *ReplayBuffer) Add(samples []PositionSample) {
	if len(samples) == 0 {
		return
	}
	r.data = append(r.data, samples...)
	if len(r.data) > r.capacity {
		over := len(r.data) - r.capacity
		r.data = append([]PositionSample(nil), r.data[over:]...)
	}
}

func (r *ReplayBuffer) Len() int {
	return len(r.data)
}

func (r *ReplayBuffer) Sample(n int, rng *rand.Rand) []PositionSample {
	if n <= 0 || len(r.data) == 0 {
		return nil
	}
	if n > len(r.data) {
		n = len(r.data)
	}
	out := make([]PositionSample, n)
	for i := 0; i < n; i++ {
		out[i] = r.data[rng.Intn(len(r.data))]
	}
	return out
}

func adaptiveRatios(batch int, total int, baseSelf float64, baseTutor float64, baseBoss float64, baseWeak float64, adaptive bool) (float64, float64, float64, float64) {
	if !adaptive {
		t := baseSelf + baseTutor + baseBoss + baseWeak
		if t <= 0 {
			return 0.40, 0.50, 0.05, 0.05
		}
		return baseSelf / t, baseTutor / t, baseBoss / t, baseWeak / t
	}
	if total <= 0 {
		total = 1
	}
	weak := baseWeak
	if weak < 0 {
		weak = 0
	}
	if weak > 0.25 {
		weak = 0.25
	}
	scale := 1.0 - weak
	if scale <= 0 {
		scale = 1
	}
	p := float64(batch) / float64(total)
	if p < 0.30 {
		return 0.35 * scale, 0.60 * scale, 0.05 * scale, weak
	}
	if p < 0.70 {
		return 0.55 * scale, 0.35 * scale, 0.10 * scale, weak
	}
	return 0.45 * scale, 0.25 * scale, 0.30 * scale, weak
}

type TrainingCheckpoint struct {
	Batch      int
	BestElo    int
	TutorDepth int
	WeakDepth  int
	Net        NetCheckpoint
	SavedAt    time.Time
}

func saveCheckpoint(path string, cp TrainingCheckpoint) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(cp)
}

func loadCheckpoint(path string) (TrainingCheckpoint, error) {
	var cp TrainingCheckpoint
	f, err := os.Open(path)
	if err != nil {
		return cp, err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	err = dec.Decode(&cp)
	return cp, err
}

func scoreFromOurPerspective(rec GameResult, ourColor Color) float64 {
	if rec == Draw {
		return 0.5
	}
	if (rec == WhiteWin && ourColor == White) || (rec == BlackWin && ourColor == Black) {
		return 1.0
	}
	return 0.0
}

func newStockfishPool(count int, depth int) ([]*StockfishEngine, chan *StockfishEngine, error) {
	if count < 1 {
		count = 1
	}
	pool := make([]*StockfishEngine, 0, count)
	ch := make(chan *StockfishEngine, count)
	for i := 0; i < count; i++ {
		sf, err := NewStockfish(depth)
		if err != nil {
			for _, e := range pool {
				e.Close()
			}
			close(ch)
			return nil, nil, err
		}
		pool = append(pool, sf)
		ch <- sf
	}
	return pool, ch, nil
}

func closeStockfishPool(pool []*StockfishEngine) {
	for _, sf := range pool {
		sf.Close()
	}
}

func setStockfishPoolDepth(pool []*StockfishEngine, depth int) {
	for _, sf := range pool {
		sf.SetDepth(depth)
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	cfg := defaultConfig()
	startFresh := false

	// Parse optional CLI args:
	//   --new         start from scratch (ignore saved best checkpoint)
	//   <number>      target ELO
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--new":
			startFresh = true
		case "-h", "--help":
			fmt.Println("Usage: go run ./src [--new] [target_elo]")
			return
		default:
			if n, err := strconv.Atoi(arg); err == nil {
				cfg.TargetElo = n
			}
		}
	}

	rng := newRNG()
	net := NewAlphaNet(rng)
	replay := NewReplayBuffer(cfg.ReplayCapacity)
	bestPath := filepath.Join(filepath.Dir(cfg.CheckpointPath), "best.gob")
	currentTutorDepth := int(StockfishStandard)
	maxTutorDepth := int(StockfishBoss)
	weakDepth := max1(0, currentTutorDepth-5)

	bestElo := 400
	startBatch := 0
	loadedFrom := ""
	checkpointIncompatible := false
	if !startFresh {
		if cp, err := loadCheckpoint(bestPath); err == nil {
			if net.LoadSnapshot(cp.Net) {
				bestElo = cp.BestElo
				startBatch = cp.Batch
				if cp.TutorDepth > 0 {
					currentTutorDepth = cp.TutorDepth
				}
				if cp.WeakDepth >= 0 {
					weakDepth = cp.WeakDepth
				}
				weakDepth = max1(0, currentTutorDepth-5)
				loadedFrom = bestPath
			} else {
				checkpointIncompatible = true
				net = NewAlphaNet(rng)
				bestElo = 400
				startBatch = 0
				currentTutorDepth = int(StockfishStandard)
				weakDepth = max1(0, currentTutorDepth-5)
			}
		}
	}

	state := &TrainingState{
		Batch:        startBatch,
		TargetElo:    cfg.TargetElo,
		CurrentElo:   bestElo,
		StartTime:    time.Now(),
		LastEloCheck: time.Now(),
	}

	state.AddLog(fmt.Sprintf("Target ELO: %d", cfg.TargetElo))
	if startFresh {
		state.AddLog("Starting fresh (--new): ignoring saved checkpoints")
	} else if checkpointIncompatible {
		state.AddLog("Checkpoint incompatible — starting fresh")
	} else if loadedFrom != "" {
		state.AddLog("Loaded checkpoint: " + loadedFrom)
	} else {
		state.AddLog("No best checkpoint found; starting from random weights")
	}
	state.AddLog(fmt.Sprintf("Workers: %d  |  Batch: %d games  |  LR: %.5f", cfg.Workers, cfg.BatchSize, cfg.LearningRate))
	state.AddLog(fmt.Sprintf("Replay: cap=%d min=%d mb=%d updates=%d", cfg.ReplayCapacity, cfg.MinReplayToTrain, cfg.MiniBatchSize, cfg.UpdatesPerBatch))
	state.AddLog("Initialising Stockfish engines...")
	render(state)

	// Spin up Stockfish tutor + boss + weak pools
	poolSize := max1(cfg.Workers, 1)
	tutorPool, tutorCh, err := newStockfishPool(poolSize, currentTutorDepth)
	if err != nil {
		fmt.Fprintf(os.Stderr, "stockfish tutor init failed: %v\n", err)
		os.Exit(1)
	}
	defer closeStockfishPool(tutorPool)

	bossPool, bossCh, err := newStockfishPool(poolSize, int(StockfishBoss))
	if err != nil {
		closeStockfishPool(tutorPool)
		fmt.Fprintf(os.Stderr, "stockfish boss init failed: %v\n", err)
		os.Exit(1)
	}
	defer closeStockfishPool(bossPool)

	weakPool, weakCh, err := newStockfishPool(poolSize, weakDepth)
	if err != nil {
		closeStockfishPool(tutorPool)
		closeStockfishPool(bossPool)
		fmt.Fprintf(os.Stderr, "stockfish weak init failed: %v\n", err)
		os.Exit(1)
	}
	defer closeStockfishPool(weakPool)

	state.AddLog(fmt.Sprintf("Stockfish engine pools ready (%d each, T/W depth: %d/%d).", poolSize, currentTutorDepth, weakDepth))

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		state.AddLog("Interrupt received — stopping after current batch.")
		shutdownCP := TrainingCheckpoint{
			Batch:      state.Batch,
			BestElo:    bestElo,
			TutorDepth: currentTutorDepth,
			WeakDepth:  weakDepth,
			Net:        net.Snapshot(),
			SavedAt:    time.Now(),
		}
		if err := saveCheckpoint(cfg.CheckpointPath, shutdownCP); err != nil {
			state.AddLog(fmt.Sprintf("Checkpoint save failed: %v", err))
		} else {
			state.AddLog("Checkpoint saved: " + cfg.CheckpointPath)
		}

		bestOnDisk := -1
		bestValid := false
		if cp, err := loadCheckpoint(bestPath); err == nil && cp.Net.Version == NetCheckpointVersion {
			bestOnDisk = cp.BestElo
			bestValid = true
		}
		promotedOnShutdown := false
		if !bestValid || shutdownCP.BestElo > bestOnDisk {
			if err := saveCheckpoint(bestPath, shutdownCP); err != nil {
				state.AddLog(fmt.Sprintf("Best checkpoint save failed: %v", err))
			} else {
				promotedOnShutdown = true
				state.AddLog("Promoted best checkpoint on shutdown: " + bestPath)
			}
		}
		render(state)
		fmt.Printf("\n%s%sTraining stopped. Latest checkpoint saved to %s.%s\n", bold, yellow, cfg.CheckpointPath, reset)
		if promotedOnShutdown {
			fmt.Printf("%sBest checkpoint updated: %s%s\n", gray, bestPath, reset)
		}
		os.Exit(0)
	}()

	// ── Training loop ─────────────────────────────────────────────────────────
	batchRNG := newRNG()
	for {
		state.mu.Lock()
		batch := state.Batch + 1
		state.Batch = batch
		state.mu.Unlock()

		selfRatio, tutorRatio, bossRatio, weakRatio := adaptiveRatios(
			batch,
			cfg.ScheduleBatches,
			cfg.SelfPlayRatio,
			cfg.TutorRatio,
			cfg.BossRatio,
			cfg.WeakRatio,
			cfg.AdaptiveMix,
		)
		nSelf := int(float64(cfg.BatchSize) * selfRatio)
		nTutor := int(float64(cfg.BatchSize) * tutorRatio)
		nBoss := int(float64(cfg.BatchSize) * bossRatio)
		nWeak := int(float64(cfg.BatchSize) * weakRatio)
		assigned := nSelf + nTutor + nBoss + nWeak
		if assigned < cfg.BatchSize {
			nTutor += cfg.BatchSize - assigned
		} else if assigned > cfg.BatchSize {
			over := assigned - cfg.BatchSize
			if nTutor >= over {
				nTutor -= over
			} else {
				nTutor = 0
			}
		}

		exploreCfg := defaultExploreConfig
		exploreCfg.CurrentBatch = batch
		exploreCfg.TotalBatches = cfg.ScheduleBatches

		state.AddLog(fmt.Sprintf("Batch %d — generating %d games (S/T/B/W: %d/%d/%d/%d)", batch, cfg.BatchSize, nSelf, nTutor, nBoss, nWeak))
		render(state)

		var (
			recordsMu  sync.Mutex
			allRecords []GameRecord
			wg         sync.WaitGroup
			tutorGames int
			tutorScore float64
		)

		// Worker pool — game generation
		type job struct {
			kind string // "self" | "tutor" | "boss" | "weak"
		}
		jobs := make(chan job, cfg.BatchSize)
		for i := 0; i < nSelf; i++ {
			jobs <- job{"self"}
		}
		for i := 0; i < nTutor; i++ {
			jobs <- job{"tutor"}
		}
		for i := 0; i < nBoss; i++ {
			jobs <- job{"boss"}
		}
		for i := 0; i < nWeak; i++ {
			jobs <- job{"weak"}
		}
		close(jobs)

		for w := 0; w < cfg.Workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				workerRNG := newRNG()
				for j := range jobs {
					var rec GameRecord
					switch j.kind {
					case "self":
						rec = PlaySelf(net, workerRNG, exploreCfg)
						state.mu.Lock()
						if rec.Result == WhiteWin {
							state.SelfWins++
						} else if rec.Result == BlackWin {
							state.SelfLosses++
						}
						state.mu.Unlock()
					case "tutor":
						oppColor := White
						if workerRNG.Intn(2) == 0 {
							oppColor = Black
						}
						sf := <-tutorCh
						rec = PlayVsStockfish(net, sf, oppColor, workerRNG, exploreCfg)
						tutorCh <- sf
						state.mu.Lock()
						ourColor := oppColor.Flip()
						if (rec.Result == WhiteWin && ourColor == White) ||
							(rec.Result == BlackWin && ourColor == Black) {
							state.TutorWins++
						} else if rec.Result != Draw {
							state.TutorLosses++
						}
						state.mu.Unlock()
						recordsMu.Lock()
						tutorGames++
						tutorScore += scoreFromOurPerspective(rec.Result, ourColor)
						recordsMu.Unlock()
					case "boss":
						oppColor := White
						if workerRNG.Intn(2) == 0 {
							oppColor = Black
						}
						sf := <-bossCh
						rec = PlayVsStockfish(net, sf, oppColor, workerRNG, exploreCfg)
						bossCh <- sf
						state.mu.Lock()
						ourColor := oppColor.Flip()
						if (rec.Result == WhiteWin && ourColor == White) ||
							(rec.Result == BlackWin && ourColor == Black) {
							state.BossWins++
						} else if rec.Result != Draw {
							state.BossLosses++
						}
						state.mu.Unlock()
					case "weak":
						oppColor := White
						if workerRNG.Intn(2) == 0 {
							oppColor = Black
						}
						sf := <-weakCh
						rec = PlayVsStockfish(net, sf, oppColor, workerRNG, exploreCfg)
						weakCh <- sf
						state.mu.Lock()
						ourColor := oppColor.Flip()
						if (rec.Result == WhiteWin && ourColor == White) ||
							(rec.Result == BlackWin && ourColor == Black) {
							state.WeakWins++
						} else if rec.Result != Draw {
							state.WeakLosses++
						}
						state.mu.Unlock()
					}
					recordsMu.Lock()
					allRecords = append(allRecords, rec)
					recordsMu.Unlock()
				}
			}()
		}
		wg.Wait()

		if tutorGames > 0 {
			score := tutorScore / float64(tutorGames)
			if score >= 0.80 && currentTutorDepth < maxTutorDepth {
				currentTutorDepth++
				setStockfishPoolDepth(tutorPool, currentTutorDepth)
				weakDepth = max1(0, currentTutorDepth-5)
				setStockfishPoolDepth(weakPool, weakDepth)
				state.AddLog(fmt.Sprintf("Tutor promotion: score %.0f%% (%d games) -> tutor depth %d, weak depth %d", score*100, tutorGames, currentTutorDepth, weakDepth))
			}
		}

		// Count games
		state.mu.Lock()
		state.TotalGames += len(allRecords)
		state.mu.Unlock()

		// ── Training step: batch all position samples ─────────────────────────
		state.AddLog(fmt.Sprintf("Batch %d — training on positions...", batch))
		render(state)

		var nSamples int

		var allSamples []PositionSample
		for _, rec := range allRecords {
			allSamples = append(allSamples, rec.Samples...)
		}
		replay.Add(allSamples)

		state.mu.Lock()
		state.ReplaySize = replay.Len()
		state.mu.Unlock()

		if replay.Len() < cfg.MinReplayToTrain {
			state.AddLog(fmt.Sprintf("Replay warmup: %d/%d samples", replay.Len(), cfg.MinReplayToTrain))
			render(state)
		} else {
			var totalLoss float32
			var valueLoss float32
			var policyLoss float32
			for u := 0; u < cfg.UpdatesPerBatch; u++ {
				mb := replay.Sample(cfg.MiniBatchSize, batchRNG)
				if len(mb) == 0 {
					continue
				}
				vl, pl, tl := net.TrainMiniBatch(
					mb,
					cfg.LearningRate,
					cfg.ValueLossWeight,
					cfg.PolicyLossWeight,
					cfg.WeightDecay,
					cfg.GradClipNorm,
				)
				valueLoss += vl
				policyLoss += pl
				totalLoss += tl
				nSamples += len(mb)
			}

			den := float32(max1(cfg.UpdatesPerBatch, 1))
			state.mu.Lock()
			state.AvgLoss = totalLoss / den
			state.AvgValueLoss = valueLoss / den
			state.AvgPolicyLoss = policyLoss / den
			state.mu.Unlock()

			state.AddLog(fmt.Sprintf("Batch %d done — %d samples, loss=%.4f (v=%.4f p=%.4f)",
				batch, nSamples, totalLoss/den, valueLoss/den, policyLoss/den))

			if cfg.CheckpointEvery > 0 && batch%cfg.CheckpointEvery == 0 {
				if err := saveCheckpoint(cfg.CheckpointPath, TrainingCheckpoint{
					Batch:      batch,
					BestElo:    bestElo,
					TutorDepth: currentTutorDepth,
					WeakDepth:  weakDepth,
					Net:        net.Snapshot(),
					SavedAt:    time.Now(),
				}); err == nil {
					state.AddLog("Checkpoint saved: " + cfg.CheckpointPath)
				}
			}
		}

		// ── ELO probe ─────────────────────────────────────────────────────────
		if cfg.EloCheckEvery > 0 && batch%cfg.EloCheckEvery == 0 {
			state.AddLog(fmt.Sprintf("Batch %d — probing ELO...", batch))
			render(state)

			elo := EstimateElo(net, cfg.EloGamesPerTier, func(msg string) {
				state.AddLog(msg)
				render(state)
			})

			state.mu.Lock()
			state.CurrentElo = elo
			state.LastEloCheck = time.Now()
			state.mu.Unlock()

			state.AddLog(fmt.Sprintf("ELO estimate: %d  (target: %d)", elo, cfg.TargetElo))
			if elo > bestElo {
				bestElo = elo
				_ = saveCheckpoint(bestPath, TrainingCheckpoint{
					Batch:      batch,
					BestElo:    bestElo,
					TutorDepth: currentTutorDepth,
					WeakDepth:  weakDepth,
					Net:        net.Snapshot(),
					SavedAt:    time.Now(),
				})
				state.AddLog("Promoted new best checkpoint: " + bestPath)
			}
			render(state)

			if elo >= cfg.TargetElo {
				_ = saveCheckpoint(cfg.CheckpointPath, TrainingCheckpoint{
					Batch:      batch,
					BestElo:    bestElo,
					TutorDepth: currentTutorDepth,
					WeakDepth:  weakDepth,
					Net:        net.Snapshot(),
					SavedAt:    time.Now(),
				})
				render(state)
				fmt.Printf("\n%s%s🎉  Target ELO %d reached!  Current ELO: %d%s\n\n",
					bold, green, cfg.TargetElo, elo, reset)
				fmt.Printf("%sTraining complete after %d batches / %d games.%s\n\n",
					cyan, batch, state.TotalGames, reset)
				fmt.Printf("%sCheckpoint saved to %s%s\n\n", gray, cfg.CheckpointPath, reset)
				return
			}
		}

		render(state)
	}
}

func max1(a, b int) int {
	if a > b {
		return a
	}
	return b
}
