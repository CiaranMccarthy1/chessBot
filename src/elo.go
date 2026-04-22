package main

import (
	"math"
)

// EloEstimator estimates the bot's ELO by playing a small test tournament
// against fixed-depth Stockfish instances at known approximate ELO ratings.
// It uses a binary search over depth levels to bracket the true ELO.
//
// Approximate Stockfish ELO by depth:
//   depth 1 → ~600
//   depth 2 → ~900
//   depth 3 → ~1100
//   depth 4 → ~1300
//   depth 5 → ~1500
//   depth 6 → ~1700
//   depth 7 → ~1900
//   depth 8 → ~2100
//   depth 9 → ~2250
//   depth 10 → ~2400
//   depth 11 → ~2500
//   depth 12 → ~2600

var stockfishDepthElo = []struct {
	depth int
	elo   int
}{
	{1, 600},
	{2, 900},
	{3, 1100},
	{4, 1300},
	{5, 1500},
	{6, 1700},
	{7, 1900},
	{8, 2100},
	{9, 2250},
	{10, 2400},
	{11, 2500},
	{12, 2600},
}

// WinRate converts score (0..1) to expected ELO difference.
func EloFromScore(score float64) float64 {
	if score <= 0 {
		return -1000
	}
	if score >= 1 {
		return 1000
	}
	return -400 * math.Log10(1/score-1)
}

// EstimateElo plays gamesPerDepth games against each depth tier,
// returns interpolated ELO estimate.
func EstimateElo(net *AlphaNet, gamesPerDepth int, log func(string)) int {
	bestElo := 400 // floor
	hitCeiling := false
	evalExplore := ExploreConfig{
		SelfExploreStart: 0,
		SelfExploreEnd:   0,
		VsExploreStart:   0,
		VsExploreEnd:     0,
		OpeningPly:       0,
		OpeningTemp:      0,
		CurrentBatch:     1,
		TotalBatches:     1,
	}

	for i, tier := range stockfishDepthElo {
		sf, err := NewStockfish(tier.depth)
		if err != nil {
			log("  elo: stockfish init failed, skipping tier")
			continue
		}

		wins, draws, losses := 0, 0, 0
		for g := 0; g < gamesPerDepth; g++ {
			// Alternate colours
			oppColor := White
			if g%2 == 0 {
				oppColor = Black
			}
			record := PlayVsStockfish(net, sf, oppColor, newRNG(), evalExplore)
			switch record.Result {
			case WhiteWin:
				if oppColor == Black {
					wins++
				} else {
					losses++
				}
			case BlackWin:
				if oppColor == White {
					wins++
				} else {
					losses++
				}
			default:
				draws++
			}
		}
		sf.Close()

		total := float64(wins + draws + losses)
		if total == 0 {
			continue
		}
		score := (float64(wins) + 0.5*float64(draws)) / total
		low, high := scoreCI95(score, total)
		eloDiff := EloFromScore(score)
		estimatedElo := int(float64(tier.elo) + eloDiff)

		log("")
		log("  vs Stockfish depth " + itoa(tier.depth) + " (~" + itoa(tier.elo) + " ELO):")
		log("  W=" + itoa(wins) + " D=" + itoa(draws) + " L=" + itoa(losses) +
			"  score=" + ftoa(score, 2) + "  CI95=[" + ftoa(low, 2) + "," + ftoa(high, 2) + "]  estimated ELO=" + itoa(estimatedElo))

		if score >= 0.5 {
			bestElo = estimatedElo
			if i == len(stockfishDepthElo)-1 {
				hitCeiling = true
			}
		} else {
			// We lost more than half — don't go deeper
			break
		}
	}

	if hitCeiling {
		log("  elo: reached highest configured depth tier; extend stockfishDepthElo for higher estimates")
	}

	return bestElo
}

func scoreCI95(score float64, n float64) (float64, float64) {
	if n <= 1 {
		return score, score
	}
	se := math.Sqrt((score * (1.0 - score)) / n)
	low := score - 1.96*se
	high := score + 1.96*se
	if low < 0 {
		low = 0
	}
	if high > 1 {
		high = 1
	}
	return low, high
}

// simple helpers to avoid fmt import cycles
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	buf := make([]byte, 20)
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

func ftoa(f float64, decimals int) string {
	// manual float formatting to 2dp
	neg := f < 0
	if neg {
		f = -f
	}
	intPart := int(f)
	frac := f - float64(intPart)
	mul := 1
	for i := 0; i < decimals; i++ {
		mul *= 10
	}
	fracPart := int(frac*float64(mul) + 0.5)
	s := itoa(intPart) + "."
	fs := itoa(fracPart)
	for len(fs) < decimals {
		fs = "0" + fs
	}
	s += fs
	if neg {
		s = "-" + s
	}
	return s
}
