package main

import (
	"math"
	"math/rand"

	gt "github.com/CiaranMccarthy1/go-torch"
)

// ── Piece-square tables (White perspective, row 0 = rank 8) ──────────────────

var pawnPST = [8][8]float32{
	{0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
	{0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50},
	{0.10, 0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.10},
	{0.05, 0.05, 0.10, 0.25, 0.25, 0.10, 0.05, 0.05},
	{0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00},
	{0.05, -0.05, -0.10, 0.00, 0.00, -0.10, -0.05, 0.05},
	{0.05, 0.10, 0.10, -0.20, -0.20, 0.10, 0.10, 0.05},
	{0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
}

var knightPST = [8][8]float32{
	{-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50},
	{-0.40, -0.20, 0.00, 0.00, 0.00, 0.00, -0.20, -0.40},
	{-0.30, 0.00, 0.10, 0.15, 0.15, 0.10, 0.00, -0.30},
	{-0.30, 0.05, 0.15, 0.20, 0.20, 0.15, 0.05, -0.30},
	{-0.30, 0.00, 0.15, 0.20, 0.20, 0.15, 0.00, -0.30},
	{-0.30, 0.05, 0.10, 0.15, 0.15, 0.10, 0.05, -0.30},
	{-0.40, -0.20, 0.00, 0.05, 0.05, 0.00, -0.20, -0.40},
	{-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50},
}

var bishopPST = [8][8]float32{
	{-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20},
	{-0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.10},
	{-0.10, 0.00, 0.05, 0.10, 0.10, 0.05, 0.00, -0.10},
	{-0.10, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, -0.10},
	{-0.10, 0.00, 0.10, 0.10, 0.10, 0.10, 0.00, -0.10},
	{-0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.10},
	{-0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, -0.10},
	{-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20},
}

var rookPST = [8][8]float32{
	{0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
	{0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05},
	{-0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.05},
	{-0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.05},
	{-0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.05},
	{-0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.05},
	{-0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.05},
	{0.00, 0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00},
}

var queenPST = [8][8]float32{
	{-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20},
	{-0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.10},
	{-0.10, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, -0.10},
	{-0.05, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, -0.05},
	{0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, -0.05},
	{-0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00, -0.10},
	{-0.10, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, -0.10},
	{-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20},
}

var pieceVals = map[PieceType]float32{
	Pawn: 1, Knight: 3, Bishop: 3.2, Rook: 5, Queen: 9,
}

// ── Feature encoding ──────────────────────────────────────────────────────────

// BoardToTensor encodes the full board as a flat [128] tensor.
// First 64 dims: material + PST value per square (White positive, Black negative).
// Next  64 dims: mobility map — count of legal moves targeting each square.
// Shape: [1, 128] for MatMul compatibility with W1 [128, H].
func BoardToTensor(b Board, turn Color) *gt.Tensor {
	data := make([]float32, 128)

	// Channel 0: piece values + PST
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			p := b[r][c]
			if p.Type == None {
				continue
			}
			base := pieceVals[p.Type]
			tableRow := r
			if p.Color == White {
				tableRow = 7 - r
			}
			var bonus float32
			switch p.Type {
			case Pawn:
				bonus = pawnPST[tableRow][c]
			case Knight:
				bonus = knightPST[tableRow][c]
			case Bishop:
				bonus = bishopPST[tableRow][c]
			case Rook:
				bonus = rookPST[tableRow][c]
			case Queen:
				bonus = queenPST[tableRow][c]
			}
			sign := float32(1)
			if p.Color == Black {
				sign = -1
			}
			data[r*8+c] = sign * (base + bonus)
		}
	}

	// Channel 1: mobility heat-map
	for _, color := range []Color{White, Black} {
		sign := float32(1)
		if color == Black {
			sign = -1
		}
		for _, m := range LegalMoves(b, color) {
			data[64+m.ToR*8+m.ToC] += sign * 0.05
		}
	}

	// From the perspective of the side to move
	if turn == Black {
		for i := range data {
			data[i] = -data[i]
		}
	}

	return gt.NewTensor(data, []int{1, 128}, false)
}

// ── Network ───────────────────────────────────────────────────────────────────

const (
	InputDim             = 128
	HiddenDim            = 64
	HiddenDim2           = 64
	PromoClasses         = 5 // none, knight, bishop, rook, queen
	NetCheckpointVersion = 1
)

// AlphaNet is a single network with one value head and factorized move-policy heads.
//   - Value head: predicts game outcome in [-1, +1]
//   - Policy heads: logits for from-square, to-square, and promotion class
//
// Architecture: [1,128] → MatMul W1[128,64] → ReLU → MatMul W2[64,64] → ReLU
//
//	→ (Value: MatMul Wv[64,1])
//	→ (PolicyFrom: MatMul Wpf[64,64])
//	→ (PolicyTo: MatMul Wpt[64,64])
//	→ (PolicyPromo: MatMul Wpp[64,5])
type AlphaNet struct {
	W1  *gt.Tensor // shared trunk-1 [128, 64]
	W2  *gt.Tensor // shared trunk-2 [64, 64]
	Wv  *gt.Tensor // value head     [64, 1]
	Wpf *gt.Tensor // policy from-head [64, 64]
	Wpt *gt.Tensor // policy to-head   [64, 64]
	Wpp *gt.Tensor // policy promo-head [64, 5]

	mW1  []float32
	vW1  []float32
	mW2  []float32
	vW2  []float32
	mWv  []float32
	vWv  []float32
	mWpf []float32
	vWpf []float32
	mWpt []float32
	vWpt []float32
	mWpp []float32
	vWpp []float32
	step int
}

func NewAlphaNet(rng *rand.Rand) *AlphaNet {
	fanIn := func(in, out int) float32 {
		// He init
		_ = out
		return float32(math.Sqrt(2.0 / float64(in)))
	}
	randTensor := func(rows, cols int) *gt.Tensor {
		scale := fanIn(rows, cols)
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = float32(rng.NormFloat64()) * scale
		}
		return gt.NewTensor(data, []int{rows, cols}, true)
	}
	n := &AlphaNet{
		W1:  randTensor(InputDim, HiddenDim),
		W2:  randTensor(HiddenDim, HiddenDim2),
		Wv:  randTensor(HiddenDim2, 1),
		Wpf: randTensor(HiddenDim2, 64),
		Wpt: randTensor(HiddenDim2, 64),
		Wpp: randTensor(HiddenDim2, PromoClasses),
	}
	n.mW1 = make([]float32, len(n.W1.Data))
	n.vW1 = make([]float32, len(n.W1.Data))
	n.mW2 = make([]float32, len(n.W2.Data))
	n.vW2 = make([]float32, len(n.W2.Data))
	n.mWv = make([]float32, len(n.Wv.Data))
	n.vWv = make([]float32, len(n.Wv.Data))
	n.mWpf = make([]float32, len(n.Wpf.Data))
	n.vWpf = make([]float32, len(n.Wpf.Data))
	n.mWpt = make([]float32, len(n.Wpt.Data))
	n.vWpt = make([]float32, len(n.Wpt.Data))
	n.mWpp = make([]float32, len(n.Wpp.Data))
	n.vWpp = make([]float32, len(n.Wpp.Data))
	return n
}

// Forward returns value score, factorized policy logits, and output tensors.
func (n *AlphaNet) Forward(b Board, turn Color) (float32, []float32, []float32, []float32, *gt.Tensor, *gt.Tensor, *gt.Tensor, *gt.Tensor) {
	x := BoardToTensor(b, turn)
	h1 := gt.ReLU(gt.MatMul(x, n.W1))  // [1, 64]
	h2 := gt.ReLU(gt.MatMul(h1, n.W2)) // [1, 64]
	vOut := gt.MatMul(h2, n.Wv)        // [1, 1]
	pfOut := gt.MatMul(h2, n.Wpf)      // [1, 64]
	ptOut := gt.MatMul(h2, n.Wpt)      // [1, 64]
	ppOut := gt.MatMul(h2, n.Wpp)      // [1, 5]
	return vOut.Data[0], pfOut.Data, ptOut.Data, ppOut.Data, vOut, pfOut, ptOut, ppOut
}

func (n *AlphaNet) ZeroGrad() {
	n.W1.ZeroGrad()
	n.W2.ZeroGrad()
	n.Wv.ZeroGrad()
	n.Wpf.ZeroGrad()
	n.Wpt.ZeroGrad()
	n.Wpp.ZeroGrad()
}

func softmax(logits []float32) []float32 {
	out := make([]float32, len(logits))
	if len(logits) == 0 {
		return out
	}
	maxLogit := logits[0]
	for _, x := range logits {
		if x > maxLogit {
			maxLogit = x
		}
	}
	var sum float64
	for i, x := range logits {
		e := math.Exp(float64(x - maxLogit))
		out[i] = float32(e)
		sum += e
	}
	if sum <= 0 {
		u := float32(1.0 / float64(len(out)))
		for i := range out {
			out[i] = u
		}
		return out
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

func (n *AlphaNet) clipGradients(maxNorm float32) {
	if maxNorm <= 0 {
		return
	}
	var sq float64
	for _, g := range n.W1.Grad {
		sq += float64(g * g)
	}
	for _, g := range n.W2.Grad {
		sq += float64(g * g)
	}
	for _, g := range n.Wv.Grad {
		sq += float64(g * g)
	}
	for _, g := range n.Wpf.Grad {
		sq += float64(g * g)
	}
	for _, g := range n.Wpt.Grad {
		sq += float64(g * g)
	}
	for _, g := range n.Wpp.Grad {
		sq += float64(g * g)
	}
	norm := float32(math.Sqrt(sq))
	if norm <= maxNorm || norm == 0 {
		return
	}
	scale := maxNorm / norm
	for i := range n.W1.Grad {
		n.W1.Grad[i] *= scale
	}
	for i := range n.W2.Grad {
		n.W2.Grad[i] *= scale
	}
	for i := range n.Wv.Grad {
		n.Wv.Grad[i] *= scale
	}
	for i := range n.Wpf.Grad {
		n.Wpf.Grad[i] *= scale
	}
	for i := range n.Wpt.Grad {
		n.Wpt.Grad[i] *= scale
	}
	for i := range n.Wpp.Grad {
		n.Wpp.Grad[i] *= scale
	}
}

func (n *AlphaNet) addWeightDecay(decay float32) {
	if decay <= 0 {
		return
	}
	for i := range n.W1.Grad {
		n.W1.Grad[i] += decay * n.W1.Data[i]
	}
	for i := range n.W2.Grad {
		n.W2.Grad[i] += decay * n.W2.Data[i]
	}
	for i := range n.Wv.Grad {
		n.Wv.Grad[i] += decay * n.Wv.Data[i]
	}
	for i := range n.Wpf.Grad {
		n.Wpf.Grad[i] += decay * n.Wpf.Data[i]
	}
	for i := range n.Wpt.Grad {
		n.Wpt.Grad[i] += decay * n.Wpt.Data[i]
	}
	for i := range n.Wpp.Grad {
		n.Wpp.Grad[i] += decay * n.Wpp.Data[i]
	}
}

func adamStep(param, grad, m, v []float32, lr float32, step int) {
	const (
		beta1 = float32(0.9)
		beta2 = float32(0.999)
		eps   = float32(1e-8)
	)
	b1t := float32(1.0 - math.Pow(float64(beta1), float64(step)))
	b2t := float32(1.0 - math.Pow(float64(beta2), float64(step)))
	for i := range param {
		g := grad[i]
		m[i] = beta1*m[i] + (1-beta1)*g
		v[i] = beta2*v[i] + (1-beta2)*g*g
		mHat := m[i] / b1t
		vHat := v[i] / b2t
		param[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
	}
}

func (n *AlphaNet) StepAdam(lr float32) {
	n.step++
	adamStep(n.W1.Data, n.W1.Grad, n.mW1, n.vW1, lr, n.step)
	adamStep(n.W2.Data, n.W2.Grad, n.mW2, n.vW2, lr, n.step)
	adamStep(n.Wv.Data, n.Wv.Grad, n.mWv, n.vWv, lr, n.step)
	adamStep(n.Wpf.Data, n.Wpf.Grad, n.mWpf, n.vWpf, lr, n.step)
	adamStep(n.Wpt.Data, n.Wpt.Grad, n.mWpt, n.vWpt, lr, n.step)
	adamStep(n.Wpp.Data, n.Wpp.Grad, n.mWpp, n.vWpp, lr, n.step)
}

func promoClass(pt PieceType) int {
	switch pt {
	case Knight:
		return 1
	case Bishop:
		return 2
	case Rook:
		return 3
	case Queen:
		return 4
	default:
		return 0
	}
}

func movePolicyIndex(m Move) int {
	from := m.FromR*8 + m.FromC
	to := m.ToR*8 + m.ToC
	return ((from*64)+to)*PromoClasses + promoClass(m.Promo)
}

func moveLogitFromHeads(m Move, fromLogits []float32, toLogits []float32, promoLogits []float32) float32 {
	from := m.FromR*8 + m.FromC
	to := m.ToR*8 + m.ToC
	pc := promoClass(m.Promo)
	return fromLogits[from] + toLogits[to] + promoLogits[pc]
}

// TrainStep runs one forward→backward→SGD cycle.
// target is the game result from the current side's perspective: +1 win, 0 draw, -1 loss.
func (n *AlphaNet) TrainStep(b Board, turn Color, target float32, lr float32) float32 {
	vl, pl, _ := n.TrainStepJoint(b, turn, target, -1, lr, 1.0, 0.0, 0.0, 0.0)
	return vl + pl
}

// TrainStepJoint runs a single step on value and policy heads simultaneously.
func (n *AlphaNet) TrainStepJoint(
	b Board,
	turn Color,
	valueTarget float32,
	moveTargetIdx int,
	lr float32,
	valueWeight float32,
	policyWeight float32,
	weightDecay float32,
	clipNorm float32,
) (float32, float32, float32) {
	n.ZeroGrad()
	score, fromLogits, toLogits, promoLogits, vOut, pfOut, ptOut, ppOut := n.Forward(b, turn)

	valueLoss := (score - valueTarget) * (score - valueTarget)
	vOut.Grad = []float32{valueWeight * 2 * (score - valueTarget)}
	vOut.Backward()

	policyLoss := float32(0)
	if policyWeight > 0 && moveTargetIdx >= 0 {
		moves := LegalMoves(b, turn)
		if len(moves) > 0 {
			moveLogits := make([]float32, len(moves))
			maxLogit := float32(-1e9)
			for i, m := range moves {
				ml := moveLogitFromHeads(m, fromLogits, toLogits, promoLogits)
				moveLogits[i] = ml
				if ml > maxLogit {
					maxLogit = ml
				}
			}

			exps := make([]float32, len(moves))
			var sum float64
			for i := range moves {
				e := math.Exp(float64(moveLogits[i] - maxLogit))
				exps[i] = float32(e)
				sum += e
			}
			if sum <= 0 {
				sum = 1
			}
			invSum := float32(1.0 / sum)

			fromGrad := make([]float32, 64)
			toGrad := make([]float32, 64)
			promoGrad := make([]float32, PromoClasses)

			for i, m := range moves {
				prob := exps[i] * invSum
				policyTarget := float32(0)
				if movePolicyIndex(m) == moveTargetIdx {
					policyTarget = 1
					policyLoss += -float32(math.Log(float64(maxf(prob, 1e-8))))
				}
				g := policyWeight * (prob - policyTarget)
				fromGrad[m.FromR*8+m.FromC] += g
				toGrad[m.ToR*8+m.ToC] += g
				promoGrad[promoClass(m.Promo)] += g
			}

			pfOut.Grad = fromGrad
			pfOut.Backward()
			ptOut.Grad = toGrad
			ptOut.Backward()
			ppOut.Grad = promoGrad
			ppOut.Backward()
		}
	}

	n.addWeightDecay(weightDecay)
	n.clipGradients(clipNorm)
	n.StepAdam(lr)

	return valueLoss, policyLoss, valueWeight*valueLoss + policyWeight*policyLoss
}

// TrainMiniBatch applies one optimizer step over a batch of samples.
func (n *AlphaNet) TrainMiniBatch(
	samples []PositionSample,
	lr float32,
	valueWeight float32,
	policyWeight float32,
	weightDecay float32,
	clipNorm float32,
) (float32, float32, float32) {
	if len(samples) == 0 {
		return 0, 0, 0
	}
	n.ZeroGrad()
	invN := float32(1.0 / float64(len(samples)))
	var valueLossSum float32
	var policyLossSum float32

	for _, s := range samples {
		score, fromLogits, toLogits, promoLogits, vOut, pfOut, ptOut, ppOut := n.Forward(s.Board, s.Turn)
		valueLoss := (score - s.Result) * (score - s.Result)
		valueLossSum += valueLoss
		vOut.Grad = []float32{valueWeight * 2 * (score - s.Result) * invN}
		vOut.Backward()

		moves := LegalMoves(s.Board, s.Turn)
		if len(moves) > 0 && policyWeight > 0 {
			moveLogits := make([]float32, len(moves))
			maxLogit := float32(-1e9)
			for i, m := range moves {
				ml := moveLogitFromHeads(m, fromLogits, toLogits, promoLogits)
				moveLogits[i] = ml
				if ml > maxLogit {
					maxLogit = ml
				}
			}

			exps := make([]float32, len(moves))
			var sum float64
			for i := range moves {
				e := math.Exp(float64(moveLogits[i] - maxLogit))
				exps[i] = float32(e)
				sum += e
			}
			if sum <= 0 {
				sum = 1
			}
			invSum := float32(1.0 / sum)

			fromGrad := make([]float32, 64)
			toGrad := make([]float32, 64)
			promoGrad := make([]float32, PromoClasses)

			for i, m := range moves {
				prob := exps[i] * invSum
				policyTarget := float32(0)
				if movePolicyIndex(m) == s.TargetIndex {
					policyTarget = 1
					policyLossSum += -float32(math.Log(float64(maxf(prob, 1e-8))))
				}
				g := policyWeight * (prob - policyTarget) * invN
				fromGrad[m.FromR*8+m.FromC] += g
				toGrad[m.ToR*8+m.ToC] += g
				promoGrad[promoClass(m.Promo)] += g
			}

			pfOut.Grad = fromGrad
			pfOut.Backward()
			ptOut.Grad = toGrad
			ptOut.Backward()
			ppOut.Grad = promoGrad
			ppOut.Backward()
		}
	}

	n.addWeightDecay(weightDecay)
	n.clipGradients(clipNorm)
	n.StepAdam(lr)

	valueLossAvg := valueLossSum * invN
	policyLossAvg := policyLossSum * invN
	return valueLossAvg, policyLossAvg, valueWeight*valueLossAvg + policyWeight*policyLossAvg
}

type mctsNode struct {
	board Board
	turn  Color

	parent *mctsNode
	move   Move

	moves      []Move
	priors     []float32
	children   []*mctsNode
	unexpanded int

	visits   int
	valueSum float32

	terminal bool
	termVal  float32
}

func terminalValue(b Board, turn Color) (bool, float32) {
	moves := LegalMoves(b, turn)
	if len(moves) > 0 {
		return false, 0
	}
	if IsInCheck(b, turn) {
		return true, -1
	}
	return true, 0
}

func (n *AlphaNet) newMCTSNode(parent *mctsNode, b Board, turn Color, mv Move) *mctsNode {
	term, tval := terminalValue(b, turn)
	node := &mctsNode{
		board:    b,
		turn:     turn,
		parent:   parent,
		move:     mv,
		terminal: term,
		termVal:  tval,
	}
	if term {
		return node
	}

	node.moves = LegalMoves(b, turn)
	node.children = make([]*mctsNode, len(node.moves))
	node.priors = make([]float32, len(node.moves))
	node.unexpanded = len(node.moves)

	_, fromLogits, toLogits, promoLogits, _, _, _, _ := n.Forward(b, turn)
	maxLogit := float32(-1e9)
	for _, m := range node.moves {
		s := moveLogitFromHeads(m, fromLogits, toLogits, promoLogits)
		if s > maxLogit {
			maxLogit = s
		}
	}
	var sum float64
	for i, m := range node.moves {
		e := math.Exp(float64(moveLogitFromHeads(m, fromLogits, toLogits, promoLogits) - maxLogit))
		node.priors[i] = float32(e)
		sum += e
	}
	if sum <= 0 {
		u := float32(1.0 / float64(len(node.priors)))
		for i := range node.priors {
			node.priors[i] = u
		}
	} else {
		inv := float32(1.0 / sum)
		for i := range node.priors {
			node.priors[i] *= inv
		}
	}

	return node
}

func pickRandomIndex(rng *rand.Rand, n int) int {
	if n <= 1 {
		return 0
	}
	if rng != nil {
		return rng.Intn(n)
	}
	return rand.Intn(n)
}

func clampUnit(x float32) float32 {
	if x > 1 {
		return 1
	}
	if x < -1 {
		return -1
	}
	return x
}

func randFloat64(rng *rand.Rand) float64 {
	if rng != nil {
		return rng.Float64()
	}
	return rand.Float64()
}

func randNormFloat64(rng *rand.Rand) float64 {
	if rng != nil {
		return rng.NormFloat64()
	}
	return rand.NormFloat64()
}

func sampleGamma(alpha float64, rng *rand.Rand) float64 {
	if alpha <= 0 {
		return 0
	}
	if alpha < 1 {
		u := randFloat64(rng)
		if u <= 0 {
			u = 1e-12
		}
		return sampleGamma(alpha+1, rng) * math.Pow(u, 1.0/alpha)
	}

	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := randNormFloat64(rng)
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := randFloat64(rng)
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

func sampleDirichlet(n int, alpha float64, rng *rand.Rand) []float32 {
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	if alpha <= 0 {
		u := float32(1.0 / float64(n))
		for i := range out {
			out[i] = u
		}
		return out
	}
	var sum float64
	for i := range out {
		g := sampleGamma(alpha, rng)
		out[i] = float32(g)
		sum += g
	}
	if sum <= 0 {
		u := float32(1.0 / float64(n))
		for i := range out {
			out[i] = u
		}
		return out
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

func sampleFromWeights(weights []float64, rng *rand.Rand) int {
	if len(weights) == 0 {
		return 0
	}
	var sum float64
	for _, w := range weights {
		if w > 0 {
			sum += w
		}
	}
	if sum <= 0 {
		return pickRandomIndex(rng, len(weights))
	}
	pick := randFloat64(rng) * sum
	acc := 0.0
	for i, w := range weights {
		if w <= 0 {
			continue
		}
		acc += w
		if pick <= acc {
			return i
		}
	}
	return len(weights) - 1
}

// SelectMoveMCTS picks a move with PUCT-guided Monte Carlo Tree Search.
func (n *AlphaNet) SelectMoveMCTS(b Board, turn Color, simulations int, rng *rand.Rand) Move {
	m, _ := n.SelectMoveMCTSWithPolicy(b, turn, simulations, 0, 0, 0, rng)
	return m
}

// SelectMoveMCTSWithPolicy returns the selected move and a destination-square policy
// derived from root visit counts.
func (n *AlphaNet) SelectMoveMCTSWithPolicy(
	b Board,
	turn Color,
	simulations int,
	temperature float32,
	dirichletAlpha float32,
	dirichletEpsilon float32,
	rng *rand.Rand,
) (Move, [64]float32) {
	var outPolicy [64]float32
	rootMoves := LegalMoves(b, turn)
	if len(rootMoves) == 0 {
		return Move{}, outPolicy
	}
	if len(rootMoves) == 1 {
		idx := rootMoves[0].ToR*8 + rootMoves[0].ToC
		outPolicy[idx] = 1
		return rootMoves[0], outPolicy
	}
	if simulations < 1 {
		simulations = 1
	}

	root := n.newMCTSNode(nil, b, turn, Move{})
	if root.terminal {
		idx := rootMoves[0].ToR*8 + rootMoves[0].ToC
		outPolicy[idx] = 1
		return rootMoves[0], outPolicy
	}

	if dirichletEpsilon > 0 && dirichletAlpha > 0 && len(root.priors) > 0 {
		if dirichletEpsilon > 1 {
			dirichletEpsilon = 1
		}
		noise := sampleDirichlet(len(root.priors), float64(dirichletAlpha), rng)
		mix := 1 - dirichletEpsilon
		for i := range root.priors {
			root.priors[i] = mix*root.priors[i] + dirichletEpsilon*noise[i]
		}
	}

	const cPuct = float32(1.4)

	for s := 0; s < simulations; s++ {
		node := root

		for !node.terminal && node.unexpanded == 0 {
			bestIdx := 0
			bestScore := float32(-1e9)
			sqrtParent := float32(math.Sqrt(float64(maxf(float32(node.visits), 1))))
			for i := range node.moves {
				child := node.children[i]
				if child == nil {
					continue
				}
				qChild := float32(0)
				if child.visits > 0 {
					qChild = child.valueSum / float32(child.visits)
				}
				q := -qChild
				u := cPuct * node.priors[i] * sqrtParent / (1 + float32(child.visits))
				score := q + u
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
			node = node.children[bestIdx]
		}

		if !node.terminal && node.unexpanded > 0 {
			var choices []int
			for i := range node.moves {
				if node.children[i] == nil {
					choices = append(choices, i)
				}
			}
			pick := choices[pickRandomIndex(rng, len(choices))]
			nb := ApplyMove(node.board, node.moves[pick])
			child := n.newMCTSNode(node, nb, node.turn.Flip(), node.moves[pick])
			node.children[pick] = child
			node.unexpanded--
			node = child
		}

		leafValue := node.termVal
		if !node.terminal {
			v, _, _, _, _, _, _, _ := n.Forward(node.board, node.turn)
			leafValue = clampUnit(v)
		}

		v := leafValue
		for cur := node; cur != nil; cur = cur.parent {
			cur.visits++
			cur.valueSum += v
			v = -v
		}
	}

	visitCounts := make([]float64, len(root.moves))
	var totalVisits float64
	for i := range root.moves {
		child := root.children[i]
		v := 0.0
		if child != nil {
			v = float64(child.visits)
		}
		visitCounts[i] = v
		totalVisits += v
	}

	if totalVisits <= 0 {
		u := float32(1.0 / float64(len(root.moves)))
		for _, m := range root.moves {
			outPolicy[m.ToR*8+m.ToC] += u
		}
		pick := pickRandomIndex(rng, len(root.moves))
		return root.moves[pick], outPolicy
	}

	weights := make([]float64, len(visitCounts))
	if temperature <= 0 {
		best := 0
		for i := 1; i < len(visitCounts); i++ {
			if visitCounts[i] > visitCounts[best] {
				best = i
			}
		}
		weights[best] = 1
	} else {
		invT := 1.0 / float64(temperature)
		for i, v := range visitCounts {
			if v > 0 {
				weights[i] = math.Pow(v, invT)
			}
		}
	}

	var wsum float64
	for _, w := range weights {
		wsum += w
	}
	if wsum <= 0 {
		u := float64(1.0 / float64(len(weights)))
		for i := range weights {
			weights[i] = u
		}
		wsum = 1
	}

	for i, m := range root.moves {
		p := float32(weights[i] / wsum)
		outPolicy[m.ToR*8+m.ToC] += p
	}

	picked := sampleFromWeights(weights, rng)
	return root.moves[picked], outPolicy
}

// SelectMove picks the best legal move using the policy head logits.
// Falls back to the value head via minimax if policy is degenerate.
func (n *AlphaNet) SelectMove(b Board, turn Color, searchDepth int) Move {
	if searchDepth > 1 {
		return n.SelectMoveMCTS(b, turn, searchDepth, nil)
	}

	moves := LegalMoves(b, turn)
	if len(moves) == 0 {
		return Move{}
	}
	_, fromLogits, toLogits, promoLogits, _, _, _, _ := n.Forward(b, turn)

	// Score each legal move using factorized from/to/promo logits.
	best := moves[0]
	bestScore := float32(-1e9)
	for _, m := range moves {
		score := moveLogitFromHeads(m, fromLogits, toLogits, promoLogits)
		// Blend with shallow value search
		nb := ApplyMove(b, m)
		vScore, _, _, _, _, _, _, _ := n.Forward(nb, turn.Flip())
		blended := score + (-vScore)*0.5 // vScore from opponent's view
		if blended > bestScore {
			bestScore = blended
			best = m
		}
	}
	return best
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

type NetCheckpoint struct {
	Version int
	W1      []float32
	W2      []float32
	Wv      []float32
	Wpf     []float32
	Wpt     []float32
	Wpp     []float32
	MW1     []float32
	VW1     []float32
	MW2     []float32
	VW2     []float32
	MWv     []float32
	VWv     []float32
	MWpf    []float32
	VWpf    []float32
	MWpt    []float32
	VWpt    []float32
	MWpp    []float32
	VWpp    []float32
	Step    int
}

func (n *AlphaNet) Snapshot() NetCheckpoint {
	cp := NetCheckpoint{Version: NetCheckpointVersion, Step: n.step}
	cp.W1 = append(cp.W1, n.W1.Data...)
	cp.W2 = append(cp.W2, n.W2.Data...)
	cp.Wv = append(cp.Wv, n.Wv.Data...)
	cp.Wpf = append(cp.Wpf, n.Wpf.Data...)
	cp.Wpt = append(cp.Wpt, n.Wpt.Data...)
	cp.Wpp = append(cp.Wpp, n.Wpp.Data...)
	cp.MW1 = append(cp.MW1, n.mW1...)
	cp.VW1 = append(cp.VW1, n.vW1...)
	cp.MW2 = append(cp.MW2, n.mW2...)
	cp.VW2 = append(cp.VW2, n.vW2...)
	cp.MWv = append(cp.MWv, n.mWv...)
	cp.VWv = append(cp.VWv, n.vWv...)
	cp.MWpf = append(cp.MWpf, n.mWpf...)
	cp.VWpf = append(cp.VWpf, n.vWpf...)
	cp.MWpt = append(cp.MWpt, n.mWpt...)
	cp.VWpt = append(cp.VWpt, n.vWpt...)
	cp.MWpp = append(cp.MWpp, n.mWpp...)
	cp.VWpp = append(cp.VWpp, n.vWpp...)
	return cp
}

func (n *AlphaNet) LoadSnapshot(cp NetCheckpoint) bool {
	if cp.Version != NetCheckpointVersion {
		return false
	}
	if len(cp.W1) != len(n.W1.Data) ||
		len(cp.W2) != len(n.W2.Data) ||
		len(cp.Wv) != len(n.Wv.Data) ||
		len(cp.Wpf) != len(n.Wpf.Data) ||
		len(cp.Wpt) != len(n.Wpt.Data) ||
		len(cp.Wpp) != len(n.Wpp.Data) ||
		len(cp.MW1) != len(n.mW1) ||
		len(cp.VW1) != len(n.vW1) ||
		len(cp.MW2) != len(n.mW2) ||
		len(cp.VW2) != len(n.vW2) ||
		len(cp.MWv) != len(n.mWv) ||
		len(cp.VWv) != len(n.vWv) ||
		len(cp.MWpf) != len(n.mWpf) ||
		len(cp.VWpf) != len(n.vWpf) ||
		len(cp.MWpt) != len(n.mWpt) ||
		len(cp.VWpt) != len(n.vWpt) ||
		len(cp.MWpp) != len(n.mWpp) ||
		len(cp.VWpp) != len(n.vWpp) {
		return false
	}

	copy(n.W1.Data, cp.W1)
	copy(n.W2.Data, cp.W2)
	copy(n.Wv.Data, cp.Wv)
	copy(n.Wpf.Data, cp.Wpf)
	copy(n.Wpt.Data, cp.Wpt)
	copy(n.Wpp.Data, cp.Wpp)

	copy(n.mW1, cp.MW1)
	copy(n.vW1, cp.VW1)
	copy(n.mW2, cp.MW2)
	copy(n.vW2, cp.VW2)
	copy(n.mWv, cp.MWv)
	copy(n.vWv, cp.VWv)
	copy(n.mWpf, cp.MWpf)
	copy(n.vWpf, cp.VWpf)
	copy(n.mWpt, cp.MWpt)
	copy(n.vWpt, cp.VWpt)
	copy(n.mWpp, cp.MWpp)
	copy(n.vWpp, cp.VWpp)
	n.step = cp.Step
	return true
}
