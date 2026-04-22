package main

import (
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// StockfishLevel controls search depth / strength.
type StockfishLevel int

const (
	StockfishWeak     StockfishLevel = 0  // "weak" finisher practice tier
	StockfishStandard StockfishLevel = 6  // "tutor"
	StockfishBoss     StockfishLevel = 18 // "boss" — ~5x deeper
)

// StockfishEngine manages a single Stockfish process via UCI.
type StockfishEngine struct {
	mu        sync.Mutex
	cmd       *exec.Cmd
	stdin     *bufio.Writer
	stdout    *bufio.Scanner
	stdoutCh  chan string
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once
	depth     int
}

func NewStockfish(depth int) (*StockfishEngine, error) {
	cmd := exec.Command("stockfish")
	inPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	sf := &StockfishEngine{
		cmd:      cmd,
		stdin:    bufio.NewWriter(inPipe),
		stdout:   bufio.NewScanner(outPipe),
		stdoutCh: make(chan string, 256),
		ctx:      ctx,
		cancel:   cancel,
		depth:    depth,
	}
	go sf.readStdoutLoop()

	if err := sf.send("uci"); err != nil {
		sf.Close()
		return nil, err
	}
	if !sf.waitFor("uciok", 3*time.Second) {
		sf.Close()
		return nil, fmt.Errorf("stockfish: missing uciok")
	}
	if err := sf.send("isready"); err != nil {
		sf.Close()
		return nil, err
	}
	if !sf.waitFor("readyok", 3*time.Second) {
		sf.Close()
		return nil, fmt.Errorf("stockfish: missing readyok")
	}
	if err := sf.send("ucinewgame"); err != nil {
		sf.Close()
		return nil, err
	}
	return sf, nil
}

func (sf *StockfishEngine) send(cmd string) error {
	if _, err := fmt.Fprintln(sf.stdin, cmd); err != nil {
		return err
	}
	return sf.stdin.Flush()
}

func (sf *StockfishEngine) readStdoutLoop() {
	defer close(sf.stdoutCh)
	for sf.stdout.Scan() {
		line := sf.stdout.Text()
		select {
		case sf.stdoutCh <- line:
		case <-sf.ctx.Done():
			return
		}
	}
}

func (sf *StockfishEngine) waitFor(token string, timeout time.Duration) bool {
	t := time.NewTimer(timeout)
	defer t.Stop()
	for {
		select {
		case <-sf.ctx.Done():
			return false
		case <-t.C:
			return false
		case line, ok := <-sf.stdoutCh:
			if !ok {
				return false
			}
			if strings.Contains(line, token) {
				return true
			}
		}
	}
}

// BestMove returns Stockfish's best move in UCI format for the given FEN position.
func (sf *StockfishEngine) BestMove(fen string) (string, error) {
	sf.mu.Lock()
	defer sf.mu.Unlock()

	if err := sf.send(fmt.Sprintf("position fen %s", fen)); err != nil {
		return "", err
	}
	if err := sf.send(fmt.Sprintf("go depth %d", sf.depth)); err != nil {
		return "", err
	}

	t := time.NewTimer(30 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-sf.ctx.Done():
			return "", fmt.Errorf("stockfish: process ended")
		case <-t.C:
			return "", fmt.Errorf("stockfish: timeout")
		case line, ok := <-sf.stdoutCh:
			if !ok {
				return "", fmt.Errorf("stockfish: process ended")
			}
			if strings.HasPrefix(line, "bestmove") {
				parts := strings.Fields(line)
				if len(parts) >= 2 && parts[1] != "(none)" {
					return parts[1], nil
				}
				return "", fmt.Errorf("stockfish: no move")
			}
		}
	}
}

func (sf *StockfishEngine) Close() {
	sf.closeOnce.Do(func() {
		sf.mu.Lock()
		defer sf.mu.Unlock()
		_ = sf.send("quit")
		_ = sf.cmd.Wait()
		sf.cancel()
	})
}

func (sf *StockfishEngine) SetDepth(depth int) {
	if depth < 0 {
		depth = 0
	}
	sf.mu.Lock()
	sf.depth = depth
	sf.mu.Unlock()
}

func (sf *StockfishEngine) Depth() int {
	sf.mu.Lock()
	defer sf.mu.Unlock()
	return sf.depth
}

// ── FEN generation ─────────────────────────────────────────────────────────

var pieceChar = map[Piece]byte{
	{Pawn, White}: 'P', {Knight, White}: 'N', {Bishop, White}: 'B',
	{Rook, White}: 'R', {Queen, White}: 'Q', {King, White}: 'K',
	{Pawn, Black}: 'p', {Knight, Black}: 'n', {Bishop, Black}: 'b',
	{Rook, Black}: 'r', {Queen, Black}: 'q', {King, Black}: 'k',
}

// BoardToFEN converts a board + turn to a minimal FEN string.
// Castling and en-passant are omitted for simplicity (training only).
func BoardToFEN(b Board, turn Color) string {
	var sb strings.Builder
	for r := 0; r < 8; r++ {
		empty := 0
		for c := 0; c < 8; c++ {
			p := b[r][c]
			if p.Type == None {
				empty++
			} else {
				if empty > 0 {
					sb.WriteByte(byte('0' + empty))
					empty = 0
				}
				sb.WriteByte(pieceChar[p])
			}
		}
		if empty > 0 {
			sb.WriteByte(byte('0' + empty))
		}
		if r < 7 {
			sb.WriteByte('/')
		}
	}
	if turn == White {
		sb.WriteString(" w - - 0 1")
	} else {
		sb.WriteString(" b - - 0 1")
	}
	return sb.String()
}
