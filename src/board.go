package main

import "fmt"

// ── Types ─────────────────────────────────────────────────────────────────────

type Color int8

const (
	White Color = iota
	Black
	NoColor
)

func (c Color) Flip() Color {
	if c == White {
		return Black
	}
	return White
}

type PieceType int8

const (
	None PieceType = iota
	Pawn
	Knight
	Bishop
	Rook
	Queen
	King
)

type Piece struct {
	Type  PieceType
	Color Color
}

var NoPiece = Piece{None, NoColor}

// ── Board ─────────────────────────────────────────────────────────────────────

type Board [8][8]Piece

func StartPos() Board {
	var b Board
	back := []PieceType{Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook}
	for c := 0; c < 8; c++ {
		b[0][c] = Piece{back[c], Black}
		b[1][c] = Piece{Pawn, Black}
		b[6][c] = Piece{Pawn, White}
		b[7][c] = Piece{back[c], White}
	}
	return b
}

// ── Move ──────────────────────────────────────────────────────────────────────

type Move struct {
	FromR, FromC int
	ToR, ToC     int
	Promo        PieceType
}

func (m Move) UCI() string {
	s := fmt.Sprintf("%c%d%c%d", 'a'+m.FromC, 8-m.FromR, 'a'+m.ToC, 8-m.ToR)
	if m.Promo != None {
		s += string("  nbrq "[m.Promo])
	}
	return s
}

func ParseUCI(s string, b Board, color Color) (Move, bool) {
	if len(s) < 4 {
		return Move{}, false
	}
	fc := int(s[0] - 'a')
	fr := 8 - int(s[1]-'0')
	tc := int(s[2] - 'a')
	tr := 8 - int(s[3]-'0')
	if fc < 0 || fc > 7 || tc < 0 || tc > 7 || fr < 0 || fr > 7 || tr < 0 || tr > 7 {
		return Move{}, false
	}
	promo := None
	if len(s) >= 5 {
		switch s[4] {
		case 'q':
			promo = Queen
		case 'r':
			promo = Rook
		case 'b':
			promo = Bishop
		case 'n':
			promo = Knight
		}
	}
	for _, legal := range LegalMoves(b, color) {
		if legal.FromR == fr && legal.FromC == fc && legal.ToR == tr && legal.ToC == tc {
			if promo == None || promo == legal.Promo {
				return legal, true
			}
		}
	}
	return Move{}, false
}

// ── Move application ──────────────────────────────────────────────────────────

func ApplyMove(b Board, m Move) Board {
	nb := b
	p := nb[m.FromR][m.FromC]
	nb[m.ToR][m.ToC] = p
	nb[m.FromR][m.FromC] = NoPiece
	if m.Promo != None {
		nb[m.ToR][m.ToC] = Piece{m.Promo, p.Color}
	}
	return nb
}

// ── Move generation ───────────────────────────────────────────────────────────

func rawMoves(b Board, r, c int) []Move {
	p := b[r][c]
	color := p.Color
	enemy := color.Flip()
	var moves []Move

	slide := func(dr, dc int) {
		nr, nc := r+dr, c+dc
		for nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
			t := b[nr][nc]
			if t.Type == None {
				moves = append(moves, Move{r, c, nr, nc, None})
			} else if t.Color == enemy {
				moves = append(moves, Move{r, c, nr, nc, None})
				break
			} else {
				break
			}
			nr += dr
			nc += dc
		}
	}
	jump := func(dr, dc int) {
		nr, nc := r+dr, c+dc
		if nr < 0 || nr > 7 || nc < 0 || nc > 7 {
			return
		}
		t := b[nr][nc]
		if t.Type == None || t.Color == enemy {
			moves = append(moves, Move{r, c, nr, nc, None})
		}
	}

	switch p.Type {
	case Pawn:
		dir, start := -1, 6
		if color == Black {
			dir, start = 1, 1
		}
		promRank := 0
		if color == Black {
			promRank = 7
		}
		if r+dir >= 0 && r+dir < 8 && b[r+dir][c].Type == None {
			if r+dir == promRank {
				for _, pt := range []PieceType{Queen, Rook, Bishop, Knight} {
					moves = append(moves, Move{r, c, r + dir, c, pt})
				}
			} else {
				moves = append(moves, Move{r, c, r + dir, c, None})
				if r == start && b[r+2*dir][c].Type == None {
					moves = append(moves, Move{r, c, r + 2*dir, c, None})
				}
			}
		}
		for _, dc := range []int{-1, 1} {
			nc := c + dc
			nr := r + dir
			if nc >= 0 && nc < 8 && nr >= 0 && nr < 8 {
				t := b[nr][nc]
				if t.Type != None && t.Color == enemy {
					if nr == promRank {
						for _, pt := range []PieceType{Queen, Rook, Bishop, Knight} {
							moves = append(moves, Move{r, c, nr, nc, pt})
						}
					} else {
						moves = append(moves, Move{r, c, nr, nc, None})
					}
				}
			}
		}
	case Knight:
		for _, d := range [][2]int{{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}} {
			jump(d[0], d[1])
		}
	case Bishop:
		for _, d := range [][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
			slide(d[0], d[1])
		}
	case Rook:
		for _, d := range [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}} {
			slide(d[0], d[1])
		}
	case Queen:
		for _, d := range [][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 1}, {0, -1}, {1, 0}, {-1, 0}} {
			slide(d[0], d[1])
		}
	case King:
		for dr := -1; dr <= 1; dr++ {
			for dc := -1; dc <= 1; dc++ {
				if dr != 0 || dc != 0 {
					jump(dr, dc)
				}
			}
		}
	}
	return moves
}

func IsInCheck(b Board, color Color) bool {
	kr, kc := -1, -1
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			if b[r][c].Type == King && b[r][c].Color == color {
				kr, kc = r, c
			}
		}
	}
	if kr == -1 {
		return true
	}
	enemy := color.Flip()
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			if b[r][c].Color == enemy && b[r][c].Type != None {
				for _, m := range rawMoves(b, r, c) {
					if m.ToR == kr && m.ToC == kc {
						return true
					}
				}
			}
		}
	}
	return false
}

func LegalMoves(b Board, color Color) []Move {
	var legal []Move
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			if b[r][c].Color != color || b[r][c].Type == None {
				continue
			}
			for _, m := range rawMoves(b, r, c) {
				nb := ApplyMove(b, m)
				if !IsInCheck(nb, color) {
					legal = append(legal, m)
				}
			}
		}
	}
	return legal
}

// MaterialCount returns the total material on the board for a colour.
func MaterialCount(b Board, color Color) float32 {
	vals := map[PieceType]float32{Pawn: 1, Knight: 3, Bishop: 3.2, Rook: 5, Queen: 9}
	var total float32
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			p := b[r][c]
			if p.Color == color {
				total += vals[p.Type]
			}
		}
	}
	return total
}
