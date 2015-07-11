// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file must be kept in sync with index_no_bound_checks.go.

//+build bounds

package mat64

import "github.com/gonum/blas"

// At returns the element at row r, column c.
func (m *Dense) At(r, c int) float64 {
	return m.at(r, c)
}

func (m *Dense) at(r, c int) float64 {
	if r >= m.Mat.Rows || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= m.Mat.Cols || c < 0 {
		panic(ErrColAccess)
	}
	return m.Mat.Data[r*m.Mat.Stride+c]
}

// Set sets the element at row r, column c to the value v.
func (m *Dense) Set(r, c int, v float64) {
	m.set(r, c, v)
}

func (m *Dense) set(r, c int, v float64) {
	if r >= m.Mat.Rows || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= m.Mat.Cols || c < 0 {
		panic(ErrColAccess)
	}
	m.Mat.Data[r*m.Mat.Stride+c] = v
}

// At returns the element at row r, column c. It panics if c is not zero.
func (v *Vector) At(r, c int) float64 {
	if c != 0 {
		panic(ErrColAccess)
	}
	return v.at(r)
}

func (v *Vector) at(r int) float64 {
	if r < 0 || r >= v.n {
		panic(ErrRowAccess)
	}
	return v.Mat.Data[r*v.Mat.Inc]
}

// Set sets the element at (r,c) to the value val. It panics if c is not zero.
func (v *Vector) Set(r, c int, val float64) {
	if c != 0 {
		panic(ErrColAccess)
	}
	v.set(r, val)
}

func (v *Vector) set(r int, val float64) {
	if r < 0 || r >= v.n {
		panic(ErrRowAccess)
	}
	v.Mat.Data[r*v.Mat.Inc] = val
}

// At returns the element at row r and column c.
func (t *SymDense) At(r, c int) float64 {
	return t.at(r, c)
}

func (t *SymDense) at(r, c int) float64 {
	if r >= t.Mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.Mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if r > c {
		r, c = c, r
	}
	return t.Mat.Data[r*t.Mat.Stride+c]
}

// SetSym sets the elements at (r,c) and (c,r) to the value v.
func (t *SymDense) SetSym(r, c int, v float64) {
	t.set(r, c, v)
}

func (t *SymDense) set(r, c int, v float64) {
	if r >= t.Mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.Mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if r > c {
		r, c = c, r
	}
	t.Mat.Data[r*t.Mat.Stride+c] = v
}

// At returns the element at row r, column c.
func (t *TriDense) At(r, c int) float64 {
	return t.at(r, c)
}

func (t *TriDense) at(r, c int) float64 {
	if r >= t.Mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.Mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.Mat.Uplo == blas.Upper {
		if r > c {
			return 0
		}
		return t.Mat.Data[r*t.Mat.Stride+c]
	}
	if r < c {
		return 0
	}
	return t.Mat.Data[r*t.Mat.Stride+c]
}

// SetTri sets the element of the triangular matrix at row r, column c to the value v.
// It panics if the location is outside the appropriate half of the matrix.
func (t *TriDense) SetTri(r, c int, v float64) {
	t.set(r, c, v)
}

func (t *TriDense) set(r, c int, v float64) {
	if r >= t.Mat.N || r < 0 {
		panic(ErrRowAccess)
	}
	if c >= t.Mat.N || c < 0 {
		panic(ErrColAccess)
	}
	if t.Mat.Uplo == blas.Upper && r > c {
		panic("mat64: triangular set out of bounds")
	}
	if t.Mat.Uplo == blas.Lower && r < c {
		panic("mat64: triangular set out of bounds")
	}
	t.Mat.Data[r*t.Mat.Stride+c] = v
}
