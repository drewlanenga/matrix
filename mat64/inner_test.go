// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"math"
	"testing"

	"github.com/gonum/blas/blas64"
	"gopkg.in/check.v1"
)

func (s *S) TestInner(c *check.C) {
	for i, test := range []struct {
		x []float64
		y []float64
		m [][]float64
	}{
		{
			x: []float64{5},
			y: []float64{10},
			m: [][]float64{{2}},
		},
		{
			x: []float64{5, 6, 1},
			y: []float64{10},
			m: [][]float64{{2}, {-3}, {5}},
		},
		{
			x: []float64{5},
			y: []float64{10, 15},
			m: [][]float64{{2, -3}},
		},
		{
			x: []float64{1, 5},
			y: []float64{10, 15},
			m: [][]float64{
				{2, -3},
				{4, -1},
			},
		},
		{
			x: []float64{2, 3, 9},
			y: []float64{8, 9},
			m: [][]float64{
				{2, 3},
				{4, 5},
				{6, 7},
			},
		},
		{
			x: []float64{2, 3},
			y: []float64{8, 9, 9},
			m: [][]float64{
				{2, 3, 6},
				{4, 5, 7},
			},
		},
	} {
		for _, inc := range []struct{ x, y int }{
			{1, 1},
			{1, 2},
			{2, 1},
			{2, 2},
		} {
			x := NewDense(1, len(test.x), test.x)
			m := NewDense(flatten(test.m))
			mWant := NewDense(flatten(test.m))
			y := NewDense(len(test.y), 1, test.y)

			var tmp, cell Dense
			tmp.Mul(mWant, y)
			cell.Mul(x, &tmp)

			rm, cm := cell.Dims()
			c.Check(rm, check.Equals, 1, check.Commentf("Test %v result doesn't have 1 row", i))
			c.Check(cm, check.Equals, 1, check.Commentf("Test %v result doesn't have 1 column", i))

			want := cell.At(0, 0)

			got := Inner(makeVectorInc(inc.x, test.x), m, makeVectorInc(inc.y, test.y))
			c.Check(want, check.Equals, got, check.Commentf("Test %v: want %v, got %v", i, want, got))
		}
	}
}

func (s *S) TestInnerSym(c *check.C) {
	for _, inc := range []struct{ x, y int }{
		{1, 1},
		{1, 2},
		{2, 1},
		{2, 2},
	} {
		n := 10
		xData := make([]float64, n)
		yData := make([]float64, n)
		data := make([]float64, n*n)
		for i := 0; i < n; i++ {
			xData[i] = float64(i)
			yData[i] = float64(i)
			for j := i; j < n; j++ {
				data[i*n+j] = float64(i*n + j)
				data[j*n+i] = data[i*n+j]
			}
		}
		x := makeVectorInc(inc.x, xData)
		y := makeVectorInc(inc.y, yData)
		m := NewDense(n, n, data)
		ans := Inner(x, m, y)
		sym := NewSymDense(n, data)
		// Poison the lower half of data to ensure it is not used.
		for i := 1; i < n; i++ {
			for j := 0; j < i; j++ {
				data[i*n+j] = math.NaN()
			}
		}

		if math.Abs(Inner(x, sym, y)-ans) > 1e-14 {
			c.Error("inner different symmetric and dense")
		}
	}
}

func makeVectorInc(inc int, f []float64) *Vector {
	v := &Vector{
		n: len(f),
		Mat: blas64.Vector{
			Inc:  inc,
			Data: make([]float64, (len(f)-1)*inc+1),
		},
	}

	// Contaminate backing data in all positions...
	const base = 100
	for i := range v.Mat.Data {
		v.Mat.Data[i] = float64(i + base)
	}

	// then write real elements.
	for i := range f {
		v.Mat.Data[i*inc] = f[i]
	}
	return v
}

func benchmarkInner(b *testing.B, m, n int) {
	x := NewVector(m, nil)
	randomSlice(x.Mat.Data)
	y := NewVector(n, nil)
	randomSlice(y.Mat.Data)
	data := make([]float64, m*n)
	randomSlice(data)
	mat := &Dense{Mat: blas64.General{Rows: m, Cols: n, Stride: n, Data: data}, capRows: m, capCols: n}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Inner(x, mat, y)
	}
}

func BenchmarkInnerSmSm(b *testing.B) {
	benchmarkInner(b, Sm, Sm)
}

func BenchmarkInnerMedMed(b *testing.B) {
	benchmarkInner(b, Med, Med)
}

func BenchmarkInnerLgLg(b *testing.B) {
	benchmarkInner(b, Lg, Lg)
}

func BenchmarkInnerLgSm(b *testing.B) {
	benchmarkInner(b, Lg, Sm)
}
