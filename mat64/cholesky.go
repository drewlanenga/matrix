// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the CholeskyDecomposition class from Jama 1.0.3.

package mat64

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/lapack/lapack64"
)

const badTriangle = "mat64: invalid triangle"

// Cholesky calculates the Cholesky decomposition of the matrix A and returns
// whether the matrix is positive definite. The returned matrix is either a
// lower triangular matrix such that A = L * L^T or an upper triangular matrix
// such that A = U^T * U depending on the upper parameter.
func (t *TriDense) Cholesky(a *SymDense, upper bool) (ok bool) {
	n := a.Symmetric()
	if t.isZero() {
		t.Mat = blas64.Triangular{
			N:      n,
			Stride: n,
			Diag:   blas.NonUnit,
			Data:   use(t.Mat.Data, n*n),
		}
		if upper {
			t.Mat.Uplo = blas.Upper
		} else {
			t.Mat.Uplo = blas.Lower
		}
	} else {
		if n != t.Mat.N {
			panic(ErrShape)
		}
		if (upper && t.Mat.Uplo != blas.Upper) || (!upper && t.Mat.Uplo != blas.Lower) {
			panic(ErrTriangle)
		}
	}
	copySymIntoTriangle(t, a)

	// Potrf modifies the data in place
	_, ok = lapack64.Potrf(
		blas64.Symmetric{
			N:      t.Mat.N,
			Stride: t.Mat.Stride,
			Data:   t.Mat.Data,
			Uplo:   t.Mat.Uplo,
		})
	return ok
}

// SolveCholesky finds the matrix m that solves A * m = b where A = L * L^T or
// A = U^T * U, and U or L are represented by t, placing the result in the
// receiver.
func (m *Dense) SolveCholesky(t Triangular, b Matrix) {
	_, n := t.Dims()
	bm, bn := b.Dims()
	if n != bm {
		panic(ErrShape)
	}

	m.reuseAs(bm, bn)
	if b != m {
		m.Copy(b)
	}

	// TODO(btracey): Implement an algorithm that doesn't require a copy into
	// a blas64.Triangular.
	ta := getBlasTriangular(t)

	switch ta.Uplo {
	case blas.Upper:
		blas64.Trsm(blas.Left, blas.Trans, 1, ta, m.Mat)
		blas64.Trsm(blas.Left, blas.NoTrans, 1, ta, m.Mat)
	case blas.Lower:
		blas64.Trsm(blas.Left, blas.NoTrans, 1, ta, m.Mat)
		blas64.Trsm(blas.Left, blas.Trans, 1, ta, m.Mat)
	default:
		panic(badTriangle)
	}
}

// SolveCholeskyVec finds the vector v that solves A * v = b where A = L * L^T or
// A = U^T * U, and U or L are represented by t, placing the result in the
// receiver.
func (v *Vector) SolveCholeskyVec(t Triangular, b *Vector) {
	_, n := t.Dims()
	vn := b.Len()
	if vn != n {
		panic(ErrShape)
	}
	v.reuseAs(n)
	if v != b {
		v.CopyVec(b)
	}
	ta := getBlasTriangular(t)
	switch ta.Uplo {
	case blas.Upper:
		blas64.Trsv(blas.Trans, ta, v.Mat)
		blas64.Trsv(blas.NoTrans, ta, v.Mat)
	case blas.Lower:
		blas64.Trsv(blas.NoTrans, ta, v.Mat)
		blas64.Trsv(blas.Trans, ta, v.Mat)
	default:
		panic(badTriangle)
	}
}

// SolveTri finds the matrix x that solves op(A) * X = B where A is a triangular
// matrix and op is specified by trans.
func (m *Dense) SolveTri(a Triangular, trans bool, b Matrix) {
	n, _ := a.Triangle()
	bm, bn := b.Dims()
	if n != bm {
		panic(ErrShape)
	}

	m.reuseAs(bm, bn)
	if b != m {
		m.Copy(b)
	}

	// TODO(btracey): Implement an algorithm that doesn't require a copy into
	// a blas64.Triangular.
	ta := getBlasTriangular(a)

	t := blas.NoTrans
	if trans {
		t = blas.Trans
	}
	switch ta.Uplo {
	case blas.Upper, blas.Lower:
		blas64.Trsm(blas.Left, t, 1, ta, m.Mat)
	default:
		panic(badTriangle)
	}
}
