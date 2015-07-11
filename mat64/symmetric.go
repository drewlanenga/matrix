package mat64

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

var (
	symDense *SymDense

	_ Matrix         = symDense
	_ Symmetric      = symDense
	_ RawSymmetricer = symDense
)

const badSymTriangle = "mat64: blas64.Symmetric not upper"

// SymDense is a symmetric matrix that uses Dense storage.
type SymDense struct {
	Mat blas64.Symmetric
}

// Symmetric represents a symmetric matrix (where the element at {i, j} equals
// the element at {j, i}). Symmetric matrices are always square.
type Symmetric interface {
	Matrix
	// Symmetric returns the number of rows/columns in the matrix.
	Symmetric() int
}

// A RawSymmetricer can return a view of itself as a BLAS Symmetric matrix.
type RawSymmetricer interface {
	RawSymmetric() blas64.Symmetric
}

// NewSymDense constructs an n x n symmetric matrix. If len(mat) == n * n,
// mat will be used to hold the underlying data, or if mat == nil, new data will be allocated.
// The underlying data representation is the same as a Dense matrix, except
// the values of the entries in the lower triangular portion are completely ignored.
func NewSymDense(n int, mat []float64) *SymDense {
	if n < 0 {
		panic("mat64: negative dimension")
	}
	if mat != nil && n*n != len(mat) {
		panic(ErrShape)
	}
	if mat == nil {
		mat = make([]float64, n*n)
	}
	return &SymDense{blas64.Symmetric{
		N:      n,
		Stride: n,
		Data:   mat,
		Uplo:   blas.Upper,
	}}
}

func (s *SymDense) Dims() (r, c int) {
	return s.Mat.N, s.Mat.N
}

func (s *SymDense) Symmetric() int {
	return s.Mat.N
}

// RawSymmetric returns the matrix as a blas64.Symmetric. The returned
// value must be stored in upper triangular format.
func (s *SymDense) RawSymmetric() blas64.Symmetric {
	return s.Mat
}

func (s *SymDense) isZero() bool {
	return s.Mat.N == 0
}

func (s *SymDense) AddSym(a, b Symmetric) {
	n := a.Symmetric()
	if n != b.Symmetric() {
		panic(ErrShape)
	}
	if s.isZero() {
		s.Mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Data:   use(s.Mat.Data, n*n),
			Uplo:   blas.Upper,
		}
	} else if s.Mat.N != n {
		panic(ErrShape)
	}

	if a, ok := a.(RawSymmetricer); ok {
		if b, ok := b.(RawSymmetricer); ok {
			amat, bmat := a.RawSymmetric(), b.RawSymmetric()
			for i := 0; i < n; i++ {
				btmp := bmat.Data[i*bmat.Stride+i : i*bmat.Stride+n]
				stmp := s.Mat.Data[i*s.Mat.Stride+i : i*s.Mat.Stride+n]
				for j, v := range amat.Data[i*amat.Stride+i : i*amat.Stride+n] {
					stmp[j] = v + btmp[j]
				}
			}
			return
		}
	}

	for i := 0; i < n; i++ {
		stmp := s.Mat.Data[i*s.Mat.Stride : i*s.Mat.Stride+n]
		for j := i; j < n; j++ {
			stmp[j] = a.At(i, j) + b.At(i, j)
		}
	}
}

func (s *SymDense) CopySym(a Symmetric) int {
	n := a.Symmetric()
	n = min(n, s.Mat.N)
	switch a := a.(type) {
	case RawSymmetricer:
		amat := a.RawSymmetric()
		if amat.Uplo != blas.Upper {
			panic(badSymTriangle)
		}
		for i := 0; i < n; i++ {
			copy(s.Mat.Data[i*s.Mat.Stride+i:i*s.Mat.Stride+n], amat.Data[i*amat.Stride+i:i*amat.Stride+n])
		}
	default:
		for i := 0; i < n; i++ {
			stmp := s.Mat.Data[i*s.Mat.Stride : i*s.Mat.Stride+n]
			for j := i; j < n; j++ {
				stmp[j] = a.At(i, j)
			}
		}
	}
	return n
}

// SymRankOne performs a symetric rank-one update to the matrix a and stores
// the result in the receiver
//  s = a + alpha * x * x'
func (s *SymDense) SymRankOne(a Symmetric, alpha float64, x []float64) {
	n := s.Mat.N
	var w SymDense
	if s == a {
		w = *s
	}
	if w.isZero() {
		w.Mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Uplo:   blas.Upper,
			Data:   use(w.Mat.Data, n*n),
		}
	} else if n != w.Mat.N {
		panic(ErrShape)
	}
	if s != a {
		w.CopySym(a)
	}
	if len(x) != n {
		panic(ErrShape)
	}
	blas64.Syr(alpha, blas64.Vector{Inc: 1, Data: x}, w.Mat)
	*s = w
	return
}

// RankTwo performs a symmmetric rank-two update to the matrix a and stores
// the result in the receiver
//  m = a + alpha * (x * y' + y * x')
func (s *SymDense) RankTwo(a Symmetric, alpha float64, x, y []float64) {
	n := s.Mat.N
	var w SymDense
	if s == a {
		w = *s
	}
	if w.isZero() {
		w.Mat = blas64.Symmetric{
			N:      n,
			Stride: n,
			Uplo:   blas.Upper,
			Data:   use(w.Mat.Data, n*n),
		}
	} else if n != w.Mat.N {
		panic(ErrShape)
	}
	if s != a {
		w.CopySym(a)
	}
	if len(x) != n {
		panic(ErrShape)
	}
	if len(y) != n {
		panic(ErrShape)
	}
	blas64.Syr2(alpha, blas64.Vector{Inc: 1, Data: x}, blas64.Vector{Inc: 1, Data: y}, w.Mat)
	*s = w
	return
}
