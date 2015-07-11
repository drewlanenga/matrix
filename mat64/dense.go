// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"bytes"
	"encoding/binary"

	"github.com/gonum/blas/blas64"
)

var (
	matrix *Dense

	_ Matrix       = matrix
	_ Mutable      = matrix
	_ Vectorer     = matrix
	_ VectorSetter = matrix

	_ Cloner       = matrix
	_ Viewer       = matrix
	_ RowViewer    = matrix
	_ ColViewer    = matrix
	_ RawRowViewer = matrix
	_ Grower       = matrix

	_ Adder     = matrix
	_ Suber     = matrix
	_ Muler     = matrix
	_ Dotter    = matrix
	_ ElemMuler = matrix
	_ ElemDiver = matrix
	_ Exper     = matrix

	_ Scaler  = matrix
	_ Applyer = matrix

	_ TransposeCopier = matrix
	// _ TransposeViewer = matrix

	_ Tracer = matrix
	_ Normer = matrix
	_ Sumer  = matrix

	_ Uer = matrix
	_ Ler = matrix

	_ Stacker   = matrix
	_ Augmenter = matrix

	_ Equaler       = matrix
	_ ApproxEqualer = matrix

	_ RawMatrixSetter = matrix
	_ RawMatrixer     = matrix

	_ Reseter = matrix
)

// Dense is a dense matrix representation.
type Dense struct {
	Mat blas64.General

	capRows, capCols int
}

// NewDense creates a new matrix of type Dense with dimensions r and c.
// If the mat argument is nil, a new data slice is allocated.
//
// The data must be arranged in row-major order, i.e. the (i*c + j)-th
// element in mat is the {i, j}-th element in the matrix.
func NewDense(r, c int, mat []float64) *Dense {
	if mat != nil && r*c != len(mat) {
		panic(ErrShape)
	}
	if mat == nil {
		mat = make([]float64, r*c)
	}
	return &Dense{
		Mat: blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   mat,
		},
		capRows: r,
		capCols: c,
	}
}

// reuseAs resizes an empty matrix to a r×c matrix,
// or checks that a non-empty matrix is r×c.
func (m *Dense) reuseAs(r, c int) {
	if m.Mat.Rows > m.capRows || m.Mat.Cols > m.capCols {
		// Panic as a string, not a mat64.Error.
		panic("mat64: caps not correctly set")
	}
	if m.isZero() {
		m.Mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   use(m.Mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
		return
	}
	if r != m.Mat.Rows || c != m.Mat.Cols {
		panic(ErrShape)
	}
}

func (m *Dense) isZero() bool {
	// It must be the case that m.Dims() returns
	// zeros in this case. See comment in Reset().
	return m.Mat.Stride == 0
}

// DenseCopyOf returns a newly allocated copy of the elements of a.
func DenseCopyOf(a Matrix) *Dense {
	d := &Dense{}
	d.Clone(a)
	return d
}

// SetRawMatrix sets the underlying blas64.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in b.
func (m *Dense) SetRawMatrix(b blas64.General) {
	m.capRows, m.capCols = b.Rows, b.Cols
	m.Mat = b
}

// RawMatrix returns the underlying blas64.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.General.
func (m *Dense) RawMatrix() blas64.General { return m.Mat }

// Dims returns the number of rows and columns in the matrix.
func (m *Dense) Dims() (r, c int) { return m.Mat.Rows, m.Mat.Cols }

// Caps returns the number of rows and columns in the backing matrix.
func (m *Dense) Caps() (r, c int) { return m.capRows, m.capCols }

// Col copies the elements in the jth column of the matrix into the slice dst.
// If the provided slice is nil, a new slice is first allocated.
//
// See the Vectorer interface for more information.
func (m *Dense) Col(dst []float64, j int) []float64 {
	if j >= m.Mat.Cols || j < 0 {
		panic(ErrColAccess)
	}

	if dst == nil {
		dst = make([]float64, m.Mat.Rows)
	}
	dst = dst[:min(len(dst), m.Mat.Rows)]
	blas64.Copy(len(dst),
		blas64.Vector{Inc: m.Mat.Stride, Data: m.Mat.Data[j:]},
		blas64.Vector{Inc: 1, Data: dst},
	)

	return dst
}

// ColView returns a Vector reflecting col j, backed by the matrix data.
//
// See ColViewer for more information.
func (m *Dense) ColView(j int) *Vector {
	if j >= m.Mat.Cols || j < 0 {
		panic(ErrColAccess)
	}
	return &Vector{
		Mat: blas64.Vector{
			Inc:  m.Mat.Stride,
			Data: m.Mat.Data[j : (m.Mat.Rows-1)*m.Mat.Stride+j+1],
		},
		n: m.Mat.Rows,
	}
}

// SetCol sets the elements of the matrix in the specified column to the values
// of src.
//
// See the VectorSetter interface for more information.
func (m *Dense) SetCol(j int, src []float64) int {
	if j >= m.Mat.Cols || j < 0 {
		panic(ErrColAccess)
	}

	blas64.Copy(min(len(src), m.Mat.Rows),
		blas64.Vector{Inc: 1, Data: src},
		blas64.Vector{Inc: m.Mat.Stride, Data: m.Mat.Data[j:]},
	)

	return min(len(src), m.Mat.Rows)
}

// Row copies the elements in the ith row of the matrix into the slice dst.
// If the provided slice is nil, a new slice is first allocated.
//
// See the Vectorer interface for more information.
func (m *Dense) Row(dst []float64, i int) []float64 {
	if i >= m.Mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}

	if dst == nil {
		dst = make([]float64, m.Mat.Cols)
	}
	copy(dst, m.rowView(i))

	return dst
}

// SetRow sets the elements of the matrix in the specified row to the values of
// src.
//
// See the VectorSetter interface for more information.
func (m *Dense) SetRow(i int, src []float64) int {
	if i >= m.Mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}

	copy(m.rowView(i), src)

	return min(len(src), m.Mat.Cols)
}

// RowView returns a Vector reflecting row i, backed by the matrix data.
//
// See RowViewer for more information.
func (m *Dense) RowView(i int) *Vector {
	if i >= m.Mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}
	return &Vector{
		Mat: blas64.Vector{
			Inc:  1,
			Data: m.Mat.Data[i*m.Mat.Stride : i*m.Mat.Stride+m.Mat.Cols],
		},
		n: m.Mat.Cols,
	}
}

// RawRowView returns a slice backed by the same array as backing the
// receiver.
func (m *Dense) RawRowView(i int) []float64 {
	if i >= m.Mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}
	return m.rowView(i)
}

func (m *Dense) rowView(r int) []float64 {
	return m.Mat.Data[r*m.Mat.Stride : r*m.Mat.Stride+m.Mat.Cols]
}

// View returns a new Matrix that shares backing data with the receiver.
// The new matrix is located from row i, column j extending r rows and c
// columns.
func (m *Dense) View(i, j, r, c int) Matrix {
	mr, mc := m.Dims()
	if i < 0 || i >= mr || j < 0 || j >= mc || r <= 0 || i+r > mr || c <= 0 || j+c > mc {
		panic(ErrIndexOutOfRange)
	}
	t := *m
	t.Mat.Data = t.Mat.Data[i*t.Mat.Stride+j : (i+r-1)*t.Mat.Stride+(j+c)]
	t.Mat.Rows = r
	t.Mat.Cols = c
	t.capRows -= i
	t.capCols -= j
	return &t
}

// Grow returns an expanded copy of the receiver. The copy is expanded
// by r rows and c columns. If the dimensions of the new copy are outside
// the caps of the receiver a new allocation is made, otherwise not.
func (m *Dense) Grow(r, c int) Matrix {
	if r < 0 || c < 0 {
		panic(ErrIndexOutOfRange)
	}
	if r == 0 && c == 0 {
		return m
	}

	r += m.Mat.Rows
	c += m.Mat.Cols

	var t Dense
	switch {
	case m.Mat.Rows == 0 || m.Mat.Cols == 0:
		t.Mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			// We zero because we don't know how the matrix will be used.
			// In other places, the mat is immediately filled with a result;
			// this is not the case here.
			Data: useZeroed(m.Mat.Data, r*c),
		}
	case r > m.capRows || c > m.capCols:
		cr := max(r, m.capRows)
		cc := max(c, m.capCols)
		t.Mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: cc,
			Data:   make([]float64, cr*cc),
		}
		t.capRows = cr
		t.capCols = cc
		// Copy the complete matrix over to the new matrix.
		// Including elements not currently visible.
		r, c, m.Mat.Rows, m.Mat.Cols = m.Mat.Rows, m.Mat.Cols, m.capRows, m.capCols
		t.Copy(m)
		m.Mat.Rows, m.Mat.Cols = r, c
		return &t
	default:
		t.Mat = blas64.General{
			Data:   m.Mat.Data[:(r-1)*m.Mat.Stride+c],
			Rows:   r,
			Cols:   c,
			Stride: m.Mat.Stride,
		}
	}
	t.capRows = r
	t.capCols = c
	return &t
}

// Reset zeros the dimensions of the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// See the Reseter interface for more information.
func (m *Dense) Reset() {
	// No change of Stride, Rows and Cols to 0
	// may be made unless all are set to 0.
	m.Mat.Rows, m.Mat.Cols, m.Mat.Stride = 0, 0, 0
	m.capRows, m.capCols = 0, 0
	m.Mat.Data = m.Mat.Data[:0]
}

// Clone makes a copy of a into the receiver, overwriting the previous value of
// the receiver. The clone operation does not make any restriction on shape.
//
// See the Cloner interface for more information.
func (m *Dense) Clone(a Matrix) {
	r, c := a.Dims()
	mat := blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
	}
	m.capRows, m.capCols = r, c
	switch a := a.(type) {
	case RawMatrixer:
		amat := a.RawMatrix()
		mat.Data = make([]float64, r*c)
		for i := 0; i < r; i++ {
			copy(mat.Data[i*c:(i+1)*c], amat.Data[i*amat.Stride:i*amat.Stride+c])
		}
	case Vectorer:
		mat.Data = use(m.Mat.Data, r*c)
		for i := 0; i < r; i++ {
			a.Row(mat.Data[i*c:(i+1)*c], i)
		}
	default:
		mat.Data = use(m.Mat.Data, r*c)
		m.Mat = mat
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				m.set(i, j, a.At(i, j))
			}
		}
		return
	}
	m.Mat = mat
}

// Copy makes a copy of elements of a into the receiver. It is similar to the
// built-in copy; it copies as much as the overlap between the two matrices and
// returns the number of rows and columns it copied.
//
// See the Copier interface for more information.
func (m *Dense) Copy(a Matrix) (r, c int) {
	r, c = a.Dims()
	r = min(r, m.Mat.Rows)
	c = min(c, m.Mat.Cols)

	switch a := a.(type) {
	case RawMatrixer:
		amat := a.RawMatrix()
		for i := 0; i < r; i++ {
			copy(m.Mat.Data[i*m.Mat.Stride:i*m.Mat.Stride+c], amat.Data[i*amat.Stride:i*amat.Stride+c])
		}
	case Vectorer:
		for i := 0; i < r; i++ {
			a.Row(m.Mat.Data[i*m.Mat.Stride:i*m.Mat.Stride+c], i)
		}
	default:
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				m.set(r, c, a.At(r, c))
			}
		}
	}

	return r, c
}

// U places the upper triangular matrix of a in the receiver.
//
// See the Uer interface for more information.
func (m *Dense) U(a Matrix) {
	ar, ac := a.Dims()
	if ar != ac {
		panic(ErrSquare)
	}

	if m == a {
		m.zeroLower()
		return
	}
	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		amat := a.RawMatrix()
		copy(m.Mat.Data[:ac], amat.Data[:ac])
		for j, ja, jm := 1, amat.Stride, m.Mat.Stride; ja < ar*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+m.Mat.Stride {
			zero(m.Mat.Data[jm : jm+j])
			copy(m.Mat.Data[jm+j:jm+ac], amat.Data[ja+j:ja+ac])
		}
		return
	}

	if a, ok := a.(Vectorer); ok {
		row := make([]float64, ac)
		copy(m.Mat.Data[:m.Mat.Cols], a.Row(row, 0))
		for r := 1; r < ar; r++ {
			zero(m.Mat.Data[r*m.Mat.Stride : r*(m.Mat.Stride+1)])
			copy(m.Mat.Data[r*(m.Mat.Stride+1):r*m.Mat.Stride+m.Mat.Cols], a.Row(row, r))
		}
		return
	}

	m.zeroLower()
	for r := 0; r < ar; r++ {
		for c := r; c < ac; c++ {
			m.set(r, c, a.At(r, c))
		}
	}
}

func (m *Dense) zeroLower() {
	for i := 1; i < m.Mat.Rows; i++ {
		zero(m.Mat.Data[i*m.Mat.Stride : i*m.Mat.Stride+i])
	}
}

// L places the lower triangular matrix of a in the receiver.
//
// See the Ler interface for more information.
func (m *Dense) L(a Matrix) {
	ar, ac := a.Dims()
	if ar != ac {
		panic(ErrSquare)
	}

	if m == a {
		m.zeroUpper()
		return
	}
	m.reuseAs(ar, ac)

	if a, ok := a.(RawMatrixer); ok {
		amat := a.RawMatrix()
		copy(m.Mat.Data[:ar], amat.Data[:ar])
		for j, ja, jm := 1, amat.Stride, m.Mat.Stride; ja < ac*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+m.Mat.Stride {
			zero(m.Mat.Data[jm : jm+j])
			copy(m.Mat.Data[jm+j:jm+ar], amat.Data[ja+j:ja+ar])
		}
		return
	}

	if a, ok := a.(Vectorer); ok {
		row := make([]float64, ac)
		for r := 0; r < ar; r++ {
			a.Row(row[:r+1], r)
			m.SetRow(r, row)
		}
		return
	}

	m.zeroUpper()
	for c := 0; c < ac; c++ {
		for r := c; r < ar; r++ {
			m.set(r, c, a.At(r, c))
		}
	}
}

func (m *Dense) zeroUpper() {
	for i := 0; i < m.Mat.Rows-1; i++ {
		zero(m.Mat.Data[i*m.Mat.Stride+i+1 : (i+1)*m.Mat.Stride])
	}
}

// TCopy makes a copy of the transpose the matrix represented by a, placing the
// result into the receiver.
//
// See the TransposeCopier interface for more information.
func (m *Dense) TCopy(a Matrix) {
	ar, ac := a.Dims()

	var w Dense
	if m != a {
		w = *m
	}
	w.reuseAs(ac, ar)

	switch a := a.(type) {
	case *Dense:
		for i := 0; i < ac; i++ {
			for j := 0; j < ar; j++ {
				w.set(i, j, a.at(j, i))
			}
		}
	default:
		for i := 0; i < ac; i++ {
			for j := 0; j < ar; j++ {
				w.set(i, j, a.At(j, i))
			}
		}
	}
	*m = w
}

// Stack appends the rows of b onto the rows of a, placing the result into the
// receiver.
//
// See the Stacker interface for more information.
func (m *Dense) Stack(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != bc || m == a || m == b {
		panic(ErrShape)
	}

	m.reuseAs(ar+br, ac)

	m.Copy(a)
	w := m.View(ar, 0, br, bc).(*Dense)
	w.Copy(b)
}

// Augment creates the augmented matrix of a and b, where b is placed in the
// greater indexed columns.
//
// See the Augmenter interface for more information.
func (m *Dense) Augment(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || m == a || m == b {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac+bc)

	m.Copy(a)
	w := m.View(0, ac, br, bc).(*Dense)
	w.Copy(b)
}

// MarshalBinary encodes the receiver into a binary form and returns the result.
//
// Dense is little-endian encoded as follows:
//   0 -  8  number of rows    (int64)
//   8 - 16  number of columns (int64)
//  16 - ..  matrix data elements (float64)
//           [0,0] [0,1] ... [0,ncols-1]
//           [1,0] [1,1] ... [1,ncols-1]
//           ...
//           [nrows-1,0] ... [nrows-1,ncols-1]
func (m Dense) MarshalBinary() ([]byte, error) {
	buf := bytes.NewBuffer(make([]byte, 0, m.Mat.Rows*m.Mat.Cols*sizeFloat64+2*sizeInt64))
	err := binary.Write(buf, defaultEndian, int64(m.Mat.Rows))
	if err != nil {
		return nil, err
	}
	err = binary.Write(buf, defaultEndian, int64(m.Mat.Cols))
	if err != nil {
		return nil, err
	}

	for i := 0; i < m.Mat.Rows; i++ {
		for _, v := range m.rowView(i) {
			err = binary.Write(buf, defaultEndian, v)
			if err != nil {
				return nil, err
			}
		}
	}
	return buf.Bytes(), err
}

// UnmarshalBinary decodes the binary form into the receiver.
// It panics if the receiver is a non-zero Dense matrix.
//
// See MarshalBinary for the on-disk layout.
func (m *Dense) UnmarshalBinary(data []byte) error {
	if !m.isZero() {
		panic("mat64: unmarshal into non-zero matrix")
	}

	buf := bytes.NewReader(data)
	var rows int64
	err := binary.Read(buf, defaultEndian, &rows)
	if err != nil {
		return err
	}
	var cols int64
	err = binary.Read(buf, defaultEndian, &cols)
	if err != nil {
		return err
	}

	m.Mat.Rows = int(rows)
	m.Mat.Cols = int(cols)
	m.Mat.Stride = int(cols)
	m.capRows = int(rows)
	m.capCols = int(cols)
	m.Mat.Data = use(m.Mat.Data, m.Mat.Rows*m.Mat.Cols)

	for i := range m.Mat.Data {
		err = binary.Read(buf, defaultEndian, &m.Mat.Data[i])
		if err != nil {
			return err
		}
	}

	return err
}
