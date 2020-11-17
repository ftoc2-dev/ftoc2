#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from transforms cimport *
from hermite cimport *

import numpy as np
cimport numpy as np

cdef void MonomialInterpolate(int d, int l, unsigned char[:] a):
	r"""
	Univariate Hermite interpolation with respect to the monomial basis.
	"""
	cdef int n = max(<int>log2(l) - d, 0)
	cdef np.ndarray a_pad = np.zeros(1 << (d + n), dtype = np.uint8)
	a_pad[:l] = a[:l]
	HermiteInterpolate(d, n, l, l, a_pad)
	for i in xrange(l):
		a[i] = a_pad[i]

cdef void MonomialEvaluate(int d, int l, int s, unsigned char[:] a):
	r""" 
	Univariate Hermite evaluation with respect to the monomial basis.
	"""
	cdef int c = s << d
	cdef int n = max(<int>log2(max(l, c)) - d, 0)
	cdef np.ndarray a_pad = np.zeros(1 << (n + d), dtype = np.uint8)
	a_pad[:l] = a[:l]
	HermiteEvaluate(d, n, c, l, a_pad)
	for i in xrange(c):
		a[i] = a_pad[i]

cdef void SmallMonomialToNewton(int l, unsigned char[:] a):
	r"""
	Conversion from monomial to Newton basis for l <= 2^8.
	"""
	MonomialToLCH(l, a)
	LCHToNewton(l, a)

cdef void MonomialToNewton(int d, int l, unsigned char[:] a):
	r"""
	Conversion from monomial to Newton basis.
	"""
	cdef int i
	TaylorExpansion(d, l, a)
	cdef int q = 1 << d
	cdef int l_1 = (l >> d) << d
	cdef int l_2 = l ^ l_1
	for i in xrange(0, l_1, q):
		SmallMonomialToNewton(q, a[i:])
	if l_2 > 0:
		SmallMonomialToNewton(l_2, a[l_1:])

cdef void SmallNewtonToMononial(int l, unsigned char[:] a):
	r"""
	Conversion from Newton to monomial basis for l <= 2^8.
	"""
	NewtonToLCH(l, a)
	LCHToMonomial(l, a)

cdef void NewtonToMonomial(int d, int l, unsigned char[:] a):
	r"""
	Conversion from Newton to monomial basis.
	"""
	cdef int i
	cdef int q = 1 << d
	cdef int l_1 = (l >> d) << d
	cdef int l_2 = l ^ l_1
	for i in xrange(0, l_1, q):
		SmallNewtonToMononial(q, a[i:])
	if l_2 > 0:
		SmallNewtonToMononial(l_2, a[l_1:])
	InverseTaylorExpansion(d, l, a)

cpdef void NewtonInterpolate(int d, int l, unsigned char[:] a):
	r"""
	Univariate Hermite interpolation with respect to the Newton basis.
	"""
	MonomialInterpolate(d, l, a)
	MonomialToNewton(d, l, a)

cpdef void NewtonEvaluate(int d, int l, int s, unsigned char[:] a):
	r"""
	Univariate Hermite evaluation with respect to the Newton basis.
	"""
	NewtonToMonomial(d, l, a)
	MonomialEvaluate(d, l, s, a)

def InfoSet(m, l, b):
	if m == 1:
		for i in xrange(l):
			yield i, i
		return
	for i in xrange(l):
		for j in InfoSet(m - 1, l - i, b):
			yield i + b * j[0], i + j[1]

class Codeword(np.ndarray):
	r""" Codeword class. """
	def __new__(subtype, d, m, l, s):
		r"""
		Inputs:
			d - field degree,
			m - number of variables,
			l - degree bound,
			s - derivative order bound.
		"""
		b = s << d
		assert l <= b, "Invalid parameters: l > s * q"
		codeword = super(Codeword, subtype).__new__(subtype, b ** m, np.uint8)
		codeword[:] = 0
		codeword.code_parameters = (d, m, l, s)
		codeword.b = b
		return codeword
	def __array_finalize__(self, codeword):
		if codeword is None:
			return
		self.b = getattr(codeword, 'b', None)
		self.code_parameters = getattr(codeword, 'code_parameters', None)
	def info_set(self):
		r""" Generator for the information set of the codeword. """
		_, m, l, s = self.code_parameters
		return InfoSet(m, l, self.b)

def RecoverPolynomial(d, m, l, C):
	r"""
	Computes the polynomial that corresponds to a message.
	"""
	if m == 1:
		NewtonInterpolate(d, l, C)
		return
	b = C.b
	n = b ** (m - 1)
	for i in xrange(l):
		RecoverPolynomial(d, m - 1, l - i, C[i * n:])
	for i, w in InfoSet(m - 1, l, b):
		NewtonInterpolate(d, l - w, C[i::n])

def EvalSet(d, m, l, s, b):
	n = min(l, s << d)
	if m == 1:
		for i in xrange(n):
			yield i, i, i >> d
		return
	for i in xrange(n):
		o = i >> d
		for j in EvalSet(d, m - 1, l - i, s - o, b):
			yield i + b * j[0], i + j[1], o + j[2]

def EncodePolynomial(d, m, l, s, C):
	r"""
	Computes the codewords that corresponds to a polynomial.
	"""
	if m == 1:
		NewtonEvaluate(d, l, s, C)
		return
	b = C.b
	n = b ** (m - 1)
	for i, w, o in EvalSet(d, m - 1, l, s, b):
		NewtonEvaluate(d, l - w, s - o, C[i::n])
	for i in xrange(min(l, s) << d):
		o = i >> d
		EncodePolynomial(d, m - 1, l - o, s - o, C[i * n:])

def SystematicallyEncode(C):
	r"""
	Systematic encoding algoritm.
	"""
	d, m, l, s = C.code_parameters
	if m == 1:
		# There is no need to work on the Newton basis.
		MonomialInterpolate(d, l, C)
		MonomialEvaluate(d, l, s, C)
		return
	RecoverPolynomial(d, m, l, C)
	EncodePolynomial(d, m, l, s, C)
