#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from transforms cimport *

cpdef void Evaluate(int c, int l, unsigned char[:] a):
	r"""
	Algorithm 1.
	"""
	# Concert to the LCH basis
	MonomialToLCH(l, a)
	# Evaluate on the LCH basis
	LCHEval(0, c, l, a)

cdef void Interpolate(int c, int l, unsigned char[:] a):
	r"""
	Algorithm 3.
	"""
	# Compute high coefficients on the LCH basis
	MonomialToLCHHighCoeffs(c, l, a)
	# Compute remaining coefficients on the LCH basis
	InplaceLCHInterp(0, c, l, a)
	# Convert to the monomial basis
	LCHToMonomial(l, a)

cdef void PrepareLeft(int d, int n, int c, int l, unsigned char[:] a):
	r"""
	The function PrepareLeft from Algorithm 2.
	"""
	cdef int i
	cdef int qn_half = 1 << (d + n - 1)
	cdef int overlap = 1 << (n - 1)
	cdef int t = max(c, overlap)
	cdef int k = qn_half - overlap
	cdef int v = min(l - k, qn_half)
	cdef int u = min(v, 1 << n)
	for i in xrange(t, v):
		a[i] ^= a[k + i]
	k <<= 1
	for i in xrange(t, u):
		a[i] ^= a[k + i]

cdef void PrepareRight(int d, int n, int c, int l, unsigned char[:] a):
	r"""
	The function PrepareRight from Algorithm 4.
	"""
	cdef int i, j
	cdef int qn = 1 << (d + n)
	cdef int qn_half = qn >> 1
	cdef int twon = 1 << n
	cdef int overlap = twon >> 1
	cdef int k = qn_half - overlap
	for i in xrange(l - k - 1, c - 1, -1):
		a[i] ^= a[k + i]
	cdef int t = (c >> n) << n
	cdef int r = min(c ^ t, overlap)
	for i in xrange(t + twon, qn, twon):
		for j in xrange(i, i + r):
			a[j] ^= a[j - k]
	for i in xrange(t, qn, twon):
		for j in xrange(i + r, i + overlap):
			a[j] ^= a[j - k]

cpdef void HermiteEvaluate(int d, int n, int c, int l, unsigned char[:] a):
	r"""
	Algorithm 2.
	"""
	if n == 0:
		Evaluate(c, l, a)
		return
	cdef int qn_half = 1 << (d + n - 1)
	cdef int overlap = 1 << (n - 1)
	if l <= overlap:	
		# F_0 = F and D^{2^{n-1}}F_0 + F_1 = 0
		HermiteEvaluate(d, n - 1, min(c, qn_half), l, a)
		return
	cdef int l_lhs = min(l, qn_half)
	if c <= qn_half:
		PrepareLeft(d, n, 0, l, a)
		HermiteEvaluate(d, n - 1, c, l_lhs, a)
		return
	# Taylor expansion: compute F_0 and F_1
	cdef int i, j
	cdef int k = qn_half - overlap
	for i in xrange(l - k - 1, overlap - 1, -1):
		a[i] ^= a[k + i]
	# Add D^{2^{n-1}}F_0 to F_1
	cdef int u = (l_lhs >> n) << n
	cdef int v = l_lhs ^ u
	cdef int b = v >> (n - 1)
	v ^= b << (n - 1)
	# l_lhs = 2^{n-1}(2\floor{u/2^n}+b)+v
	cdef int l_rhs = max(l - l_lhs, l_lhs - overlap + (b - 1) * v)
	for i in xrange(qn_half, qn_half + u, 1 << n):
		for j in xrange(i, i + overlap):
			a[j] ^= a[j - k]
	for j in xrange(qn_half + u, qn_half + u + b * v):
		a[j] ^= a[j - k]
	# Evaluate F_0 and D^{2^{n-1}}F_0 + F_1
	cdef int c_rhs = c - qn_half
	HermiteEvaluate(d, n - 1, qn_half, l_lhs, a)
	HermiteEvaluate(d, n - 1, c_rhs  , l_rhs, a[qn_half:])

cpdef void HermiteInterpolate(int d, int n, int c, int l, unsigned char[:] a):
	r"""
	Algorithm 4.
	"""
	if n == 0:
		Interpolate(c, l, a)
		return
	cdef int qn_half = 1 << (d + n - 1)
	if c <= qn_half:
		PrepareLeft(d, n, c, l, a)
		HermiteInterpolate(d, n - 1, c, min(l, qn_half), a)
		PrepareLeft(d, n, 0, l, a)
		return
	# Compute F_0 and D^{2^{n-1}}F_0 + F_1
	cdef int overlap = 1 << (n - 1)
	cdef int k = qn_half - overlap
	cdef int l_rhs = max(l - qn_half, k)
	HermiteInterpolate(d, n - 1, qn_half, qn_half, a)
	PrepareRight(d, n, c, l, a)
	HermiteInterpolate(d, n - 1, c - qn_half, l_rhs, a[qn_half:])
	# Add D^{2^{n-1}}F_0 to D^{2^{n-1}}F_0 + F_1
	cdef int i, j
	cdef int l_1 = (l >> n) << n
	cdef int l_2 = l ^ l_1
	if l_2 == 0:
		l_2 = 1 << n
		l_1 -= l_2
	for i in xrange(qn_half, l_1, 1 << n):
		for j in xrange(i, i + overlap):
			a[j] ^= a[j - k]
	for j in xrange(l_1, l_1 + min(l_2, overlap)):
			a[j] ^= a[j - k]
	# Inverse Taylor expansion
	for i in xrange(overlap, l - k):
		a[i] ^= a[k + i]
