#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from gf256 cimport *

cdef unsigned int log2(unsigned int n):
	r"""
	Ceiling of base two logarithm.
	"""
	cdef unsigned int[32] LOG = [
		 0,  1, 28,  2, 29, 14, 24, 3,
		30, 22, 20, 15, 25, 17,  4, 8, 
 		31, 27, 13, 23, 21, 19, 16, 7,
		26, 12, 18,  6, 11,  5, 10, 9
	]
	n -= 1
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n += 1
	cdef unsigned int l = LOG[(n * 0x077CB531U) >> 27]
	return l

cpdef void NewtonToLCH(int l, unsigned char[:] a):
	r"""
	Newton basis to LCH basis conversion.
	"""
	cdef int m, k, q, r, s, i, j
	m = log2(l) - 1
	for k in xrange(m):
		q = (l >> k) & 0x1fe
		r = q << k
		s = 1 << k
		for j in xrange(r, l - s):
			a[j] ^= mul(q, a[j + s])
		for i in xrange(2, q, 2):
			r = i << k
			for j in xrange(r, r + s):
				a[j] ^= mul(i, a[j + s])

cpdef void LCHToNewton(int l, unsigned char[:] a):
	r"""
	LCH basis to Newton basis conversion.
	"""
	cdef int m, k, q, r, s, i, j
	m = log2(l) - 2
	for k in xrange(m, -1, -1):
		q = (l >> k) & 0x1fe
		r = q << k
		s = 1 << k
		for j in xrange(r, l - s):
			a[j] ^= mul(q, a[j + s])
		for i in xrange(2, q, 2):
			r = i << k
			for j in xrange(r, r + s):
				a[j] ^= mul(i, a[j + s])

cpdef void LCHInterp(unsigned char shift, int c, int l, unsigned char[:] a):
	r"""
	Interpolation on the LCH basis.
	"""
	if l == 1:
		return
	cdef int k, K, l_, c_0, c_1, i
	k = log2(l) - 1
	K = 1 << k
	l_ = l - K
	c_0 = min(c, K)
	c_1 = c - c_0
	cdef unsigned char twiddle, right_shift, b
	twiddle = shift >> k
	for i in xrange(c_0, l_):
		a[i] ^= mul(twiddle, a[K + i])
	LCHInterp(shift, c_0, K, a[:K])
	if c_1 == 0:
		for i in xrange(min(l_, c_0)):
			a[i] ^= mul(twiddle, a[K + i])
		return
	for i in xrange(c_1, l_):
		b = mul(twiddle, a[K + i])
		a[K + i] ^= a[i]
		a[i] ^= b
	for i in xrange(l_, K):
		a[K + i] = a[i]
	right_shift = shift ^ K
	LCHInterp(right_shift, c_1, K, a[K:])
	for i in xrange(c_1):
		a[K + i] ^= a[i]
		a[i] ^= mul(twiddle, a[K + i])

cpdef void LCHEval(unsigned char shift, int c, int l, unsigned char[:] a):
	r"""
	Evaluation on the LCH basis.
	"""
	if l == 1 and c == 1:
		return
	cdef int k, K, l_0, l_1, c_0, c_1, i
	k = log2(max(l, c)) - 1
	K = 1 << k
	l_0 = min(l, K)
	l_1 = l - l_0
	c_0 = min(c, K)
	c_1 = c - c_0
	cdef unsigned char twiddle, right_shift 
	twiddle = shift >> k
	for i in xrange(l_1):
		a[i] ^= mul(twiddle, a[K + i])
	if c_1 > 0:
		for i in xrange(l_1):
			a[K + i] ^= a[i]
		for i in xrange(l_1, l_0):
			a[K + i] = a[i]
		right_shift = shift ^ K
		LCHEval(right_shift, c_1, l_0, a[K:])
	LCHEval(shift, c_0, l_0, a[:K])

cpdef void InplaceLCHEval(unsigned char shift, int c, int l, unsigned char[:] a):
	r"""
	In-place evaluation on the LCH basis.
	"""
	if l == 1 and c == 1:
		return
	cdef int k, K, l_0, l_1, c_0, c_1, t_0, t_1, i
	cdef unsigned char twiddle, right_shift
	k = log2(max(l, c)) - 1
	K = 1 << k
	l_0 = min(l, K)
	l_1 = l - l_0
	c_0 = min(c, K)
	c_1 = c - c_0
	twiddle = shift >> k
	for i in xrange(l_1):
		a[i] ^= mul(twiddle, a[K + i])
	if c_1 > 0:
		for i in xrange(l_1):
			a[K + i] ^= a[i]
		for i in xrange(l_1, min(l_0, c_1)):
			a[K + i] = a[i]
		t_0 = max(l_0, c_1)
		t_1 = max(l_1, c_1)
		right_shift = shift ^ K
		if t_0 > t_1:
			for i in xrange(l_1):
				a[i], a[K + i] = a[K + i], a[i]
			InplaceLCHEval(right_shift, c_1, l_0, a)
			for i in xrange(t_1):
				a[i], a[K + i] = a[K + i], a[i]
		else:
			InplaceLCHEval(right_shift, c_1, l_0, a[K:])
		for i in xrange(c_1, l_1):
			a[K + i] ^= a[i]
	InplaceLCHEval(shift, c_0, l_0, a)
	for i in xrange(c_0, l_1):
		a[i] ^= mul(twiddle, a[K + i])

cpdef void InplaceLCHInterp(unsigned char shift, int c, int l, unsigned char[:] a):
	r"""
	In-place interpolation on the LCH basis.
	"""
	if l == 1:
		return
	cdef int k, K, l_, c_0, c_1, i
	cdef unsigned char twiddle, right_shift
	k = log2(l) - 1
	K = 1 << k
	l_ = l - K
	c_0 = min(c, K)
	c_1 = c - c_0
	twiddle = shift >> k
	for i in xrange(c_0, l_):
		a[i] ^= mul(twiddle, a[K + i])
	InplaceLCHInterp(shift, c_0, K, a)
	if c_1 > 0:
		for i in xrange(c_1, l_):
			a[K + i] ^= a[i]
		right_shift = shift ^ K
		if K > l_:
			for i in xrange(l_):
				a[i], a[K + i] = a[K + i], a[i]
			InplaceLCHInterp(right_shift, c_1, K, a)
			for i in xrange(l_):
				a[i], a[K + i] = a[K + i], a[i]
		else:
			InplaceLCHInterp(right_shift, c_1, K, a[K:])
		for i in xrange(l_):
			a[K + i] ^= a[i]
	for i in xrange(l_):
		a[i] ^= mul(twiddle, a[K + i])

cpdef void TaylorExpansion(int t, int l, unsigned char[:] a):
	r"""
	Generalised Taylor expansion at $x^{2^t}-x$.
	"""
	cdef int i, j, k, o, q, r, s
	for k in xrange(log2(l), t, -1):
		r = 1 << (k - 1)
		s = r >> t
		o = r - s
		q = (l >> k) << k
		s -= 1
		q += s
		for i in xrange(s, q, r << 1):
			for j in xrange(i + r, i, -1):
				a[j] ^= a[j + o]
		for j in xrange(l - o - 1, q, -1):
			a[j] ^= a[j + o]

cpdef void InverseTaylorExpansion(int t, int l, unsigned char[:] a):
	r"""
	Inverse generalised Taylor expansion at $x^{2^t}-x$.
	"""
	cdef int i, j, k, o, q, r, s
	for k in xrange(t + 1, log2(l) + 1):
		r = 1 << (k - 1)
		s = r >> t
		o = r - s
		q = (l >> k) << k
		q += s	
		for i in xrange(s, q, r << 1):
			for j in xrange(i, i + r):
				a[j] ^= a[j + o]
		for j in xrange(q, l - o):
			a[j] ^= a[j + o]

cpdef void MonomialToLCH(int l, unsigned char[:] a):
	r"""
	Monomial basis to LCH basis conversion.
	"""
	if l <= 2:
		return
	if l == 3:
		a[1] ^= a[2]
		return
	if l == 4:
		a[2] ^= a[3]
		a[1] ^= a[2]
		return
	cdef int q, r, s, k
	# t = 2, k = 0
	if l > 16:
		TaylorExpansion(4, l, a)
	# t = 1, k = 0
	q = l & 0x1f0
	r = l & 0x00f
	for j in xrange(0, q, 16):
		TaylorExpansion(2, 16, a[j:])
	TaylorExpansion(2, r, a[q:])
	# t = 1, k = 2
	if l > 64:
		q >>= 4
		for j in xrange(r, 16):
			TaylorExpansion(2, q, a[j::16])
		q += 1
		for j in xrange(r):
			TaylorExpansion(2, q, a[j::16])
	# t = 0, k = 0
	q = l & 0x1fc
	for j in xrange(0, q, 4):
		TaylorExpansion(1, 4, a[j:])
	r = l & 0x003
	if r:
		TaylorExpansion(1, r, a[q:])
	# t = 0, k = 2
	if l > 8:
		q = l & 0x1f0
		for k in xrange(0, q, 16):
			for j in xrange(k, k + 4):
				TaylorExpansion(1, 4, a[j::4])
		r = l & 0x00f
		if r:
			r >>= 2
			s = l & 0x1f3
			for j in xrange(s, q ^ 4):
				TaylorExpansion(1, r, a[j::4])
			r += 1
			for j in xrange(q, s):
				TaylorExpansion(1, r, a[j::4])
	# t = 0, k = 4
	if l > 32:
		q = l & 0x1c0
		for k in xrange(0, q, 64):
			for j in xrange(k, k + 16):
				TaylorExpansion(1, 4, a[j::16])
		r = l & 0x03f
		if r:
			r >>= 4
			s = l & 0x1cf
			for j in xrange(s, q ^ 16):
				TaylorExpansion(1, r, a[j::16])
			r += 1
			for j in xrange(q, s):
				TaylorExpansion(1, r, a[j::16])
	# t = 0, k = 6
	if l > 128:
		r = l >> 6
		s = l & 0x03f
		for j in xrange(s, 64):
			TaylorExpansion(1, r, a[j::64])
		r += 1
		for j in xrange(s):
			TaylorExpansion(1, r, a[j::64])

cpdef void LCHToMonomial(int l, unsigned char[:] a):
	r"""
	Monomial basis to LCH basis conversion.
	"""
	if l <= 2:
		return
	if l == 3:
		a[1] ^= a[2]
		return
	if l == 4:
		a[1] ^= a[2]
		a[2] ^= a[3]
		return
	cdef int q, r, s, k
	# t = 0, k = 6
	if l > 128:
		r = l >> 6
		s = l & 0x03f
		for j in xrange(s, 64):
			InverseTaylorExpansion(1, r, a[j::64])
		r += 1
		for j in xrange(s):
			InverseTaylorExpansion(1, r, a[j::64])
	# t = 0, k = 4
	if l > 32:
		q = l & 0x1c0
		for k in xrange(0, q, 64):
			for j in xrange(k, k ^ 16):
				InverseTaylorExpansion(1, 4, a[j::16])
		r = (l & 0x030) >> 4
		if r:
			s = l & 0x1cf
			for j in xrange(s, q ^ 16):
				InverseTaylorExpansion(1, r, a[j::16])
			r += 1
			for j in xrange(q, s):
				InverseTaylorExpansion(1, r, a[j::16])
	# t = 0, k = 2
	if l > 8:
		q = l & 0x1f0
		for k in xrange(0, q, 16):
			for j in xrange(k, k ^ 4):
				InverseTaylorExpansion(1, 4, a[j::4])
		r = (l & 0x00c) >> 2
		if r:
			s = l & 0x1f3
			for j in xrange(s, q ^ 4):
				InverseTaylorExpansion(1, r, a[j::4])
			r += 1
			for j in xrange(q, s):
				InverseTaylorExpansion(1, r, a[j::4])
	# t = 0, k = 0
	q = l & 0x1fc
	for j in xrange(0, q, 4):
		InverseTaylorExpansion(1, 4, a[j:])
	r = l & 0x003
	if r:
		InverseTaylorExpansion(1, r, a[q:])
	# t = 1, k = 2
	q = l & 0x1f0
	r = l & 0x00f
	if l > 64:
		s = q >> 4
		for j in xrange(r, 16):
			InverseTaylorExpansion(2, s, a[j::16])
		s += 1
		for j in xrange(r):
			InverseTaylorExpansion(2, s, a[j::16])
	# t = 1, k = 0
	for j in xrange(0, q, 16):
		InverseTaylorExpansion(2, 16, a[j:])
	if r:
		InverseTaylorExpansion(2, r, a[q:])
	# t = 2, k = 0
	if l > 16:
		InverseTaylorExpansion(4, l, a)

cpdef void TaylorExpansionHighCoeffs(int t, int c, int l, unsigned char[:] a):
	r"""
	High coefficients of the generalised Taylor expansion at $x^{2^t}-x$.
	"""
	cdef int i, j, k, o, q, r, s, u
	for k in xrange(log2(l), t, -1):
		r = 1 << (k - 1)
		s = r >> t
		o = r - s
		q = (l >> k) << k
		u = ((c - s) >> k) << k
		if u == q:
			for j in xrange(l - o - 1, c - 1, -1):
				a[j] ^= a[j + o]
		else:
			s -= 1
			q += s
			u += s + r
			for j in xrange(u, c - 1, -1):
					a[j] ^= a[j + o]
			for i in xrange(u + r, q, r << 1):
				for j in xrange(i + r, i, -1):
					a[j] ^= a[j + o]
			for j in xrange(l - o - 1, q, -1):
				a[j] ^= a[j + o]

cpdef void MonomialToLCHHighCoeffs(int c, int l, unsigned char[:] a):
	r"""
	Monomial basis to LCH basis conversion.
	"""
	if l <= c + 1:
		return
	if l <= 2:
		return
	if l == 3 and c <= 1:
		a[1] ^= a[2]
		return
	if l == 4 and c <= 2:
		a[2] ^= a[3]
		if c <= 1:
			a[1] ^= a[2]
		return
	cdef int q, r, s, k, q_, r_, s_
	# t = 2, k = 0
	if l > 16:
		TaylorExpansionHighCoeffs(4, c, l, a)
	# t = 1, k = 0
	q = l & 0x1f0
	r = l & 0x00f
	q_ = c & 0x1f0
	r_ = c & 0x00f
	if q_ < q:
		TaylorExpansionHighCoeffs(2, r_, 16, a[q_:])
		for j in xrange(q_ + 16, q, 16):
			TaylorExpansion(2, 16, a[j:])
		if r:
			TaylorExpansion(2, r, a[q:])
	elif r:
		TaylorExpansionHighCoeffs(2, r_, r, a[q:])
	# t = 1, k = 2
	if l > 64:
		q >>= 4
		q_ >>= 4
		if q_ == q:
			q += 1
			for j in xrange(r_, r):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
		elif r_ <= r:
			for j in xrange(r, 16):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
			q += 1
			for j in xrange(r_, r):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
			q_ += 1
			for j in xrange(r_):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
		else:
			for j in xrange(r_, 16):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
			q_ += 1
			for j in xrange(r, r_):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
			q += 1
			for j in xrange(r):
				TaylorExpansionHighCoeffs(2, q_, q, a[j::16])
	# t = 0, k = 0
	q = l & 0x1fc
	r = l & 0x003
	q_ = c & 0x1fc
	r_ = c & 0x003
	if q_ < q:
		TaylorExpansionHighCoeffs(1, r_, 4, a[q_:])
		for j in xrange(q_ + 4, q, 4):
			TaylorExpansion(1, 4, a[j:])
		if r:
			TaylorExpansion(1, r, a[q:])
	elif r:
		TaylorExpansionHighCoeffs(1, r_, r, a[q:])
	# t = 0, k = 2
	if l > 8:
		q = l & 0x1f0
		r = (l & 0x00c) >> 2
		q_ = c & 0x1f0
		r_ = (c & 0x00c) >> 2
		s_ = c & 0x1f3
		if q_ < q:
			for j in xrange(q_, s_):
					TaylorExpansionHighCoeffs(1, r_ + 1, 4, a[j::4])
			for j in xrange(s_, q_ ^ 4):
					TaylorExpansionHighCoeffs(1, r_, 4, a[j::4])
			for k in xrange(q_ + 16, q, 16):
				for j in xrange(k, k ^ 4):
					TaylorExpansion(1, 4, a[j::4])
		if r:
			s = l & 0x1f3
			if q_ < q:
				for j in xrange(s, q ^ 4):
					TaylorExpansion(1, r, a[j::4])
				r += 1
				for j in xrange(q, s):
					TaylorExpansion(1, r, a[j::4])
			elif s_ <= s:
				for j in xrange(s, q ^ 4):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
				r += 1
				for j in xrange(s_, s):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
				r_ += 1
				for j in xrange(q, s_):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
			else:
				for j in xrange(s_, q ^ 4):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
				r_ += 1
				for j in xrange(s, s_):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
				r += 1
				for j in xrange(q, s):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::4])
	# t = 0, k = 4
	if l > 32:
		q = l & 0x1c0
		r = (l & 0x030) >> 4
		q_ = c & 0x1c0
		r_ = (c & 0x030) >> 4
		s_ = c & 0x1cf
		if q_ < q:
			for j in xrange(q_, s_):
					TaylorExpansionHighCoeffs(1, r_ + 1, 4, a[j::16])
			for j in xrange(s_, q_ ^ 16):
					TaylorExpansionHighCoeffs(1, r_, 4, a[j::16])
			for k in xrange(q_ + 64, q, 64):
				for j in xrange(k, k ^ 16):
					TaylorExpansion(1, 4, a[j::16])
		if r:
			s = l & 0x1cf
			if q_ < q:
				for j in xrange(s, q ^ 16):
					TaylorExpansion(1, r, a[j::16])
				r += 1
				for j in xrange(q, s):
					TaylorExpansion(1, r, a[j::16])
			elif s_ <= s:
				for j in xrange(s, q ^ 16):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
				r += 1
				for j in xrange(s_, s):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
				r_ += 1
				for j in xrange(q, s_):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
			else:
				for j in xrange(s_, q ^ 16):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
				r_ += 1
				for j in xrange(s, s_):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
				r += 1
				for j in xrange(q, s):
					TaylorExpansionHighCoeffs(1, r_, r, a[j::16])
	# t = 0, k = 6
	if l > 128:
		r = l >> 6
		s = l & 0x03f
		r_ = c >> 6
		s_ = c & 0x03f
		if s_ <= s:
			for j in xrange(s, 64):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
			r += 1
			for j in xrange(s_, s):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
			r_ += 1
			for j in xrange(s_):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
		else:
			for j in xrange(s_, 64):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
			r_ += 1
			for j in xrange(s, s_):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
			r += 1
			for j in xrange(s):
				TaylorExpansionHighCoeffs(1, r_, r, a[j::64])
