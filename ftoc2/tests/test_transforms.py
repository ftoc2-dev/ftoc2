from ftoc2.transforms import *
from copy import deepcopy
import numpy as np
import unittest

shift = 0xaa
LCH      = np.load('ftoc2/tests/lch.npy')
NEWTON   = np.load('ftoc2/tests/newton.npy')
LAGRANGE = np.load('ftoc2/tests/lagrange.npy')
MONOMIAL = np.load('ftoc2/tests/monomial.npy')
TAYLOR   = np.load('ftoc2/tests/taylor.npy')

"""
The array LCH defines a family of polynomials: for l in {1,...,256}, define

	f_l = LCH[0] * X_0 + LCH[1] * X_1 + ... + LCH[l - 1] * X_{l - 1},

where X_0,X_1,...,X_255 are the LCH basis polynomials. The remaining arrays
are 2-dimensional and contain the coefficients of the f_l on one of the
various bases or their evaluations. Specifically, the following hold for l 
in {1,...,256}:

- NEWTON[l] is the coefficient vector of f_l on the Newon basis,
- MONOMIAL[l] is the coefficient vector of f_l on the monomial basis,
- Taylor[l] is the coefficient vector of the Taylor expansion of f_1 at x^2-x,
- LAGRANGE[l] is equal to [f_l(omega_0),f_l(omega_1),...,f_l(omega_255)].

"""

class TestTransforms(unittest.TestCase):
	r""" Test transforms """

	def test_NewtonToLCH(self):
		r""" Test Newton to LCH conversion """
		for l in xrange(1, 257):
			f = deepcopy(NEWTON[l][:l])
			NewtonToLCH(l, f)
			self.assertTrue((f == LCH[:l]).all())

	def test_LCHToNewton(self):
		r""" Test LCH to Newton conversion """
		for l in xrange(1, 257):
			f = deepcopy(LCH[:l])
			LCHToNewton(l, f)
			self.assertTrue((f == NEWTON[l][:l]).all())

	def test_LCHInterp(self):
		r""" Test interpolation on the LCH basis"""
		for l in xrange(1, 257):
			for c in xrange(1, l + 1):
				f = np.concatenate((LAGRANGE[l][:c], LCH[c:]))
				LCHInterp(shift, c, l, f)
				self.assertTrue((f[:c] == LCH[:c]).all())

	def test_LCHEval(self):
		r""" Test evaluation on the LCH basis"""
		for l in xrange(1, 257):
			for c in xrange(1, 257):
				f = deepcopy(LCH)
				LCHEval(shift, c, l, f)
				self.assertTrue((f[:c] == LAGRANGE[l][:c]).all())

	def test_InplaceLCHInterp(self):
		r""" Test in-place interpolation on the LCH basis"""
		for l in xrange(1, 257):
			for c in xrange(1, l + 1):
				f = np.concatenate((LAGRANGE[l][:c], LCH[c:l]))
				InplaceLCHInterp(shift, c, l, f)
				self.assertTrue((f == LCH[:l]).all())

	def test_InplaceLCHEval(self):
		r""" Test in-place evaluation on the LCH basis"""
		for l in xrange(1, 257):
			for c in xrange(1, 257):
				f = deepcopy(LCH)
				InplaceLCHEval(shift, c, l, f)
				self.assertTrue((f[:c] == LAGRANGE[l][:c]).all())
				self.assertTrue((f[c:l] == LCH[c:l]).all())

	def test_TaylorExpansion(self):
		r""" Test Taylor expansion """
		for l in xrange(1, 257):
			f = deepcopy(MONOMIAL[l][:l])
			TaylorExpansion(1, l, f)
			self.assertTrue((f == TAYLOR[l][:l]).all())

	def test_InverseTaylorExpansion(self):
		r""" Test inverse Taylor expansion """
		for l in xrange(1, 257):
			f = deepcopy(TAYLOR[l][:l])
			InverseTaylorExpansion(1, l, f)
			self.assertTrue((f == MONOMIAL[l][:l]).all())

	def test_MonomialToLCH(self):
		r""" Test Monomial to LCH conversion """
		for l in xrange(1, 257):
			f = deepcopy(MONOMIAL[l][:l])
			MonomialToLCH(l, f)
			self.assertTrue((f == LCH[:l]).all())

	def test_LCHToMonomial(self):
		r""" Test LCH to Monomial conversion """
		for l in xrange(1, 257):
			f = deepcopy(LCH[:l])
			LCHToMonomial(l, f)
			self.assertTrue((f == MONOMIAL[l][:l]).all())

	def test_MonomialToLCHHighCoeffs(self):
		r""" Test Monomial to LCH conversion (high coeffs) """
		for l in xrange(1, 257):
			for c in xrange(l + 1):
				f = deepcopy(MONOMIAL[l][:l])
				MonomialToLCHHighCoeffs(c, l, f)
				self.assertTrue((f[:c] == MONOMIAL[l][:c]).all())
				self.assertTrue((f[c:] == LCH[c:l]).all())	

	def test_TaylorExpansionHighCoefficients(self):
		r""" Test Taylor expansion (high coeffs) """
		for l in xrange(1, 257):
			for c in xrange(0, l + 1):
				f = deepcopy(MONOMIAL[l][:l])
				TaylorExpansionHighCoeffs(1, c, l, f)
				self.assertTrue((f[:c] == MONOMIAL[l][:c]).all())
				self.assertTrue((f[c:] == TAYLOR[l][c:l]).all())

	def test_MonomialToLCHHighCoeffs(self):
		r""" Test Monomial to LCH conversion (high coeffs) """
		for l in xrange(1, 257):
			for c in xrange(0, l + 1):
				f = deepcopy(MONOMIAL[l][:l])
				MonomialToLCHHighCoeffs(c, l, f)
				self.assertTrue((f[:c] == MONOMIAL[l][:c]).all())
				self.assertTrue((f[c:] == LCH[c:l]).all())

