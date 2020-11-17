from ftoc2.hermite import *
import numpy as np
import unittest

COEFFICIENTS = np.load('ftoc2/tests/lch.npy')
EVALUATIONS  = np.load('ftoc2/tests/evaluations.npy')
ZEROS        = np.zeros(256, dtype=np.uint8)

class TestHermite(unittest.TestCase):
	r""" Test hermite """

	def test_HermiteEvaluate(self):
		r""" Test Hermite evaluate """
		for l in xrange(1, 257):
			for c in xrange(1, 257):
				f = np.concatenate((COEFFICIENTS[:l], ZEROS[l:]))
				HermiteEvaluate(2, 6, c, l, f)
				self.assertTrue((f[:c] == EVALUATIONS[l][:c]).all())

	def test_HermiteInterpolate(self):
		r""" Test Hermite interpolate """
		for l in xrange(1, 257):
			for c in xrange(1, l + 1):
				f = np.concatenate((EVALUATIONS[l][:c], COEFFICIENTS[c:l], ZEROS[l:]))
				HermiteInterpolate(2, 6, c, l, f)
				self.assertTrue((f[:c] == COEFFICIENTS[:c]).all())
