import sys
import numpy
from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Build import cythonize

ExtensionModules = cythonize(
	[
		Extension("ftoc2.gf256"     , ["ftoc2/gf256.pyx"]     ),
		Extension("ftoc2.transforms", ["ftoc2/transforms.pyx"]),
		Extension("ftoc2.hermite"   , ["ftoc2/hermite.pyx"]   ),
		Extension(
			"ftoc2.encode",
			["ftoc2/encode.pyx"],
			include_dirs = [numpy.get_include()]
		),
	],
)

class RunTestsCommand(Command):
	description = 'Run all tests'
	user_options = []
	def initialize_options(self):
		pass
	def finalize_options(self):
		pass
	def run(self):
		from ftoc2 import tests
		import unittest
		unittest.main(tests, argv=sys.argv[:1], verbosity=2)

setup(
    name = "ftoc2",
    packages = ['ftoc2', 'ftoc2.tests'],
	cmdclass = {'test': RunTestsCommand},
    ext_modules =  ExtensionModules,
)

