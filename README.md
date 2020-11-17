# Overview

ftoc2 implements various fast transforms based on the algorithms of [4, 6, 5, 3],
allowing for rapid conversion between several polynomial bases associated to the
subspaces of a characteristic two finite field. The transforms are specialised
to F_256, the finite field of order 256, and take advantage of the existence of
Cantor bases in this field in order to simplify the algorithms and improve their
complexity. The transforms are used to implement the algorithms of [2], providing
fast algorithms for solving certain Hermite interpolation and evaluation problems
over F_256. In-turn, these algorithms together with some of the transforms are
used to implement the fast systematic encoding algorithm of [1] for multiplicity
codes over F_256.

The following is a short description of each module in ftoc2:

 - gf256: Arithmetic for F_256, the finite field of order 256.
 - transforms: Change of basis algorithms for polynomials over F_256, based on
   the algorithms of [4, 6, 5, 3].
 - hermite: Hermite interpolation and evaluation over F_256, based on the
   algorithms of [2].
 - encode: Systematic encoding of multiplicity codes over F_256, based on the
   algorithms of [1].

Each module extends the previous one, and thus is dependent on those before it.
The tests double as examples of how to use the code.

# Installation

To build:

	python setup.py build_ext --inplace

To run tests:

	python setup.py test

To build documentation, compile docs/doc.tex.

# References

1. Nicholas Coxon, "Fast systematic encoding of multiplicity codes",
   J. Symbolic Comput. 94 (2019), 234–254.
2. Nicholas Coxon, "Fast Hermite interpolation and evaluation over finite
   fields of characteristic two", J. Symbolic Comput. 98 (2020), 270–283.
3. Nicholas Coxon, "Fast transforms over finite fields of characteristic two",
   J. Symbolic Comput., to appear.
4. Shuhong Gao and Todd Mateer, "Additive fast Fourier transforms over finite
   fields", IEEE Trans. Inform. Theory, vol. 56 , no. 12, 2010, pp. 6265-6272.
5. Sian-Jheng Lin, Tareq Y. Al-Naffouri, Yunghsiang S. Han, and Wei-Ho	Chung,
   "Novel polynomial basis with fast Fourier transform and its application to
   Reed-Solomon erasure codes", IEEE Trans. Inform. Theory, vol. 62, no. 11,
   2016, pp. 6284--6299.
6. Sian-Jheng Lin, Wei-Ho Chung, and Yunghsiang S. Han, "Novel polynomial basis
   and its application to Reed-Solomon erasure codes", 55th Annual IEEE
   Symposium on Foundations of Computer Science-FOCS 2014, IEEE Computer Soc.,
   Los Alamitos, CA, 2014, pp. 316--325.