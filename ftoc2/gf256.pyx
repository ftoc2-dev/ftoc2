#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cdef unsigned char[256] LOG = [
	0x00, 0x00, 0xaa, 0x55, 0x88, 0x22, 0x11, 0x44,
	0xcc, 0xbb, 0x33, 0xee, 0x99, 0x77, 0xdd, 0x66,
	0xde, 0xed, 0x38, 0x83, 0x7b, 0xb7, 0xe0, 0x0e,
	0xbd, 0xdb, 0x07, 0x70, 0x1c, 0xc1, 0xf6, 0x6f,
	0x26, 0xb8, 0x62, 0x8b, 0xb1, 0x86, 0x1b, 0x68,
	0x98, 0xe2, 0x89, 0x2e, 0x1a, 0xc6, 0xa1, 0x6c,
	0x4c, 0x71, 0x17, 0xc4, 0xd0, 0x36, 0x63, 0x0d,
	0xd8, 0x43, 0x34, 0x8d, 0x31, 0xc5, 0x5c, 0x13,
	0x8f, 0x96, 0xf7, 0xc0, 0xf8, 0x69, 0x7f, 0x0c,
	0x4a, 0x67, 0x46, 0x40, 0xa4, 0x76, 0x64, 0x04,
	0x3e, 0x5a, 0xdf, 0x03, 0xa5, 0xe3, 0x30, 0xfd,
	0x19, 0x01, 0x29, 0x9d, 0x10, 0x91, 0xd9, 0x92,
	0x1f, 0x2d, 0x81, 0xef, 0x18, 0xfe, 0xf1, 0xd2,
	0x08, 0xc8, 0x49, 0xec, 0x94, 0xce, 0x80, 0x8c,
	0x25, 0xb3, 0x20, 0x23, 0x32, 0x02, 0x3b, 0x52,
	0x7c, 0xb4, 0x06, 0xbf, 0xfb, 0x60, 0x4b, 0xc7,
	0x05, 0x8a, 0xe8, 0xad, 0x3d, 0xba, 0x84, 0x3c,
	0x50, 0xa8, 0x8e, 0xda, 0xd3, 0xab, 0x48, 0xc3,
	0x97, 0xb2, 0xf3, 0x73, 0xa9, 0x9c, 0xc2, 0x7d,
	0x2b, 0x79, 0x37, 0x3f, 0xc9, 0x9a, 0xd7, 0x2c,
	0x14, 0x2a, 0xa3, 0xb6, 0xea, 0xf4, 0xf0, 0x12,
	0x3a, 0x6b, 0x41, 0xa2, 0x0f, 0x21, 0xae, 0x4f,
	0x72, 0xa6, 0xf5, 0x0b, 0x5e, 0xca, 0xcf, 0xcd,
	0xb0, 0x5f, 0x6a, 0x27, 0xdc, 0xfc, 0xac, 0xe5,
	0x0a, 0x15, 0x5b, 0xd1, 0x78, 0x09, 0x7a, 0x75,
	0x87, 0x90, 0xa7, 0x57, 0xa0, 0x51, 0xb5, 0x1d,
	0x58, 0xaf, 0x93, 0x35, 0x56, 0xf2, 0x7e, 0x6e,
	0x2f, 0x65, 0xe6, 0xe7, 0xfa, 0x85, 0x53, 0x39,
	0xcb, 0x59, 0xb9, 0xf9, 0x61, 0xbe, 0x4e, 0xd4,
	0xe4, 0x4d, 0x16, 0xeb, 0x9b, 0x9f, 0xbc, 0x95,
	0x28, 0x54, 0x6d, 0x47, 0x24, 0xe1, 0xd5, 0xe9,
	0x9e, 0x5d, 0x1e, 0x42, 0x74, 0xd6, 0x45, 0x82
]

cdef unsigned char[255] EXP = [
	0x01, 0x59, 0x75, 0x53, 0x4f, 0x80, 0x7a, 0x1a,
	0x68, 0xc5, 0xc0, 0xb3, 0x47, 0x37, 0x17, 0xac,
	0x5c, 0x06, 0xa7, 0x3f, 0xa0, 0xc1, 0xea, 0x32,
	0x64, 0x58, 0x2c, 0x26, 0x1c, 0xcf, 0xfa, 0x60,
	0x72, 0xad, 0x05, 0x73, 0xf4, 0x70, 0x20, 0xbb,
	0xf0, 0x5a, 0xa1, 0x98, 0x9f, 0x61, 0x2b, 0xd8,
	0x56, 0x3c, 0x74, 0x0a, 0x3a, 0xd3, 0x35, 0x9a,
	0x12, 0xdf, 0xa8, 0x76, 0x87, 0x84, 0x50, 0x9b,
	0x4b, 0xaa, 0xfb, 0x39, 0x07, 0xfe, 0x4a, 0xf3,
	0x8e, 0x6a, 0x48, 0x7e, 0x30, 0xe9, 0xe6, 0xaf,
	0x88, 0xcd, 0x77, 0xde, 0xf1, 0x03, 0xd4, 0xcb,
	0xd0, 0xe1, 0x51, 0xc2, 0x3e, 0xf9, 0xb4, 0xb9,
	0x7d, 0xe4, 0x22, 0x36, 0x4e, 0xd9, 0x0f, 0x49,
	0x27, 0x45, 0xba, 0xa9, 0x2f, 0xf2, 0xd7, 0x1f,
	0x1b, 0x31, 0xb0, 0x93, 0xfc, 0xc7, 0x4d, 0x0d,
	0xc4, 0x99, 0xc6, 0x14, 0x78, 0x97, 0xd6, 0x46,
	0x6e, 0x62, 0xff, 0x13, 0x86, 0xdd, 0x25, 0xc8,
	0x04, 0x2a, 0x81, 0x23, 0x6f, 0x3b, 0x8a, 0x40,
	0xc9, 0x5d, 0x5f, 0xd2, 0x6c, 0xef, 0x41, 0x90,
	0x28, 0x0c, 0x9d, 0xec, 0x95, 0x5b, 0xf8, 0xed,
	0xcc, 0x2e, 0xab, 0xa2, 0x4c, 0x54, 0xb1, 0xca,
	0x89, 0x94, 0x02, 0x8d, 0xbe, 0x83, 0xae, 0xd1,
	0xb8, 0x24, 0x91, 0x71, 0x79, 0xce, 0xa3, 0x15,
	0x21, 0xe2, 0x85, 0x09, 0xee, 0x18, 0xe5, 0x7b,
	0x43, 0x1d, 0x96, 0x8f, 0x33, 0x3d, 0x2d, 0x7f,
	0x69, 0x9c, 0xb5, 0xe0, 0x08, 0xb7, 0x6d, 0xb6,
	0x34, 0xc3, 0x67, 0x8c, 0xe7, 0xf6, 0xfd, 0x9e,
	0x38, 0x5e, 0x8b, 0x19, 0xbc, 0x0e, 0x10, 0x52,
	0x16, 0xf5, 0x29, 0x55, 0xe8, 0xbf, 0xda, 0xdb,
	0x82, 0xf7, 0xa4, 0xeb, 0x6b, 0x11, 0x0b, 0x63,
	0xa6, 0x66, 0xd5, 0x92, 0xa5, 0xb2, 0x1e, 0x42,
	0x44, 0xe3, 0xdc, 0x7c, 0xbd, 0x57, 0x65
]

cdef unsigned char mul(unsigned char x, unsigned char y):
	r"""
	Returns the product of two elements x and y in F_256.
	"""
	cdef int t
	if x == 0 or y == 0:
		return 0
	t = LOG[x] + LOG[y]
	if t >= 255:
		t -= 255
	return EXP[t]
