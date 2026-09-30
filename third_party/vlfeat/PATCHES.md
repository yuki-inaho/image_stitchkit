# VLFeat SIFT subset

Source: user-provided GPU-accelerated-Natural-Image-Stitching-with-Global-Similarity-Prior-master.zip,
`vlfeat-0.9.20`. Only SIFT, scalar image primitives, and their required headers are retained.
The BSD license is in COPYING. The SIFT algorithm is retained; the convolution boundary traversal is locally repaired as described below.

Local changes: host.h recognizes VL_STATIC on MSVC and compiler-provided little-endian
macros (including AArch64). The legacy `snprintf`/`isnan` redefinitions are applied only
on MSVC before VS2015 (`_MSC_VER < 1900`); modern MSVC declares both in its headers, so
defining them caused `C1189` in `<stdio.h>`. Scalar paths disable SSE2, AVX, OpenMP, and
VLFeat's threading runtime. runtime.c provides the four allocator functions actually used
by this subset. Allocation failure in this legacy C dependency terminates the process;
ordinary input, solver, file I/O, model, and GPU errors use C++ exceptions. SIFT calls are
serialized to protect VLFeat's process-global exponential lookup initialization.

UBSan found undefined pointer arithmetic in upstream `vl_imconvcol_vf` / `vd`:
negative filter offsets were mixed with unsigned strides, and pointers were formed
before the source/filter arrays. The scalar convolution traversal now uses clamped
integer indices and direct destination indexing. Filter tap order, zero/continuity
padding, transpose and subsampling semantics are retained. The sanitizer remains
enabled for this dependency; the diagnostic is not suppressed.
