/* Small static runtime for the vendored SIFT/imopv subset.
 * No global configuration, architecture-specific assembly, or DLL initialization.
 * VLFeat algorithms and headers retain their original BSD license. */
#include "vl/generic.h"
#include <stdlib.h>
#include <stdio.h>
void *vl_malloc(size_t n) { void *p = malloc(n); if (!p && n) { fputs("SIFT allocation failed\n", stderr); abort(); } return p; }
void *vl_realloc(void *p, size_t n) { void *q = realloc(p,n); if (!q && n) { fputs("SIFT allocation failed\n", stderr); abort(); } return q; }
void *vl_calloc(size_t n, size_t s) { void *p = calloc(n,s); if (!p && n && s) { fputs("SIFT allocation failed\n", stderr); abort(); } return p; }
void vl_free(void *p) { free(p); }
