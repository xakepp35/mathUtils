#pragma once

#include <xmmintrin.h>

// i would like not to expose every #define from sse_mathfun, so i written separate header
#define USE_SSE2

__m128 log_ps(__m128 x);
__m128 exp_ps(__m128 x);
__m128 sin_ps(__m128 x);
__m128 cos_ps(__m128 x);
void sincos_ps(__m128 x, __m128 *s, __m128 *c);
