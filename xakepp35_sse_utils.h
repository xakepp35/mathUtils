#pragma once

#include <xmmintrin.h>

__m128 _mm_cross_ps(__m128 a0, __m128 a1, __m128 b0, __m128 b1);
__m128 _mm_dot_ps(__m128 a0, __m128 a1, __m128 b0, __m128 b1);

// normalizes angle from [-2*pi..2*pi) to [-pi..pi) range
__m128 pu_check_normalize_angle(__m128 srcAngle);

// normalizes finite angle of any magnitude to [-pi..pi] range. +pi is also could be a result, because of implementation technique used
__m128 pu_fmod_normalize_angle(__m128 srcAngle);

__m128 pu_ray_segment_distance_inverse(__m128 s0qp0, __m128 s0qp1, __m128 s0s10, __m128 s0s11, __m128 d0, __m128 d1);
__m128 pu_circle_segment_collides(__m128 s0qp0, __m128 s0qp1, __m128 s0s10, __m128 s0s11, __m128 rSqr);

__m128 _mm_tanh_ps(__m128 x);