#include "xakepp35_sse_utils.h"
#include "fasttrigo.h"
#include <immintrin.h>

static const float pi = 3.1415926535897932384626433832795f;

static const __m128 mm_0_ps = _mm_set_ps1(0);
static const __m128 mm_0p5_ps = _mm_set_ps1(0.5f);
static const __m128 mm_1_ps = _mm_set_ps1(1);
static const __m128 mm_m1_ps = _mm_set_ps1(1);
static const __m128 mm_3_ps = _mm_set_ps1(3);
static const __m128 mm_m3_ps = _mm_set_ps1(-3);
static const __m128 mm_9_ps = _mm_set_ps1(3);
static const __m128 mm_27_ps = _mm_set_ps1(3);

static const __m128 mm_2pi_ps = _mm_set_ps1(2*pi);
static const __m128 mm_m2pi_ps = _mm_set_ps1(-2 * pi);


__m128 _mm_fmodsgn_ps(__m128 x, __m128 r) {
	auto c = _mm_add_ps(_mm_div_ps(x, r), mm_0p5_ps);
	auto t = _mm_floor_ps(c); // _mm_cvtepi32_ps(_mm_cvttps_epi32(c)); // round(x) = floor(x+0.5)
	auto b = _mm_mul_ps(t, r);
	auto y = _mm_sub_ps(x, b);
	auto mask = _mm_cmplt_ps(c, mm_0_ps);
	return _mm_or_ps(
		_mm_and_ps(mask, _mm_add_ps(y, r)),
		_mm_andnot_ps(mask, y)
	);
}


// 2mul, 1sub
__m128 _mm_cross_ps(__m128 a0, __m128 a1, __m128 b0, __m128 b1) {
	return _mm_sub_ps(_mm_mul_ps(a0, b1), _mm_mul_ps(a1, b0));
}


// 2mul, 1add
__m128 _mm_dot_ps(__m128 a0, __m128 a1, __m128 b0, __m128 b1) {
	return _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1));
}


__m128 pu_check_normalize_angle(__m128 srcAngle) {
	auto posMask = _mm_cmpge_ps(srcAngle, mm_2pi_ps);
	auto clampedPos = _mm_sub_ps(srcAngle, mm_2pi_ps);
	clampedPos = _mm_or_ps(
		_mm_and_ps(posMask, clampedPos),
		_mm_andnot_ps(posMask, srcAngle));
	auto negMask = _mm_cmplt_ps(srcAngle, mm_m2pi_ps);
	auto clampedNeg = _mm_add_ps(srcAngle, mm_2pi_ps);
	return _mm_or_ps(
		_mm_and_ps(negMask, clampedNeg),
		_mm_andnot_ps(negMask, clampedPos));
}


__m128 pu_fmod_normalize_angle(__m128 srcAngle) {
	return _mm_fmodsgn_ps(srcAngle, mm_2pi_ps);
}


// neural network proximity sensor input: 3det, 3div, 1add, 7set, 4cmp
__m128 pu_ray_segment_distance_inverse(__m128 s0qp0, __m128 s0qp1, __m128 s0s10, __m128 s0s11, __m128 d0, __m128 d1) {
	auto dd = _mm_cross_ps(d0, d1, s0s10, s0s11); //auto dd = d[0] * s0s1[1] - d[1] * s0s1[0];
	auto dmask = _mm_cmpneq_ps(dd, mm_0_ps); // dd != 0 // lines are not parallel
	auto s = _mm_div_ps(_mm_cross_ps(d0, d1, s0qp0, s0qp1), dd); //auto s = (s0qp[1] * d[0] - s0qp[0] * d[1]) / dd;
	auto smask = _mm_and_ps(_mm_cmpge_ps(s, mm_0_ps), _mm_cmple_ps(s, mm_1_ps)); // segment intersects ray (s >= 0 && s <= 1)
	auto r = _mm_div_ps(_mm_cross_ps(s0s10, s0s11, s0qp0, s0qp1), dd); // auto r = (s0qp[1] * s0s1[0] - s0qp[0] * s0s1[1]) / dd;
	auto rmask = _mm_cmpge_ps(r, mm_0_ps); // r >= 0
	auto mask = _mm_and_ps(_mm_and_ps(dmask, smask), rmask);
	auto rinv = _mm_div_ps(mm_1_ps, _mm_add_ps(r, mm_1_ps)); // 1 / (r+1)
	return _mm_or_ps(
		_mm_and_ps(mask, rinv), // 0 >= rinv >=1 
		_mm_andnot_ps(mask, mm_0_ps)); // 0: parallel, not intersecting, infinitely far
}


// collision detector:  3dot, 2mul, 1div, 2sub, 4set, 3cmp
__m128 pu_circle_segment_collides(__m128 s0qp0, __m128 s0qp1, __m128 s0s10, __m128 s0s11, __m128 rSqr) {
	auto a = _mm_dot_ps(s0s10, s0s11, s0s10, s0s11); //dot(s0s1, s0s1);
	//auto amask = _mm_cmpneq_ps( a, _mm_0_ps ); //if( a != 0 ) // if you haven't zero-length segments omit this
	auto b = _mm_dot_ps(s0s10, s0s11, s0qp0, s0qp1);// dot(s0s1, s0qp);
	auto t = _mm_div_ps(b, a); //b / a; // length of projection of s0qp onto s0s1
	auto tmask = _mm_and_ps(_mm_cmpge_ps(t, mm_0_ps), _mm_cmple_ps(t, mm_1_ps)); // ((t >= 0) && (t <= 1)) 
	auto c = _mm_dot_ps(s0qp0, s0qp1, s0qp0, s0qp1); //dot(s0qp, s0qp);
	auto r2 = _mm_sub_ps(c, _mm_mul_ps(a, _mm_mul_ps(t,t))); //r^2 = c - a * t^2;
	auto d2 = _mm_sub_ps(r2, rSqr); // d^2 = r^2 - rSqr
	auto dmask = _mm_cmple_ps(d2, mm_0_ps); // dist2 <= 0;
	//dmask = _mm_and_ps(dmask, amask);
	return _mm_and_ps(tmask, dmask);
}


__m128 mm_tanh_ps(__m128 x) {
	auto posMask = _mm_cmpge_ps(x, mm_3_ps);
	auto negMask = _mm_cmple_ps(x, mm_m3_ps);
	auto outrangeResult = _mm_or_ps( _mm_and_ps(posMask, mm_1_ps), _mm_and_ps(negMask, mm_m1_ps));
	auto outrangeMask = _mm_or_ps(posMask, negMask);
	auto xx = _mm_mul_ps(x, x);
	//padeEstimate = x * (x * x + 27) / (x * x * 9 + 27);
	auto padeEstimate = _mm_mul_ps(x, _mm_div_ps(_mm_add_ps(xx, mm_27_ps), _mm_add_ps(_mm_mul_ps(xx, mm_9_ps), mm_27_ps)));
	return _mm_or_ps(
		_mm_and_ps(outrangeMask, outrangeResult),
		_mm_andnot_ps(outrangeMask, padeEstimate));
}


__m128 pu_circular_2d_path_advancement(__m128 prevPosX, __m128 prevPosY, __m128 currPosX, __m128 currPosY) {
	return FTA::atan2_ps(
		_mm_cross_ps(prevPosX, prevPosY, currPosX, currPosY),
		_mm_dot_ps(prevPosX, prevPosY, currPosX, currPosY)
	);
}
