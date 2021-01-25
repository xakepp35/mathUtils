#pragma once

#pragma once
#include <emmintrin.h> /* SSE2 */
#include <stdint.h>
#include <array>

// A small RNG just used to initialize the higher-quality generators below.
uint64_t splitmix64(uint64_t state);

// Mask off all but the mantissa bits. This is a bit wasteful of the entropy, but it's quick. 
// Then set the exponent to all 1s (biased exponent) to get ^0 power. This produces a uniform in the [1,2] range
__m128 random_bits_to_uniform_1_2(__m128i random_bits);
__m128 random_bits_to_uniform_0_1(__m128i random_bits);
__m128 random_bits_to_uniform_k_b(__m128i random_bits, __m128 k, __m128 b);


class generator_seed {
public:

	// uses externally provided entropy, rndSeed != 0
	generator_seed(uint64_t rndSeed = 1);

protected:

	uint64_t* get_seed_buffer();
	std::array< __m128i, 2 > _vState;
};

class xoroshiro :
	public generator_seed
{
public:

	xoroshiro();
	xoroshiro(uint64_t rndSeed);

	void jump();  // TODO: implement jump
	__m128i raw();
	void raw(__m128i* outBuffer, size_t len);
	__m128 normal(float k = 1.0f, float b = 0.0f);
	void normal(__m128* outBuffer, size_t len, float k = 1.0f, float b = 0.0f);
	void binomial(__m128* outBuffer, size_t len, const int N, const float p);

};


class xorshift128plus :
	public generator_seed
{
public:

	xorshift128plus();
	xorshift128plus(uint64_t rndSeed);

	void jump();  // TODO: implement jump
	__m128i raw();
	void raw(__m128i* outBuffer, size_t len);
	__m128 normal(float k = 1.0f, float b = 0.0f);
	void normal(__m128* outBuffer, size_t len, float k = 1.0f, float b = 0.0f);
	void binomial(__m128* outBuffer, size_t len, const int N, const float p);

};

/*
void xoroshiro_init(xor_rng_state*, uint64_t seed);
void xoroshiro(xor_rng_state* state, float* buf, size_t len);
void xoroshiro_binomial(xor_rng_state* state,


void xorshift128plus_init(xor_rng_state* state, uint64_t seed);
void xorshift128plus(xor_rng_state* state, float* buf, size_t len);
void xorshift128plus_binomial(xor_rng_state* state,
const int N, const float p,
float* output, const size_t len);*/