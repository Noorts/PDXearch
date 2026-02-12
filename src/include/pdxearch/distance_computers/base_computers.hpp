#pragma once

#include "pdxearch/common.hpp"

#ifdef __ARM_NEON
#include "pdxearch/distance_computers/neon_computers.hpp"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "pdxearch/distance_computers/avx2_computers.hpp"
#endif

#ifdef __AVX512F__
#include "pdxearch/distance_computers/avx512_computers.hpp"
#endif

// Fallback to scalar computer.
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
#include "pdxearch/distance_computers/scalar_computers.hpp"
#endif

// TODO: Support SVE

namespace PDX {

template <DistanceFunction alpha, Quantization q>
class DistanceComputer {};

template <>
class DistanceComputer<L2, Quantization::F32> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
	using computer = ScalarComputer<L2, F32>;
#else
	using computer = SIMDComputer<L2, F32>;
#endif

public:
	constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
	constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
	constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
	constexpr static auto Vertical = computer::VerticalPruning<false, false>;

	constexpr static auto VerticalBlock = computer::Vertical;
	constexpr static auto Horizontal = computer::Horizontal;
};

}; // namespace PDX

