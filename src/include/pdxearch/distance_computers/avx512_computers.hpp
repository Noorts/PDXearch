#pragma once

#include <cstdint>
#include <cassert>
#include <immintrin.h>
#include "pdxearch/common.hpp"
#include "pdxearch/distance_computers/scalar_computers.hpp"

namespace PDX {

template <DistanceMetric alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::F32> {
public:
	using DISTANCE_TYPE = DistanceType_t<F32>;
	using QUERY_TYPE = QuantizedVectorType_t<F32>;
	using DATA_TYPE = DataType_t<F32>;
	using scalar_computer = ScalarComputer<DistanceMetric::L2SQ, Quantization::F32>;

	template <bool SKIP_PRUNED>
	static void Vertical(const QUERY_TYPE *__restrict query, const DATA_TYPE *__restrict data, size_t n_vectors,
	                     size_t total_vectors, size_t start_dimension, size_t end_dimension, DISTANCE_TYPE *distances_p,
	                     const uint32_t *pruning_positions = nullptr) {
		size_t dimensions_jump_factor = total_vectors;
		for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
			size_t offset_to_dimension_start = dimension_idx * dimensions_jump_factor;
			for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
				auto true_vector_idx = vector_idx;
				if constexpr (SKIP_PRUNED) {
					true_vector_idx = pruning_positions[vector_idx];
				}
				DISTANCE_TYPE to_multiply = query[dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
				distances_p[true_vector_idx] += to_multiply * to_multiply;
			}
		}
	}

	static DISTANCE_TYPE Horizontal(const QUERY_TYPE *__restrict vector1, const DATA_TYPE *__restrict vector2,
	                                size_t num_dimensions) {
		__m512 d2_vec = _mm512_setzero();
		__m512 a_vec, b_vec;
	simsimd_l2sq_f32_skylake_cycle:
		if (num_dimensions < 16) {
			__mmask16 mask = static_cast<__mmask16>(_bzhi_u32(0xFFFFFFFF, num_dimensions));
			a_vec = _mm512_maskz_loadu_ps(mask, vector1);
			b_vec = _mm512_maskz_loadu_ps(mask, vector2);
			num_dimensions = 0;
		} else {
			a_vec = _mm512_loadu_ps(vector1);
			b_vec = _mm512_loadu_ps(vector2);
			vector1 += 16, vector2 += 16, num_dimensions -= 16;
		}
		__m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
		d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
		if (num_dimensions) {
			goto simsimd_l2sq_f32_skylake_cycle;
		}

		// _simsimd_reduce_f32x16_skylake
		__m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
		__m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
		r = _mm_hadd_ps(r, r);
		return _mm_cvtss_f32(_mm_hadd_ps(r, r));
	};
};

} // namespace PDX
