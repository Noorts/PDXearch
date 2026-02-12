#pragma once

#include <cstdint>
#include <cstdio>
#include "arm_neon.h"
#include "pdxearch/common.hpp"

namespace PDX {

template <DistanceMetric alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::F32> {
public:
	using DISTANCE_TYPE = DistanceType_t<F32>;
	using QUERY_TYPE = QuantizedVectorType_t<F32>;
	using DATA_TYPE = DataType_t<F32>;

	// Defer to the scalar kernel
	template <bool USE_DIMENSIONS_REORDER, bool SKIP_PRUNED>
	static void VerticalPruning(const QUERY_TYPE *__restrict query, const DATA_TYPE *__restrict data, size_t n_vectors,
	                            size_t total_vectors, size_t start_dimension, size_t end_dimension,
	                            DISTANCE_TYPE *distances_p, const uint32_t *pruning_positions,
	                            const uint32_t *indices_dimensions, const int32_t *dim_clip_value,
	                            const float *scaling_factors) {
		size_t dimensions_jump_factor = total_vectors;
		for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
			uint32_t true_dimension_idx = dimension_idx;
			if constexpr (USE_DIMENSIONS_REORDER) {
				true_dimension_idx = indices_dimensions[dimension_idx];
			}
			size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
			for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
				auto true_vector_idx = vector_idx;
				if constexpr (SKIP_PRUNED) {
					true_vector_idx = pruning_positions[vector_idx];
				}
				float to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
				distances_p[true_vector_idx] += to_multiply * to_multiply;
			}
		}
	}

	// Defer to the scalar kernel
	static void Vertical(const QUERY_TYPE *__restrict query, const DATA_TYPE *__restrict data, size_t start_dimension,
	                     size_t end_dimension, DISTANCE_TYPE *distances_p, const float *scaling_factors) {
		for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
			size_t dimension_idx = dim_idx;
			size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
			for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
				float to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
				distances_p[vector_idx] += to_multiply * to_multiply;
			}
		}
	}

	static DISTANCE_TYPE Horizontal(const QUERY_TYPE *__restrict vector1, const DATA_TYPE *__restrict vector2,
	                                size_t num_dimensions, const float *scaling_factors = nullptr) {
#if defined(__APPLE__)
		float distance = 0.0;
#pragma clang loop vectorize(enable)
		for (size_t i = 0; i < num_dimensions; ++i) {
			float diff = vector1[i] - vector2[i];
			distance += diff * diff;
		}
		return distance;
#else
		float32x4_t sum_vec = vdupq_n_f32(0);
		size_t i = 0;
		for (; i + 4 <= num_dimensions; i += 4) {
			float32x4_t a_vec = vld1q_f32(vector1 + i);
			float32x4_t b_vec = vld1q_f32(vector2 + i);
			float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
			sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
		}
		DISTANCE_TYPE distance = vaddvq_f32(sum_vec);
		for (; i < num_dimensions; ++i) {
			float diff = vector1[i] - vector2[i];
			distance += diff * diff;
		}
		return distance;
#endif
	};
};

} // namespace PDX
