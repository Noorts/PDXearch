#pragma once

#include <cstdint>
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

	template <bool SKIP_PRUNED>
	static void Vertical(const QUERY_TYPE *__restrict query, const DATA_TYPE *__restrict data, size_t n_vectors,
	                     size_t total_vectors, size_t start_dimension, size_t end_dimension, DISTANCE_TYPE *distances_p,
	                     const uint32_t *pruning_positions) {
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
#if defined(__APPLE__)
		DISTANCE_TYPE distance = 0.0;
#pragma clang loop vectorize(enable)
		for (size_t i = 0; i < num_dimensions; ++i) {
			DISTANCE_TYPE diff = vector1[i] - vector2[i];
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
			DISTANCE_TYPE diff = vector1[i] - vector2[i];
			distance += diff * diff;
		}
		return distance;
#endif
	};
};


template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::U8>{
public:
    using DISTANCE_TYPE = DistanceType_t<U8>;
    using QUERY_TYPE = QuantizedVectorType_t<U8>;
    using DATA_TYPE = DataType_t<U8>;

    template<bool SKIP_PRUNED>
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t n_vectors,
            size_t total_vectors,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const uint32_t * pruning_positions = nullptr
    ){
        // TODO: Handle tail in dimension length, for now im not going to worry on that as all the datasets are divisible by 4
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED){
                const uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
                const uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
                for (; i + 4 <= n_vectors; i+=4) {
                    // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
                    uint32x4_t res = vld1q_u32(&distances_p[i]);
                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
                    uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                    vst1q_u32(&distances_p[i], vdotq_u32(res, diff_u8, diff_u8));
                }
            }
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED){
                    vector_idx = pruning_positions[vector_idx];
                }
                // L2
                int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                                           (to_multiply_b * to_multiply_b) +
                                           (to_multiply_c * to_multiply_c) +
                                           (to_multiply_d * to_multiply_d);

            }
        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions
    ){
        uint32x4_t sum_vec = vdupq_n_u32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
            sum_vec = vdotq_u32(sum_vec, d_vec, d_vec);
        }
        DISTANCE_TYPE distance = vaddvq_u32(sum_vec);
        for (; i < num_dimensions; ++i) {
            int n = (int)vector1[i] - vector2[i];
            distance += n * n;
        }
        return distance;
    };
};


} // namespace PDX
