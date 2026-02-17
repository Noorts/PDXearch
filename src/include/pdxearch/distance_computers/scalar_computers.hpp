#pragma once

#include <cstdint>
#include "pdxearch/common.hpp"

namespace PDX {

template <DistanceMetric alpha, Quantization q>
class ScalarComputer {};

template <>
class ScalarComputer<DistanceMetric::L2SQ, Quantization::F32> {
public:
	using DISTANCE_TYPE = DistanceType_t<F32>;
	using QUERY_TYPE = QuantizedEmbeddingType_t<F32>;
	using DATA_TYPE = DataType_t<F32>;

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
		DISTANCE_TYPE distance = 0.0;
#pragma clang loop vectorize(enable)
		for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
			DISTANCE_TYPE to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
			distance += to_multiply * to_multiply;
		}
		return distance;
	};
};

template <>
class ScalarComputer<DistanceMetric::L2SQ, Quantization::U8> {
public:
	using DISTANCE_TYPE = DistanceType_t<U8>;
	using QUERY_TYPE = QuantizedEmbeddingType_t<U8>;
	using DATA_TYPE = DataType_t<U8>;

	template <bool SKIP_PRUNED>
	static void Vertical(const QUERY_TYPE *__restrict query, const DATA_TYPE *__restrict data, size_t n_vectors,
	                     size_t total_vectors, size_t start_dimension, size_t end_dimension, DISTANCE_TYPE *distances_p,
	                     const uint32_t *pruning_positions = nullptr) {
		for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx += 4) {
			uint32_t dimension_idx = dim_idx;
			size_t offset_to_dimension_start = dimension_idx * total_vectors;
			for (size_t i = 0; i < n_vectors; ++i) {
				size_t vector_idx = i;
				if constexpr (SKIP_PRUNED) {
					vector_idx = pruning_positions[vector_idx];
				}
				int da = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
				int db = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
				int dc = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
				int dd = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
				distances_p[vector_idx] += (da * da) + (db * db) + (dc * dc) + (dd * dd);
			}
		}
	}

	static DISTANCE_TYPE Horizontal(const QUERY_TYPE *__restrict vector1, const DATA_TYPE *__restrict vector2,
	                                size_t num_dimensions) {
		DISTANCE_TYPE distance = 0;
		for (size_t i = 0; i < num_dimensions; ++i) {
			int diff = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
			distance += diff * diff;
		}
		return distance;
	};
};

} // namespace PDX
