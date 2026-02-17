#pragma once

#include <cstdint>
#include <cassert>
#include <vector>
#include <memory>
#include "pdxearch/common.hpp"

namespace PDX {

template <Quantization q>
class IndexPDXIVF {};

template <>
class IndexPDXIVF<F32> {
public:
	using cluster_t = Cluster<F32>;

	const uint32_t num_dimensions {};
	const uint64_t total_num_embeddings {};
	const uint32_t num_clusters {};
	const uint32_t num_vertical_dimensions {};
	const uint32_t num_horizontal_dimensions {};
	std::vector<cluster_t> clusters;
	const bool is_normalized {};
	std::vector<float> centroids;

	IndexPDXIVF(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized)
	    : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings), num_clusters(num_clusters),
	      num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
	      num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
	      is_normalized(is_normalized) {
		clusters.reserve(num_clusters);
	}
};

template <>
class IndexPDXIVF<U8> {
public:
	using cluster_t = Cluster<U8>;

	const uint32_t num_dimensions {};
	const uint64_t total_num_embeddings {};
	const uint32_t num_clusters {};
	const uint32_t num_vertical_dimensions {};
	const uint32_t num_horizontal_dimensions {};
	std::vector<cluster_t> clusters;
	const bool is_normalized {};
	std::vector<float> centroids;

	const float quantization_scale = 1.0f;
	const float quantization_scale_squared = 1.0f;
	const float inverse_quantization_scale_squared = 1.0f;
	const float quantization_base = 0.0f;

	IndexPDXIVF(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized,
	            float quantization_scale, float quantization_base)
	    : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings), num_clusters(num_clusters),
	      num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
	      num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
	      is_normalized(is_normalized), quantization_scale(quantization_scale),
	      quantization_scale_squared(quantization_scale * quantization_scale),
	      inverse_quantization_scale_squared(1.0f / (quantization_scale * quantization_scale)),
	      quantization_base(quantization_base) {
		clusters.reserve(num_clusters);
	}
};

} // namespace PDX
