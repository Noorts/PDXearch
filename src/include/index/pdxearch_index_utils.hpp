#pragma once

#include <Eigen/Dense>
#include <random>

#include "duckdb/storage/storage_info.hpp"
#include "pdxearch/common.hpp"
#include "pdxearch/index_base/pdx_ivf.hpp"
#include "pdxearch/quantizers/global.hpp"
#include "pdxearch/pruners/adsampling.hpp"

namespace duckdb {

[[nodiscard]] inline constexpr idx_t GetRowGroupId(const row_t row_id) {
	return row_id / DEFAULT_ROW_GROUP_SIZE;
}

[[nodiscard]] inline constexpr uint32_t ComputeNumberOfClusters(const uint32_t num_embeddings) {
	// Based on:
	// https://github.com/cwida/PDX/blob/91618e01e574e594e27c71abfe3b1d5094657d53/benchmarks/python_scripts/setup_core_index.py#L17-L22

	if (num_embeddings < 500000) {
		return std::ceil(2 * std::sqrt(num_embeddings));
	} else if (num_embeddings < 2500000) {
		return std::ceil(4 * std::sqrt(num_embeddings));
	} else {
		return std::ceil(8 * std::sqrt(num_embeddings));
	}
}

// Generate a rotation matrix suitable for PDXearch's ADSampling pruning algorithm.
//
// Based on https://github.com/cwida/PDX/blob/main/python/pdxearch/preprocessors.py#L39
[[nodiscard]] inline unique_ptr<float[]> GenerateRandomRotationMatrix(const size_t num_dimensions, const int32_t seed) {
	auto rotation_matrix = make_uniq_array<float>(num_dimensions * num_dimensions);

	// TODO: Confirm seed handling is sound.
	std::mt19937 gen(seed);
	std::normal_distribution<float> normal_dist;

	Eigen::MatrixXf random_matrix {
	    Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_dimensions), static_cast<Eigen::Index>(num_dimensions))};
	for (idx_t i = 0; i < num_dimensions; ++i) {
		for (idx_t j = 0; j < num_dimensions; ++j) {
			random_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = normal_dist(gen);
		}
	}

	const Eigen::HouseholderQR<Eigen::MatrixXf> qr {random_matrix};
	const Eigen::MatrixXf transformation_matrix {qr.householderQ()};

	for (idx_t i = 0; i < num_dimensions; ++i) {
		for (idx_t j = 0; j < num_dimensions; ++j) {
			rotation_matrix[i * num_dimensions + j] =
			    transformation_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
		}
	}

	return rotation_matrix;
}

// Store the embeddings into this cluster's preallocated buffers in the transposed PDX layout.
//
// See the README of the following for a description of the PDX layout:
// https://github.com/cwida/pdx
template <PDX::Quantization q, typename T>
inline void StoreClusterEmbeddings(typename PDX::IndexPDXIVF<q>::CLUSTER_TYPE &cluster,
                                   const PDX::IndexPDXIVF<q> &index, const T *embeddings, const size_t num_embeddings);

template <>
inline void
StoreClusterEmbeddings<PDX::Quantization::F32, float>(PDX::IndexPDXIVF<PDX::Quantization::F32>::CLUSTER_TYPE &cluster,
                                                      const PDX::IndexPDXIVF<PDX::Quantization::F32> &index,
                                                      const float *const embeddings, const size_t num_embeddings) {
	// Store the cluster's data using the transposed PDX layout for float32 as described in:
	// https://github.com/cwida/pdx?tab=readme-ov-file#the-data-layout

	// Vertical dimensions.
	for (size_t dim = 0; dim < index.num_vertical_dimensions; ++dim) {
		for (size_t embedding = 0; embedding < num_embeddings; embedding++) {
			cluster.data[dim * num_embeddings + embedding] = embeddings[(embedding * index.num_dimensions) + dim];
		}
	}

	// Horizontal dimensions decomposed every 64 dimensions.
	constexpr size_t HORIZONTAL_SUBVECTOR_LENGTH = 64;
	size_t current_horizontal_offset = index.num_vertical_dimensions * num_embeddings;

	for (size_t dim_group = 0; dim_group < index.num_horizontal_dimensions; dim_group += HORIZONTAL_SUBVECTOR_LENGTH) {
		size_t group_size = std::min(HORIZONTAL_SUBVECTOR_LENGTH, index.num_horizontal_dimensions - dim_group);
		size_t actual_dim = index.num_vertical_dimensions + dim_group;

		for (size_t embedding = 0; embedding < num_embeddings; embedding++) {
			memcpy(cluster.data + current_horizontal_offset + embedding * group_size,
			       embeddings + ((embedding * index.num_dimensions) + actual_dim), group_size * sizeof(float));
		}
		current_horizontal_offset += num_embeddings * group_size;
	}
}

class EmbeddingPreprocessor {
private:
	// For rotation matrix multiplication.
	PDX::ADSamplingPruner<PDX::F32> pruner;
	// For normalization.
	PDX::Quantizer quantizer;
	const size_t num_dimensions;

public:
	explicit EmbeddingPreprocessor(const size_t num_dimensions, const float *const rotation_matrix)
	    : pruner(num_dimensions, rotation_matrix), quantizer(num_dimensions), num_dimensions(num_dimensions) {
	}

	// Warning: modifies the input_embedding.
	void PreprocessEmbedding(float *const input_embedding, float *const output_embedding, const bool normalize) const {
		// In-place normalization.
		if (normalize) {
			quantizer.NormalizeQuery(input_embedding, input_embedding);
		}
		pruner.PreprocessQuery(input_embedding, output_embedding);
	}

	// Warning: modifies the input_embeddings.
	void PreprocessEmbeddings(float *const input_embeddings, float *const output_embeddings,
	                          const size_t num_embeddings, const bool normalize) const {
		// In-place normalization.
		if (normalize) {
			for (size_t i = 0; i < num_embeddings; i++) {
				quantizer.NormalizeQuery(input_embeddings + i * num_dimensions, input_embeddings + i * num_dimensions);
			}
		}
		pruner.PreprocessEmbeddings(input_embeddings, output_embeddings, num_embeddings);
	}
};

[[nodiscard]] inline constexpr bool DistanceMetricRequiresNormalization(const PDX::DistanceMetric distance_metric) {
	return distance_metric == PDX::DistanceMetric::COSINE || distance_metric == PDX::DistanceMetric::IP;
}

} // namespace duckdb
