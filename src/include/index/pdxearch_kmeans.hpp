#pragma once

#include "duckdb/common/assert.hpp"
#include <vector>
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexFlat.h"

namespace duckdb {

struct KMeansResult {
	// Row-major buffer of all centroids (num_clusters * num_dimensions).
	std::unique_ptr<float[]> centroids;

	// Mapping from a centroid to its embeddings.
	//
	// The embeddings are represented as indices into the original `embeddings` array. The `embeddings` array was passed
	// as a parameter to the `ComputeKMeans` function.
	//
	// `assignments[0] -> [1, 3]` means that the 2nd and 4th embeddings in the `embeddings` array belong to the 0th
	// cluster/centroid.
	std::vector<std::vector<uint64_t>> assignments;

	KMeansResult(uint32_t num_dimensions, uint32_t num_clusters)
	    : centroids(std::make_unique<float[]>(static_cast<size_t>(num_clusters * num_dimensions))),
	      assignments(num_clusters) {
	}
};

// Compute centroids (clusters) and centroid-to-embedding assignments using FAISS.
[[nodiscard]] inline KMeansResult ComputeKMeans(const float *const embeddings, const uint64_t num_embeddings,
                                                const uint32_t num_dimensions, const uint32_t num_clusters) {
	D_ASSERT(num_embeddings >= 1);
	D_ASSERT(num_dimensions >= 1);
	D_ASSERT(num_clusters >= 1);

	faiss::IndexFlatL2 faiss_quantizer(num_dimensions);
	faiss::IndexIVFFlat faiss_index(&faiss_quantizer, num_dimensions, num_clusters);

	// Configure optional sampling
	constexpr double SAMPLING_RATIO = 1.0;
	static_assert(0.0 < SAMPLING_RATIO && SAMPLING_RATIO <= 1.0, "SAMPLING_RATIO must be in (0.0, 1.0].");
	if constexpr (SAMPLING_RATIO < 1.0) {
		const int max_embeddings_per_centroid =
		    static_cast<int>(std::ceil((SAMPLING_RATIO * static_cast<double>(num_embeddings)) / num_clusters));
		faiss_index.cp.max_points_per_centroid = max_embeddings_per_centroid;
	}

	// Compute centroids and assignments
	D_ASSERT(!faiss_index.is_trained);
	faiss_index.train(static_cast<int64_t>(num_embeddings), embeddings);
	faiss_index.add(static_cast<int64_t>(num_embeddings), embeddings);

	auto result = KMeansResult(num_dimensions, num_clusters);

	// Extract centroids
	faiss_quantizer.reconstruct_n(0, faiss_quantizer.ntotal, result.centroids.get());

	// Extract assignments
	for (uint64_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
		const size_t cluster_size = faiss_index.invlists->list_size(cluster_idx);
		D_ASSERT(0 < cluster_size);
		D_ASSERT(cluster_size <= PDX::MAX_EMBEDDINGS_PER_CLUSTER);
		result.assignments[cluster_idx].reserve(cluster_size);

		const faiss::idx_t *faiss_ids = faiss_index.invlists->get_ids(cluster_idx);

		for (uint64_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
			// embedding_idx indexes into the vectors and row_ids arrays. It is not necessarily equal to the row id.
			auto embedding_idx = static_cast<uint64_t>(faiss_ids[position_in_cluster]);
			result.assignments[cluster_idx].emplace_back(embedding_idx);
		}

		faiss_index.invlists->release_ids(cluster_idx, faiss_ids);
	}

	return result;
};

} // namespace duckdb
