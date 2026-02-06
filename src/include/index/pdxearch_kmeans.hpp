#pragma once

#include "duckdb/common/assert.hpp"
#include <vector>
#include "pdxearch/common.hpp"
#include "superkmeans/superkmeans.h"

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

// Compute centroids (clusters) and centroid-to-embedding assignments using SuperKMeans.
[[nodiscard]] inline KMeansResult ComputeKMeans(const float *const embeddings, const uint64_t num_embeddings,
                                                const uint32_t num_dimensions, const uint32_t num_clusters,
                                                const PDX::DistanceMetric distance_metric) {
	D_ASSERT(num_embeddings >= 1);
	D_ASSERT(num_dimensions >= 1);
	D_ASSERT(num_clusters >= 1);

	auto result = KMeansResult(num_dimensions, num_clusters);

	// Compute centroids
	skmeans::SuperKMeansConfig config;
	config.sampling_fraction = 1.0f;
	config.angular = distance_metric == PDX::DistanceMetric::COSINE || distance_metric == PDX::DistanceMetric::IP;
	config.data_already_rotated = true;
	auto kmeans = skmeans::SuperKMeans(num_clusters, num_dimensions, config);
	std::vector<float> centroids = kmeans.Train(embeddings, num_embeddings);

	// Extract centroids
	std::memcpy(result.centroids.get(), centroids.data(), centroids.size() * sizeof(float));

	// Extract assignments
	// SuperKMeans returns assignment from vec_id (not row_id) to centroid_idx
	std::vector<uint32_t> assignments = kmeans.Assign(embeddings, centroids.data(), num_embeddings, num_clusters);
	// Convert into assignment from centroid_idx to vec_id (not row_id)
	result.assignments.resize(num_clusters);
	for (uint64_t vec_id = 0; vec_id < num_embeddings; vec_id++) {
		result.assignments[assignments[vec_id]].emplace_back(vec_id);
	}

	return result;
};

} // namespace duckdb
