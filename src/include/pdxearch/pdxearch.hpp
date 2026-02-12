#pragma once

#include <numeric>
#include <queue>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include "pdxearch/common.hpp"
#include "pdxearch/db_mock/predicate_evaluator.hpp"
#include "pdxearch/distance_computers/base_computers.hpp"
#include "pdxearch/quantizers/global.h"
#include "pdxearch/index_base/pdx_ivf.hpp"
#include "pdxearch/pruners/adsampling.hpp"

namespace PDX {

template <Quantization q = F32, class Index = IndexPDXIVF<q>, class Quantizer = Global8Quantizer<q>,
          DistanceMetric alpha = DistanceMetric::L2SQ, class Pruner = ADSamplingPruner<q>>
class PDXearch {
public:
	using DISTANCES_TYPE = DistanceType_t<q>;
	using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
	using INDEX_TYPE = Index;
	using CLUSTER_TYPE = Cluster<q>;
	using KNNCandidate_t = KNNCandidate<q>;
	using VectorComparator_t = VectorComparator<q>;

	Quantizer quantizer;
	Pruner &pruner;
	INDEX_TYPE &pdx_data;

	PDXearch(INDEX_TYPE &data_index, Pruner &pruner)
	    : quantizer(data_index.num_dimensions), pruner(pruner), pdx_data(data_index),
	      dim_clip_value(data_index.num_dimensions, 0), quantized_query_buf(data_index.num_dimensions) {
		cluster_indices_in_access_order.resize(pdx_data.num_clusters);
		cluster_offsets.resize(pdx_data.num_clusters);
		for (size_t i = 0; i < pdx_data.num_clusters; ++i) {
			cluster_offsets[i] = total_embeddings;
			total_embeddings += pdx_data.clusters[i].num_embeddings;
		}
	}

	void SetNProbe(size_t nprobe) {
		ivf_nprobe = nprobe;
	}

	[[nodiscard]] static std::vector<KNNCandidate_t>
	BuildResultSetFromHeap(uint32_t k,
	                       std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> &heap) {
		// Pop the initialization element from the heap, as it can't be part of the result.
		if (!heap.empty() && heap.top().distance == std::numeric_limits<float>::max()) {
			heap.pop();
		}

		int32_t result_set_size = std::min(heap.size(), static_cast<size_t>(k));
		std::vector<KNNCandidate_t> result;
		result.resize(result_set_size);
		for (int32_t i = result_set_size - 1; i >= 0; --i) {
			const KNNCandidate_t &embedding = heap.top();
			result[i].distance = embedding.distance;
			result[i].index = embedding.index;
			heap.pop();
		}
		return result;
	}

	std::vector<size_t> cluster_offsets;

protected:
	float selectivity_threshold = 0.80;
	size_t ivf_nprobe = 0;

	size_t total_embeddings {0};

	// Prioritized list of indices of the clusters to probe. E.g., [0, 2, 1].
	std::vector<uint32_t> cluster_indices_in_access_order;

	// Indexes into the cluster_indices_in_access_order list. This offset is
	// incremented by 1 after probing a cluster.
	uint64_t cluster_indices_in_access_order_offset {0};

	// Start: State for the current filtered search.
	uint32_t k = 0;
	QUANTIZED_VECTOR_TYPE *prepared_query = nullptr;
	// Predicate evaluator for this rowgroup.
	std::unique_ptr<PredicateEvaluator> predicate_evaluator;
	// End

	Heap<q> *best_k = nullptr;
	std::mutex *best_k_mutex = nullptr;

	// Per-search query buffers, filled by PrepareQuery in InitializeSearch (U8 path).
	// Used by Search/FilteredSearch and passed to Warmup/Prune.
	std::vector<int32_t> dim_clip_value;
	std::vector<QUANTIZED_VECTOR_TYPE> quantized_query_buf;

	template <Quantization Q = q>
	void ResetPruningDistances(size_t n_vectors, DistanceType_t<Q> *pruning_distances) {
		memset(static_cast<void *>(pruning_distances), 0, n_vectors * sizeof(DistanceType_t<Q>));
	}

	// The pruning threshold by default is the top of the heap
	template <Quantization Q = q>
	void
	GetPruningThreshold(uint32_t k,
	                    std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	                    DistanceType_t<Q> &pruning_threshold, uint32_t current_dimension_idx) {
		const std::lock_guard<std::mutex> lock(*best_k_mutex);
		pruning_threshold = pruner.template GetPruningThreshold<Q>(k, heap, current_dimension_idx);
	};

	template <Quantization Q = q>
	void EvaluatePruningPredicateScalar(uint32_t &n_pruned, size_t n_vectors, DistanceType_t<Q> *pruning_distances,
	                                    const DistanceType_t<Q> pruning_threshold) {
		for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
			n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
		}
	};

	template <Quantization Q = q>
	void EvaluatePruningPredicateOnPositionsArray(size_t n_vectors, size_t &n_vectors_not_pruned,
	                                              uint32_t *pruning_positions, DistanceType_t<Q> pruning_threshold,
	                                              DistanceType_t<Q> *pruning_distances) {
		n_vectors_not_pruned = 0;
		for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
			pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
			n_vectors_not_pruned += pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
		}
	};

	template <Quantization Q = q>
	void InitPositionsArray(size_t n_vectors, size_t &n_vectors_not_pruned, uint32_t *pruning_positions,
	                        DistanceType_t<Q> pruning_threshold, DistanceType_t<Q> *pruning_distances) {
		n_vectors_not_pruned = 0;
		for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
			pruning_positions[n_vectors_not_pruned] = vector_idx;
			n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
		}
	};

	template <Quantization Q = q>
	void InitPositionsArrayFromSelectionVector(size_t n_vectors, size_t &n_vectors_not_pruned,
	                                           uint32_t *pruning_positions, const uint8_t *selection_vector) {
		n_vectors_not_pruned = 0;
		for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
			pruning_positions[n_vectors_not_pruned] = vector_idx;
			n_vectors_not_pruned += selection_vector[vector_idx] == 1;
		}
	};

	template <Quantization Q = q>
	void MaskDistancesWithSelectionVector(size_t n_vectors, DistanceType_t<Q> *pruning_distances,
	                                      const uint8_t *selection_vector) {
		for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
			if (selection_vector[vector_idx] == 0) {
				// Why max()/2? To prevent overflow if distances are still added to these
				pruning_distances[vector_idx] = std::numeric_limits<DistanceType_t<Q>>::max() / 2;
			}
		}
	};

	static void GetClustersAccessOrderIVF(const float *__restrict query, const INDEX_TYPE &data, size_t nprobe,
	                                      std::vector<uint32_t> &clusters_indices) {
		std::vector<float> distances_to_centroids;
		distances_to_centroids.resize(data.num_clusters);
		for (size_t cluster_idx = 0; cluster_idx < data.num_clusters; cluster_idx++) {
			distances_to_centroids[cluster_idx] = DistanceComputer<DistanceMetric::L2SQ, F32>::Horizontal(
			    query, data.centroids.get() + cluster_idx * data.num_dimensions, data.num_dimensions, nullptr);
		}
		clusters_indices.resize(data.num_clusters);
		std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
		if (nprobe >= data.num_clusters) {
			std::sort(clusters_indices.begin(), clusters_indices.end(),
			          [&distances_to_centroids](size_t i1, size_t i2) {
				          return distances_to_centroids[i1] < distances_to_centroids[i2];
			          });
		} else {
			std::partial_sort(clusters_indices.begin(), clusters_indices.begin() + static_cast<int64_t>(nprobe),
			                  clusters_indices.end(), [&distances_to_centroids](size_t i1, size_t i2) {
				                  return distances_to_centroids[i1] < distances_to_centroids[i2];
			                  });
		}
	}

	// On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors is low
	template <Quantization Q = q, bool FILTERED = false>
	void Warmup(const QuantizedVectorType_t<Q> *__restrict query, const DataType_t<Q> *__restrict data,
	            const size_t n_vectors, uint32_t k, float tuples_threshold, uint32_t *pruning_positions,
	            DistanceType_t<Q> *pruning_distances, DistanceType_t<Q> &pruning_threshold,
	            std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	            uint32_t &current_dimension_idx, size_t &n_vectors_not_pruned, const int32_t *dim_clip_value,
	            uint32_t passing_tuples = 0, uint8_t *selection_vector = nullptr) {
		current_dimension_idx = 0;
		size_t cur_subgrouping_size_idx = 0;
		size_t tuples_needed_to_exit =
		    static_cast<size_t>(std::ceil(tuples_threshold * static_cast<float>(n_vectors)));
		ResetPruningDistances<Q>(n_vectors, pruning_distances);
		uint32_t n_tuples_to_prune = 0;
		if constexpr (FILTERED) {
			float selection_percentage = (static_cast<float>(passing_tuples) / static_cast<float>(n_vectors));
			MaskDistancesWithSelectionVector(n_vectors, pruning_distances, selection_vector);
			if (selection_percentage < 0.20) {
				// Go directly to the PRUNE phase for direct tuples access in the Horizontal block
				return;
			}
		}
		GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
		while (n_tuples_to_prune < tuples_needed_to_exit && current_dimension_idx < pdx_data.num_vertical_dimensions) {
			size_t last_dimension_to_fetch =
			    std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
			             pdx_data.num_vertical_dimensions);
			DistanceComputer<alpha, Q>::Vertical(query, data, n_vectors, n_vectors, current_dimension_idx,
			                                     last_dimension_to_fetch, pruning_distances, pruning_positions,
			                                     dim_clip_value);
			current_dimension_idx = last_dimension_to_fetch;
			cur_subgrouping_size_idx += 1;
			GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
			n_tuples_to_prune = 0;
			EvaluatePruningPredicateScalar<Q>(n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold);
		}
	}

	// We scan only the not-yet pruned vectors
	template <Quantization Q = q>
	void Prune(const QuantizedVectorType_t<Q> *__restrict query, const DataType_t<Q> *__restrict data,
	           const size_t n_vectors, uint32_t k, uint32_t *pruning_positions, DistanceType_t<Q> *pruning_distances,
	           DistanceType_t<Q> &pruning_threshold,
	           std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	           uint32_t &current_dimension_idx, size_t &n_vectors_not_pruned, const int32_t *dim_clip_value) {
		GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
		InitPositionsArray<Q>(n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances);
		size_t cur_n_vectors_not_pruned = 0;
		size_t current_vertical_dimension = current_dimension_idx;
		size_t current_horizontal_dimension = 0;
		while (pdx_data.num_horizontal_dimensions && n_vectors_not_pruned &&
		       current_horizontal_dimension < pdx_data.num_horizontal_dimensions) {
			cur_n_vectors_not_pruned = n_vectors_not_pruned;
			size_t offset_data =
			    (pdx_data.num_vertical_dimensions * n_vectors) + (current_horizontal_dimension * n_vectors);
			for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
				size_t v_idx = pruning_positions[vector_idx];
				size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
				__builtin_prefetch(data + data_pos, 0, 3);
			}
			size_t offset_query = pdx_data.num_vertical_dimensions + current_horizontal_dimension;
			for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
				size_t v_idx = pruning_positions[vector_idx];
				size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
				pruning_distances[v_idx] +=
				    DistanceComputer<alpha, Q>::Horizontal(query + offset_query, data + data_pos, H_DIM_SIZE, nullptr);
			}
			// end of clipping
			current_horizontal_dimension += H_DIM_SIZE;
			current_dimension_idx += H_DIM_SIZE;
			GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
			assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
			EvaluatePruningPredicateOnPositionsArray<Q>(cur_n_vectors_not_pruned, n_vectors_not_pruned,
			                                            pruning_positions, pruning_threshold, pruning_distances);
		}
		// GO THROUGH THE REST IN THE VERTICAL
		while (n_vectors_not_pruned && current_vertical_dimension < pdx_data.num_vertical_dimensions) {
			cur_n_vectors_not_pruned = n_vectors_not_pruned;
			size_t last_dimension_to_test_idx =
			    std::min(current_vertical_dimension + H_DIM_SIZE, static_cast<size_t>(pdx_data.num_vertical_dimensions));
			DistanceComputer<alpha, Q>::VerticalPruning(query, data, cur_n_vectors_not_pruned, n_vectors,
			                                            current_vertical_dimension, last_dimension_to_test_idx,
			                                            pruning_distances, pruning_positions,
			                                            dim_clip_value);
			current_dimension_idx = std::min(current_dimension_idx + H_DIM_SIZE, static_cast<size_t>(pdx_data.num_dimensions));
			current_vertical_dimension =
			    std::min(static_cast<uint32_t>(current_vertical_dimension + H_DIM_SIZE), pdx_data.num_vertical_dimensions);
			assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
			GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
			EvaluatePruningPredicateOnPositionsArray<Q>(cur_n_vectors_not_pruned, n_vectors_not_pruned,
			                                            pruning_positions, pruning_threshold, pruning_distances);
			if (current_dimension_idx == pdx_data.num_dimensions) {
				break;
			}
		}
	}

	template <bool IS_PRUNING = false, Quantization Q = q>
	void MergeIntoHeap(const uint32_t *vector_indices, size_t n_vectors, uint32_t k, const uint32_t *pruning_positions,
	                   DistanceType_t<Q> *pruning_distances, DistanceType_t<Q> *distances,
	                   std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap) {
		for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
			size_t index = position_idx;
			// DISTANCES_TYPE current_distance;
			float current_distance;
			if constexpr (IS_PRUNING) {
				index = pruning_positions[position_idx];
				current_distance = pruning_distances[index];
			} else {
				current_distance = distances[index];
			}
			if (heap.size() < k || current_distance < heap.top().distance) {
				KNNCandidate<Q> embedding {};
				embedding.distance = current_distance;
				embedding.index = vector_indices[index];
				if (heap.size() >= k) {
					heap.pop();
				}
				heap.push(embedding);
			}
		}
	}

public:
	/******************************************************************
	 * Search methods (for parallel implementation)
	 ******************************************************************/

	// Initialization that works for both the Search and FilteredSearch methods.
	void InitializeSearch(float *__restrict const preprocessed_query, const uint32_t k, Heap<q> &heap,
	                      std::mutex &heap_mutex, std::unique_ptr<PredicateEvaluator> predicate_evaluator = nullptr) {
		this->best_k = &heap;
		this->best_k_mutex = &heap_mutex;
		this->k = k;
		this->predicate_evaluator = std::move(predicate_evaluator);

		GetClustersAccessOrderIVF(preprocessed_query, pdx_data, pdx_data.num_clusters, cluster_indices_in_access_order);

		cluster_indices_in_access_order_offset = 0; // Reset cluster index offset for new search.
		if constexpr (q == U8) {
			quantizer.PrepareQuery(preprocessed_query, pdx_data.for_base, pdx_data.scale_factor, dim_clip_value.data(),
			                       quantized_query_buf.data());
			this->prepared_query = quantized_query_buf.data();
		} else {
			this->prepared_query = preprocessed_query;
		}
	}

	void Search(const size_t num_clusters_to_probe) {
		// Partial precondition check to ensure the state was reset / initialized.
		assert(best_k);
		assert(best_k_mutex);
		assert(k != 0);
		assert(prepared_query);

		alignas(64) DISTANCES_TYPE pruning_distances[MAX_EMBEDDINGS_PER_CLUSTER];
		alignas(64) uint32_t pruning_positions[MAX_EMBEDDINGS_PER_CLUSTER];

		const size_t end_idx = std::min<size_t>(num_clusters_to_probe, cluster_indices_in_access_order.size());
		for (size_t cluster_idx = 0; cluster_idx < end_idx; ++cluster_idx) {
			DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = cluster_indices_in_access_order[cluster_idx];
			const CLUSTER_TYPE &cluster = pdx_data.clusters[current_cluster_idx];

			Warmup(prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold, pruning_positions,
			       pruning_distances, pruning_threshold, *best_k, current_dimension_idx, n_vectors_not_pruned,
			       dim_clip_value.data());
			Prune(prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances,
			      pruning_threshold, *best_k, current_dimension_idx, n_vectors_not_pruned, dim_clip_value.data());
			if (n_vectors_not_pruned) {
				const std::lock_guard<std::mutex> lock(*best_k_mutex);
				MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances,
				                    nullptr, *best_k);
			}
		}
	}

	void FilteredSearch(const size_t num_clusters_to_probe) {
		// Partial precondition check to ensure the state was reset / initialized.
		assert(best_k);
		assert(best_k_mutex);
		assert(k != 0);
		assert(predicate_evaluator);
		assert(prepared_query);

		alignas(64) DISTANCES_TYPE pruning_distances[MAX_EMBEDDINGS_PER_CLUSTER];
		alignas(64) uint32_t pruning_positions[MAX_EMBEDDINGS_PER_CLUSTER];

		const size_t end_idx = std::min<size_t>(cluster_indices_in_access_order_offset + num_clusters_to_probe,
		                                        cluster_indices_in_access_order.size());
		for (; cluster_indices_in_access_order_offset < end_idx; ++cluster_indices_in_access_order_offset) {
			DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = cluster_indices_in_access_order[cluster_indices_in_access_order_offset];
			auto [selection_vector, passing_tuples] =
			    predicate_evaluator->GetSelectionVector(current_cluster_idx, cluster_offsets[current_cluster_idx]);
			if (passing_tuples == 0) {
				continue;
			}
			const CLUSTER_TYPE &cluster = pdx_data.clusters[current_cluster_idx];

			Warmup<q, true>(prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold,
			                pruning_positions, pruning_distances, pruning_threshold, *best_k, current_dimension_idx,
			                n_vectors_not_pruned, dim_clip_value.data(), passing_tuples, selection_vector);
			Prune(prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances,
			      pruning_threshold, *best_k, current_dimension_idx, n_vectors_not_pruned, dim_clip_value.data());
			if (n_vectors_not_pruned) {
				const std::lock_guard<std::mutex> lock(*best_k_mutex);
				MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances,
				                    nullptr, *best_k);
			}
		}
		assert(cluster_indices_in_access_order_offset <= cluster_indices_in_access_order.size());
	}

	/******************************************************************
	 * Search methods (for the global implementation)
	 ******************************************************************/

	// On the first bucket, we do a full scan (we do not prune vectors)
	template <Quantization Q = q>
	void Start(const QuantizedVectorType_t<Q> *__restrict query, const DataType_t<Q> *data, const size_t n_vectors,
	           uint32_t k, const uint32_t *vector_indices, uint32_t *pruning_positions,
	           DistanceType_t<Q> *pruning_distances,
	           std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	           const int32_t *dim_clip_value) {
		ResetPruningDistances<Q>(n_vectors, pruning_distances); // THIS
		DistanceComputer<alpha, Q>::Vertical(query, data, n_vectors, n_vectors, 0, pdx_data.num_vertical_dimensions,
		                                     pruning_distances, pruning_positions,
		                                     dim_clip_value);
		for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions;
		     horizontal_dimension += H_DIM_SIZE) {
			for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
				size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) + (horizontal_dimension * n_vectors) +
				                  (vector_idx * H_DIM_SIZE);
				pruning_distances[vector_idx] += DistanceComputer<alpha, Q>::Horizontal(
				    query + pdx_data.num_vertical_dimensions + horizontal_dimension, data + data_pos, H_DIM_SIZE,
				    nullptr);
			}
		}
		size_t max_possible_k =
		    std::min(static_cast<size_t>(k) - heap.size(), n_vectors); // Note: Start() should not be called if heap.size() >= k
		std::vector<size_t> indices_sorted;
		indices_sorted.resize(n_vectors);
		std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
		std::partial_sort(
		    indices_sorted.begin(), indices_sorted.begin() + static_cast<int64_t>(max_possible_k), indices_sorted.end(),
		    [pruning_distances](size_t i1, size_t i2) { return pruning_distances[i1] < pruning_distances[i2]; });
		// insert first k results into the heap
		for (size_t idx = 0; idx < max_possible_k; ++idx) {
			auto embedding = KNNCandidate<Q> {};
			size_t index = indices_sorted[idx];
			embedding.index = vector_indices[index];
			embedding.distance = pruning_distances[index];
			heap.push(embedding);
		}
	}

	// On the first bucket, we do a full scan (we do not prune vectors)
	template <Quantization Q = q>
	void FilteredStart(const QuantizedVectorType_t<Q> *__restrict query, const DataType_t<Q> *data,
	                   const size_t n_vectors, uint32_t k, const uint32_t *vector_indices, uint32_t *pruning_positions,
	                   DistanceType_t<Q> *pruning_distances,
	                   std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	                   const int32_t *dim_clip_value, uint8_t *selection_vector, uint32_t passing_tuples) {
		ResetPruningDistances<Q>(n_vectors, pruning_distances);
		size_t n_vectors_not_pruned = 0;
		DistanceType_t<Q> pruning_threshold = std::numeric_limits<DistanceType_t<Q>>::max();
		float selection_percentage = (static_cast<float>(passing_tuples) / static_cast<float>(n_vectors));
		InitPositionsArrayFromSelectionVector<Q>(n_vectors, n_vectors_not_pruned, pruning_positions, selection_vector);
		// Always start with horizontal block, regardless of selectivity
		for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions;
		     horizontal_dimension += H_DIM_SIZE) {
			size_t offset_data = (pdx_data.num_vertical_dimensions * n_vectors) + (horizontal_dimension * n_vectors);
			for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
				size_t v_idx = pruning_positions[vector_idx];
				size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
				pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
				    query + pdx_data.num_vertical_dimensions + horizontal_dimension, data + data_pos, H_DIM_SIZE,
				    nullptr);
			}
		}
		if (selection_percentage > 0.20) { // TODO: 0.20 comes from the `selectivity_threshold`
			// It is then faster to do the full scan (thanks to SIMD)
			DistanceComputer<alpha, Q>::Vertical(query, data, n_vectors, n_vectors, 0, pdx_data.num_vertical_dimensions,
			                                     pruning_distances, pruning_positions,
			                                     dim_clip_value);
		} else {
			// We access individual values
			DistanceComputer<alpha, Q>::VerticalPruning(
			    query, data, n_vectors_not_pruned, n_vectors, 0, pdx_data.num_vertical_dimensions, pruning_distances,
			    pruning_positions, dim_clip_value);
		}
		// TODO: Everything down from here is a bottleneck when selection % is ultra low
		size_t max_possible_k = std::min(static_cast<size_t>(k) - heap.size(), static_cast<size_t>(passing_tuples));
		MaskDistancesWithSelectionVector(n_vectors, pruning_distances, selection_vector);
		std::vector<size_t> indices_sorted;
		indices_sorted.resize(n_vectors);
		std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
		std::partial_sort(
		    indices_sorted.begin(), indices_sorted.begin() + static_cast<int64_t>(max_possible_k), indices_sorted.end(),
		    [pruning_distances](size_t i1, size_t i2) { return pruning_distances[i1] < pruning_distances[i2]; });
		// insert first k results into the heap
		for (size_t idx = 0; idx < max_possible_k; ++idx) {
			auto embedding = KNNCandidate<Q> {};
			size_t index = indices_sorted[idx];
			embedding.index = vector_indices[index];
			embedding.distance = pruning_distances[index];
			heap.push(embedding);
		}
	}

	std::vector<KNNCandidate_t> SearchGlobal(const float *__restrict const raw_query, const uint32_t k) {
		Heap<q> local_heap {};
		std::mutex local_mutex;
		best_k_mutex = &local_mutex;
		std::vector<float> query(pdx_data.num_dimensions);
		if (!pdx_data.is_normalized) {
			pruner.PreprocessQuery(raw_query, query.data());
		} else {
			std::vector<float> normalized_query(pdx_data.num_dimensions);
			quantizer.NormalizeQuery(raw_query, normalized_query.data());
			pruner.PreprocessQuery(normalized_query.data(), query.data());
		}
		size_t clusters_to_visit =
		    (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters) ? pdx_data.num_clusters : ivf_nprobe;
		// TODO: Incorporate this to U8 PDX (no IVF2)
		// GetClustersAccessOrderIVFPDX(query);
		std::vector<uint32_t> local_cluster_order;
		GetClustersAccessOrderIVF(query.data(), pdx_data, clusters_to_visit, local_cluster_order);
		// PDXearch core
		std::vector<int32_t> local_dim_clip_value(pdx_data.num_dimensions, 0);
		std::vector<QUANTIZED_VECTOR_TYPE> local_quantized_query(pdx_data.num_dimensions);
		QUANTIZED_VECTOR_TYPE *local_prepared_query;
		if constexpr (q == U8) {
			quantizer.PrepareQuery(query.data(), pdx_data.for_base, pdx_data.scale_factor, local_dim_clip_value.data(),
			                       local_quantized_query.data());
			local_prepared_query = local_quantized_query.data();
		} else {
			local_prepared_query = query.data();
		}

		alignas(64) DISTANCES_TYPE pruning_distances[MAX_EMBEDDINGS_PER_CLUSTER];
		alignas(64) uint32_t pruning_positions[MAX_EMBEDDINGS_PER_CLUSTER];

		for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
			DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = local_cluster_order[cluster_idx];
			CLUSTER_TYPE &cluster = pdx_data.clusters[current_cluster_idx];
			if (local_heap.size() < k) {
				// We cannot prune until we fill the heap
				Start(local_prepared_query, cluster.data, cluster.num_embeddings, k, cluster.indices, pruning_positions,
				      pruning_distances, local_heap, local_dim_clip_value.data());
				continue;
			}
			Warmup(local_prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold,
			       pruning_positions, pruning_distances, pruning_threshold, local_heap, current_dimension_idx,
			       n_vectors_not_pruned, local_dim_clip_value.data());
			Prune(local_prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances,
			      pruning_threshold, local_heap, current_dimension_idx, n_vectors_not_pruned, local_dim_clip_value.data());
			if (n_vectors_not_pruned) {
				MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances,
				                    nullptr, local_heap);
			}
		}
		return BuildResultSetFromHeap(k, local_heap);
	}

	std::vector<KNNCandidate_t> FilteredSearchGlobal(const float *__restrict const raw_query, const uint32_t k,
	                                                 const PredicateEvaluator &predicate_evaluator) {
		Heap<q> local_heap {};
		std::mutex local_mutex;
		best_k_mutex = &local_mutex;
		std::vector<float> query(pdx_data.num_dimensions);
		if (!pdx_data.is_normalized) {
			pruner.PreprocessQuery(raw_query, query.data());
		} else {
			std::vector<float> normalized_query(pdx_data.num_dimensions);
			quantizer.NormalizeQuery(raw_query, normalized_query.data());
			pruner.PreprocessQuery(normalized_query.data(), query.data());
		}

		size_t clusters_to_visit =
		    (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters) ? pdx_data.num_clusters : ivf_nprobe;

		std::vector<uint32_t> local_cluster_order;
		GetClustersAccessOrderIVF(query.data(), pdx_data, clusters_to_visit, local_cluster_order);
		// PDXearch core
		std::vector<int32_t> local_dim_clip_value(pdx_data.num_dimensions, 0);
		std::vector<QUANTIZED_VECTOR_TYPE> local_quantized_query(pdx_data.num_dimensions);
		QUANTIZED_VECTOR_TYPE *local_prepared_query;
		if constexpr (q == U8) {
			quantizer.PrepareQuery(query.data(), pdx_data.for_base, pdx_data.scale_factor, local_dim_clip_value.data(),
			                       local_quantized_query.data());
			local_prepared_query = local_quantized_query.data();
		} else {
			local_prepared_query = query.data();
		}

		alignas(64) DISTANCES_TYPE pruning_distances[MAX_EMBEDDINGS_PER_CLUSTER];
		alignas(64) uint32_t pruning_positions[MAX_EMBEDDINGS_PER_CLUSTER];

		for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
			DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = local_cluster_order[cluster_idx];
			auto [selection_vector, passing_tuples] =
			    predicate_evaluator.GetSelectionVector(current_cluster_idx, cluster_offsets[current_cluster_idx]);
			if (passing_tuples == 0) {
				continue;
			}
			CLUSTER_TYPE &cluster = pdx_data.clusters[current_cluster_idx];
			if (local_heap.size() < k) {
				// We cannot prune until we fill the heap
				FilteredStart(local_prepared_query, cluster.data, cluster.num_embeddings, k, cluster.indices,
				              pruning_positions, pruning_distances, local_heap, local_dim_clip_value.data(), selection_vector,
				              passing_tuples);
				continue;
			}
			Warmup<q, true>(local_prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold,
			                pruning_positions, pruning_distances, pruning_threshold, local_heap, current_dimension_idx,
			                n_vectors_not_pruned, local_dim_clip_value.data(), passing_tuples, selection_vector);
			Prune(local_prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances,
			      pruning_threshold, local_heap, current_dimension_idx, n_vectors_not_pruned, local_dim_clip_value.data());
			if (n_vectors_not_pruned) {
				MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances,
				                    nullptr, local_heap);
			}
		}
		return BuildResultSetFromHeap(k, local_heap);
	}
};

} // namespace PDX
