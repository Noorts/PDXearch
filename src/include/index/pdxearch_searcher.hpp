#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>

#include "pdx/common.hpp"
#include "pdx/db_mock/predicate_evaluator.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/searcher.hpp"

namespace duckdb {

// A PDXearch searcher that is initialized once and can then be asked to probe the next few clusters repeatedly, while
// merging its results into a heap that is shared with the searchers of the other row groups.
//
// Upstream PDX only offers a single-shot `Search()` / `FilteredSearch()` that probes `n_probe` clusters and returns.
// The filtered index scan needs the iterative variant: it keeps probing clusters until it has found `k` results that
// pass the filter.
//
// `GetPruningThreshold`, `Warmup` and `Prune` are copied from upstream because the shared heap must be read while
// holding `best_k_mutex`, and upstream's `Warmup`/`Prune` call their own (non-virtual, unlocked) variant.
template <PDX::Quantization Q>
class IterativePDXearch : public PDX::PDXearch<Q> {
public:
	using base_t = PDX::PDXearch<Q>;
	using typename base_t::cluster_t;
	using typename base_t::data_t;
	using typename base_t::distance_computer_t;
	using typename base_t::distance_t;
	using typename base_t::index_t;
	using typename base_t::quantized_embedding_t;
	using typename base_t::tombstones_t;

	// Upstream declares this protected, but the physical scan operators need it to turn the shared heap into a result
	// set.
	using base_t::BuildResultSetFromHeap;

	IterativePDXearch(index_t &data_index, PDX::ADSamplingPruner &pruner)
	    : base_t(data_index, pruner), access_order(new uint32_t[data_index.num_clusters]),
	      quantized_query_buf(new quantized_embedding_t[data_index.num_dimensions]) {
	}

	// Initialization that works for both the Search and FilteredSearch methods.
	void InitializeSearch(float *PDX_RESTRICT const preprocessed_query, const uint32_t k, PDX::Heap &heap,
	                      std::mutex &heap_mutex,
	                      std::unique_ptr<PDX::PredicateEvaluator> predicate_evaluator = nullptr) {
		this->best_k = &heap;
		this->best_k_mutex = &heap_mutex;
		this->k = k;
		this->predicate_evaluator = std::move(predicate_evaluator);

		base_t::GetClustersAccessOrderIVF(preprocessed_query, this->pdx_data, this->pdx_data.num_clusters,
		                                  access_order.get());

		access_order_offset = 0; // Reset cluster index offset for new search.
		if constexpr (Q == PDX::U8) {
			this->quantizer.QuantizeEmbedding(preprocessed_query, this->pdx_data.quantization_base,
			                                  this->pdx_data.quantization_scale, quantized_query_buf.get());
			this->prepared_query = quantized_query_buf.get();
		} else {
			this->prepared_query = preprocessed_query;
		}
	}

	void Search(const size_t num_clusters_to_probe) {
		// Partial precondition check to ensure the state was reset / initialized.
		assert(best_k);
		assert(best_k_mutex);
		assert(this->k != 0);
		assert(this->prepared_query);

		const size_t buffer_size = this->pdx_data.max_cluster_capacity;
		std::unique_ptr<distance_t[]> pruning_distances(new distance_t[buffer_size]);
		std::unique_ptr<uint32_t[]> pruning_positions(new uint32_t[buffer_size]);

		const size_t end_idx = std::min<size_t>(num_clusters_to_probe, this->pdx_data.num_clusters);
		for (size_t cluster_idx = 0; cluster_idx < end_idx; ++cluster_idx) {
			distance_t pruning_threshold = std::numeric_limits<distance_t>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = access_order[cluster_idx];
			const cluster_t &cluster = this->pdx_data.clusters[current_cluster_idx];
			if (cluster.num_embeddings == 0) {
				continue;
			}

			Warmup(this->prepared_query, cluster.data, cluster.used_capacity, cluster.max_capacity, this->k,
			       this->selectivity_threshold, pruning_positions.get(), pruning_distances.get(), pruning_threshold,
			       *best_k, current_dimension_idx, n_vectors_not_pruned, cluster.tombstones);
			Prune(this->prepared_query, cluster.data, cluster.used_capacity, cluster.max_capacity, this->k,
			      pruning_positions.get(), pruning_distances.get(), pruning_threshold, *best_k, current_dimension_idx,
			      n_vectors_not_pruned, cluster.tombstones);
			if (n_vectors_not_pruned) {
				const std::lock_guard<std::mutex> lock(*best_k_mutex);
				this->MergeIntoHeap(cluster.indices, n_vectors_not_pruned, this->k, pruning_positions.get(),
				                    pruning_distances.get(), *best_k);
			}
		}
	}

	// Tries to probe the next few clusters. Class state tracks which clusters have already been probed. Stops probing
	// once all clusters have been probed.
	void FilteredSearch(const size_t num_clusters_to_try_to_probe) {
		// Partial precondition check to ensure the state was reset / initialized.
		assert(best_k);
		assert(best_k_mutex);
		assert(this->k != 0);
		assert(this->predicate_evaluator);
		assert(this->prepared_query);

		const size_t buffer_size = this->pdx_data.max_cluster_capacity;
		std::unique_ptr<distance_t[]> pruning_distances(new distance_t[buffer_size]);
		std::unique_ptr<uint32_t[]> pruning_positions(new uint32_t[buffer_size]);

		const size_t end_idx =
		    std::min<size_t>(access_order_offset + num_clusters_to_try_to_probe, this->pdx_data.num_clusters);
		for (; access_order_offset < end_idx; ++access_order_offset) {
			distance_t pruning_threshold = std::numeric_limits<distance_t>::max();
			uint32_t current_dimension_idx = 0;
			size_t n_vectors_not_pruned = 0;

			const size_t current_cluster_idx = access_order[access_order_offset];
			auto [selection_vector, passing_tuples] = this->predicate_evaluator->GetSelectionVector(
			    current_cluster_idx, this->pdx_data.cluster_offsets[current_cluster_idx]);
			if (passing_tuples == 0) {
				continue;
			}
			const cluster_t &cluster = this->pdx_data.clusters[current_cluster_idx];
			if (cluster.num_embeddings == 0) {
				continue;
			}

			Warmup<true>(this->prepared_query, cluster.data, cluster.used_capacity, cluster.max_capacity, this->k,
			             this->selectivity_threshold, pruning_positions.get(), pruning_distances.get(),
			             pruning_threshold, *best_k, current_dimension_idx, n_vectors_not_pruned, cluster.tombstones,
			             passing_tuples, selection_vector);
			Prune<true>(this->prepared_query, cluster.data, cluster.used_capacity, cluster.max_capacity, this->k,
			            pruning_positions.get(), pruning_distances.get(), pruning_threshold, *best_k,
			            current_dimension_idx, n_vectors_not_pruned, cluster.tombstones, selection_vector);
			if (n_vectors_not_pruned) {
				const std::lock_guard<std::mutex> lock(*best_k_mutex);
				this->MergeIntoHeap(cluster.indices, n_vectors_not_pruned, this->k, pruning_positions.get(),
				                    pruning_distances.get(), *best_k);
			}
		}
		assert(access_order_offset <= this->pdx_data.num_clusters);
	}

protected:
	// The pruning threshold by default is the top of the heap. Copied from `PDX::PDXearch` so that the shared heap is
	// only read while holding `best_k_mutex`.
	void GetPruningThreshold(uint32_t k, PDX::Heap &heap, distance_t &pruning_threshold,
	                         uint32_t current_dimension_idx) {
		const std::lock_guard<std::mutex> lock(*best_k_mutex);
		const float float_threshold = this->pruner.GetPruningThreshold(k, heap, current_dimension_idx);
		if constexpr (Q == PDX::U8) {
			// We need to avoid undefined behaviour when overflow happens
			const float scaled = float_threshold * this->pdx_data.quantization_scale_squared;
			pruning_threshold = scaled >= static_cast<float>(std::numeric_limits<distance_t>::max())
			                        ? std::numeric_limits<distance_t>::max()
			                        : static_cast<distance_t>(scaled);
		} else {
			pruning_threshold = float_threshold;
		}
	}

	// On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors is low. Copied from
	// `PDX::PDXearch` so that it picks up the locking `GetPruningThreshold` above.
	template <bool FILTERED = false>
	void Warmup(const quantized_embedding_t *PDX_RESTRICT query, const data_t *PDX_RESTRICT data,
	            const size_t n_vectors, const size_t buffer_stride, uint32_t k, float tuples_threshold,
	            uint32_t *pruning_positions, distance_t *pruning_distances, distance_t &pruning_threshold,
	            PDX::Heap &heap, uint32_t &current_dimension_idx, size_t &n_vectors_not_pruned,
	            const tombstones_t &tombstones, uint32_t passing_tuples = 0, uint8_t *selection_vector = nullptr) {
		current_dimension_idx = 0;
		size_t cur_subgrouping_size_idx = 0;
		size_t tuples_needed_to_exit = static_cast<size_t>(std::ceil(tuples_threshold * static_cast<float>(n_vectors)));
		this->ResetPruningDistances(n_vectors, pruning_distances);
		this->MaskDistancesWithTombstones(tombstones, pruning_distances);
		uint32_t n_tuples_to_prune = 0;
		if constexpr (FILTERED) {
			float selection_percentage = (static_cast<float>(passing_tuples) / static_cast<float>(n_vectors));
			this->MaskDistancesWithSelectionVector(n_vectors, pruning_distances, selection_vector);
			if (selection_percentage < (1 - tuples_threshold)) {
				// Go directly to the PRUNE phase for direct tuples access in the Horizontal block
				return;
			}
		}
		GetPruningThreshold(k, heap, pruning_threshold, current_dimension_idx);
		while (n_tuples_to_prune < tuples_needed_to_exit &&
		       current_dimension_idx < this->pdx_data.num_vertical_dimensions) {
			size_t last_dimension_to_fetch =
			    std::min(current_dimension_idx + PDX::DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
			             this->pdx_data.num_vertical_dimensions);
			distance_computer_t::Vertical(query, data, n_vectors, buffer_stride, current_dimension_idx,
			                              last_dimension_to_fetch, pruning_distances, pruning_positions);
			current_dimension_idx = last_dimension_to_fetch;
			cur_subgrouping_size_idx += 1;
			GetPruningThreshold(k, heap, pruning_threshold, current_dimension_idx);
			n_tuples_to_prune = 0;
			this->EvaluatePruningPredicateScalar(n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold);
		}
	}

	// We scan only the not-yet pruned vectors. Copied from `PDX::PDXearch` so that it picks up the locking
	// `GetPruningThreshold` above.
	template <bool FILTERED = false>
	void Prune(const quantized_embedding_t *PDX_RESTRICT query, const data_t *PDX_RESTRICT data, const size_t n_vectors,
	           const size_t buffer_stride, uint32_t k, uint32_t *pruning_positions, distance_t *pruning_distances,
	           distance_t &pruning_threshold, PDX::Heap &heap, uint32_t &current_dimension_idx,
	           size_t &n_vectors_not_pruned, const tombstones_t &tombstones,
	           const uint8_t *selection_vector = nullptr) {
		GetPruningThreshold(k, heap, pruning_threshold, current_dimension_idx);
		this->MaskDistancesWithTombstones(tombstones, pruning_distances);
		this->template InitPositionsArray<FILTERED>(n_vectors, n_vectors_not_pruned, pruning_positions,
		                                            pruning_threshold, pruning_distances, selection_vector);
		size_t cur_n_vectors_not_pruned = 0;
		size_t current_vertical_dimension = current_dimension_idx;
		size_t current_horizontal_dimension = 0;
		while (this->pdx_data.num_horizontal_dimensions && n_vectors_not_pruned &&
		       current_horizontal_dimension < this->pdx_data.num_horizontal_dimensions) {
			cur_n_vectors_not_pruned = n_vectors_not_pruned;
			size_t offset_data = (this->pdx_data.num_vertical_dimensions * buffer_stride) +
			                     (current_horizontal_dimension * buffer_stride);
			for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
				size_t v_idx = pruning_positions[vector_idx];
				size_t data_pos = offset_data + (v_idx * PDX::H_DIM_SIZE);
				__builtin_prefetch(data + data_pos, 0, 3);
			}
			size_t offset_query = this->pdx_data.num_vertical_dimensions + current_horizontal_dimension;
			for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
				size_t v_idx = pruning_positions[vector_idx];
				size_t data_pos = offset_data + (v_idx * PDX::H_DIM_SIZE);
				pruning_distances[v_idx] +=
				    distance_computer_t::Horizontal(query + offset_query, data + data_pos, PDX::H_DIM_SIZE);
			}
			// end of clipping
			current_horizontal_dimension += PDX::H_DIM_SIZE;
			current_dimension_idx += PDX::H_DIM_SIZE;
			GetPruningThreshold(k, heap, pruning_threshold, current_dimension_idx);
			assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
			this->EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned, n_vectors_not_pruned,
			                                               pruning_positions, pruning_threshold, pruning_distances);
		}
		// GO THROUGH THE REST IN THE VERTICAL
		while (n_vectors_not_pruned && current_vertical_dimension < this->pdx_data.num_vertical_dimensions) {
			cur_n_vectors_not_pruned = n_vectors_not_pruned;
			size_t last_dimension_to_test_idx = std::min(current_vertical_dimension + PDX::H_DIM_SIZE,
			                                             static_cast<size_t>(this->pdx_data.num_vertical_dimensions));
			distance_computer_t::VerticalPruning(query, data, cur_n_vectors_not_pruned, buffer_stride,
			                                     current_vertical_dimension, last_dimension_to_test_idx,
			                                     pruning_distances, pruning_positions);
			current_dimension_idx =
			    std::min(current_dimension_idx + PDX::H_DIM_SIZE, static_cast<size_t>(this->pdx_data.num_dimensions));
			current_vertical_dimension = std::min(static_cast<uint32_t>(current_vertical_dimension + PDX::H_DIM_SIZE),
			                                      this->pdx_data.num_vertical_dimensions);
			assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
			GetPruningThreshold(k, heap, pruning_threshold, current_dimension_idx);
			this->EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned, n_vectors_not_pruned,
			                                               pruning_positions, pruning_threshold, pruning_distances);
			if (current_dimension_idx == this->pdx_data.num_dimensions) {
				break;
			}
		}
	}

private:
	// The top-k heap shared by all row groups' searchers. Guarded by `best_k_mutex`.
	PDX::Heap *best_k = nullptr;
	std::mutex *best_k_mutex = nullptr;

	// Prioritized list of indices of the clusters to probe. E.g., [0, 2, 1].
	std::unique_ptr<uint32_t[]> access_order;
	// Indexes into the `access_order` list. This offset is incremented by 1 after probing a cluster.
	uint64_t access_order_offset {0};

	// Per-search query buffer, filled by QuantizeEmbedding in InitializeSearch (U8 path).
	std::unique_ptr<quantized_embedding_t[]> quantized_query_buf;
};

} // namespace duckdb
