#pragma once

#include "duckdb/common/exception.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

#include "pdxearch/common.hpp"
#include "pdxearch/db_mock/predicate_evaluator.hpp"
#include "pdxearch/index_base/pdx_ivf.hpp"
#include "pdxearch/pruners/adsampling.hpp"
#include "pdxearch/pdxearch.hpp"
#include "duckdb/common/helper.hpp"
#include "index/pdxearch_index_utils.hpp"
#include "index/pdxearch_kmeans.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

class PDXearchWrapper {
public:
	static constexpr PDX::DistanceMetric DEFAULT_DISTANCE_METRIC = PDX::DistanceMetric::L2SQ;
	static constexpr PDX::Quantization DEFAULT_QUANTIZATION = PDX::Quantization::U8;
	static constexpr int32_t DEFAULT_N_PROBE = 128;

private:
	const uint32_t num_dimensions;
	// Whether the embeddings (both the query embeddings and those stored in the
	// index) are normalized. The PDXearch kernel only implements Euclidian
	// distance (L2SQ). To compute the cosine and inner product distances we
	// normalize and then use L2SQ.
	const bool is_normalized;

	// The fields below are index options that can be set during index creation:
	// `CREATE INDEX ON t USING PDXearch(vec) WITH (n_probe = 10, seed = 42)`.
	// See `pdxearch_index_plan.cpp` for the validation logic, and
	// `pdxearch_index.cpp` for the usage.
	const PDX::DistanceMetric distance_metric;
	const PDX::Quantization quantization;
	// Between 16 and 128 is common. Setting this to 0 will probe all clusters.
	// Can be set at index build time using the 'n_probe' index option, else
	// uses `DEFAULT_N_PROBE`. At query time the 'pdxearch_n_probe' runtime
	// setting will take precedence over the n_probe saved here.
	const uint32_t n_probe;
	// Seed that currently affects the random rotation matrix generation.
	const int32_t seed;

protected:
	const unique_ptr<float[]> rotation_matrix;

public:
	PDXearchWrapper(PDX::Quantization quantization, PDX::DistanceMetric distance_metric, uint32_t num_dimensions,
	                uint32_t n_probe, int32_t seed)
	    : num_dimensions(num_dimensions), is_normalized(DistanceMetricRequiresNormalization(distance_metric)),
	      distance_metric(distance_metric), quantization(quantization), n_probe(n_probe), seed(seed),
	      rotation_matrix(GenerateRandomRotationMatrix(num_dimensions, seed)) {
	}
	virtual ~PDXearchWrapper() = default;

	void Insert(row_t row_id, const float *embedding) {
		throw NotImplementedException("PDXearchWrapper::Insert() not implemented");
	}
	void Delete(row_t row_id) {
		throw NotImplementedException("PDXearchWrapper::Delete() not implemented");
	}

	uint64_t GetInMemorySize() const {
		throw NotImplementedException("PDXearchWrapper::GetInMemorySize() not implemented");
	}

	uint32_t GetNumDimensions() const {
		return num_dimensions;
	}
	PDX::DistanceMetric GetDistanceMetric() const {
		return distance_metric;
	}
	PDX::Quantization GetQuantization() const {
		return quantization;
	}
	bool IsNormalized() const {
		return is_normalized;
	}
	uint32_t GetNProbe() const {
		return n_probe;
	}
	int32_t GetSeed() const {
		return seed;
	}
	float *GetRotationMatrix() const {
		return rotation_matrix.get();
	}
};

struct RowIdClusterMapping {
	uint32_t cluster_id;
	uint32_t index_in_cluster;
};

// All state required for storing a row group's embeddings and searching them.
template <PDX::Quantization Q>
class PDXRowGroup {
public:
	// Row group embedding storage and metadata.
	std::unique_ptr<PDX::IndexPDXIVF<Q>> index;
	std::vector<RowIdClusterMapping> row_id_metadata {DEFAULT_ROW_GROUP_SIZE};

	std::unique_ptr<PDX::ADSamplingPruner<Q>> pruner;
	// The searcher is reinitialized and reused across DuckDB queries.
	std::unique_ptr<PDX::PDXearch<Q>> searcher;
};

// The PDXearchWrapper for the parallel implementation. The parallel implementation uses a separate index for each row
// group. This allows the creation of the index and searching in it to be parallelized at the row group level.
template <PDX::Quantization Q>
class PDXearchWrapperParallel : public PDXearchWrapper {
public:
	// We aim for a 1:256 ratio of clusters to embeddings. As a DuckDB rowgroup
	// size is usually 122880, we set 480 clusters per row group. While some
	// row groups might be smaller, 480 is still a good number, even if the
	// row group falls down to 40k embeddings.
	static constexpr size_t DEFAULT_N_CLUSTERS_PER_ROW_GROUP = 480;

private:
	uint32_t num_clusters_per_row_group {};
	std::vector<PDXRowGroup<Q>> row_groups;

public:
	PDXearchWrapperParallel(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe,
	                        int32_t seed, idx_t estimated_cardinality)
	    : PDXearchWrapper(Q, distance_metric, num_dimensions, n_probe, seed) {
		if (estimated_cardinality == 0) {
			throw InternalException(
			    "Something went wrong: estimated_cardinality is 0. This is likely because a malformed persisted index "
			    "was loaded. Index persistence is not supported yet, but DuckDB will still try to persist it. Open "
			    "your database file manually (duckdb test.db) and drop the index(es). Run 'SELECT sql FROM "
			    "duckdb_indexes();' to see the indexes, and then 'DROP INDEX index_name;' to drop the unused "
			    "index(es).");
		}
		const idx_t estimated_num_row_groups =
		    static_cast<idx_t>(std::ceil((float)estimated_cardinality / DEFAULT_ROW_GROUP_SIZE));
		D_ASSERT(estimated_num_row_groups > 0);
		row_groups.resize(estimated_num_row_groups);

		num_clusters_per_row_group = DEFAULT_N_CLUSTERS_PER_ROW_GROUP;
	}

	// Initialize the wrapper's state for this row group. This is called once per row group.
	void SetUpIndexForRowGroup(const row_t *const row_ids, const float *const embeddings, const idx_t num_embeddings,
	                           const idx_t row_group_id) {
		PDXRowGroup<Q> &row_group = row_groups[row_group_id];

		const auto num_dimensions = GetNumDimensions();
		// Additional constraints on the number of dimensions are enforced in `pdxearch_index_plan.cpp`.
		D_ASSERT(num_dimensions > 0);
		D_ASSERT(num_embeddings > 0);
		D_ASSERT(num_clusters_per_row_group > 0);
		// TODO(@lkuffo): See issue #38. num_embeddings < DEFAULT_N_CLUSTERS_PER_ROW_GROUP is currently not handled.
		D_ASSERT(num_embeddings >= num_clusters_per_row_group);

		// Compute quantization parameters for U8 from the embeddings' value range.
		float quantization_base = 0.0f;
		float quantization_scale = 1.0f;
		if constexpr (Q == PDX::U8) {
			float global_min = std::numeric_limits<float>::max();
			float global_max = std::numeric_limits<float>::lowest();
			for (idx_t i = 0; i < static_cast<idx_t>(num_embeddings) * num_dimensions; ++i) {
				global_min = std::min(global_min, embeddings[i]);
				global_max = std::max(global_max, embeddings[i]);
			}
			quantization_base = global_min;
			float range = global_max - global_min;
			quantization_scale = (range > 0) ? 255.0f / range : 1.0f;
		}

		if constexpr (Q == PDX::U8) {
			row_group.index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings,
			                                                  num_clusters_per_row_group, IsNormalized(),
			                                                  quantization_scale, quantization_base);
		} else {
			row_group.index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings,
			                                                  num_clusters_per_row_group, IsNormalized());
		}
		row_group.pruner = make_uniq<PDX::ADSamplingPruner<Q>>(num_dimensions, rotation_matrix.get());

		// Compute K-means centroids and embedding-to-centroid assignment (always on float embeddings).
		KMeansResult kmeans_result = ComputeKMeans(embeddings, num_embeddings, num_dimensions,
		                                           num_clusters_per_row_group, GetDistanceMetric(), GetSeed());

		// Store centroids.
		row_group.index->centroids = std::move(kmeans_result.centroids);

		// Row-major buffer that the current cluster's embeddings are "gathered" into. This buffer is the source for
		// StoreClusterEmbeddings, the result of which is persistently stored in the index. The buffer is reused across
		// clusters. For F32: buffer is float. For U8: buffer is uint8_t (quantized).
		using EmbeddingStorageType = PDX::DataType_t<Q>;
		size_t max_cluster_size = 0;
		for (size_t i = 0; i < num_clusters_per_row_group; i++) {
			max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
		}
		auto tmp_cluster_embeddings =
		    std::make_unique<EmbeddingStorageType[]>(static_cast<uint64_t>(max_cluster_size * num_dimensions));

		// Set up the IVF clusters' metadata and store the embeddings.
		for (size_t cluster_idx = 0; cluster_idx < num_clusters_per_row_group; cluster_idx++) {
			const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
			auto &cluster = row_group.index->clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = kmeans_result.assignments[cluster_idx][position_in_cluster];
				const row_t row_id = row_ids[embedding_idx];

				row_group.row_id_metadata[row_id % DEFAULT_ROW_GROUP_SIZE] = {
				    static_cast<uint32_t>(cluster_idx), static_cast<uint32_t>(position_in_cluster)};
				cluster.indices[position_in_cluster] = row_id;

				if constexpr (Q == PDX::U8) {
					// Quantize float embedding to uint8.
					for (size_t d = 0; d < num_dimensions; d++) {
						float val = embeddings[embedding_idx * num_dimensions + d];
						int rounded = static_cast<int>(std::round((val - quantization_base) * quantization_scale));
						tmp_cluster_embeddings[position_in_cluster * num_dimensions + d] =
						    static_cast<uint8_t>(std::clamp(rounded, 0, 255));
					}
				} else {
					memcpy(tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
					       embeddings + (embedding_idx * num_dimensions), num_dimensions * sizeof(float));
				}
			}

			StoreClusterEmbeddings<Q, EmbeddingStorageType>(cluster, *row_group.index, tmp_cluster_embeddings.get(),
			                                                cluster_size);
		}

		// Note: the searcher depends on a fully initialized index in its constructor.
		row_group.searcher = make_uniq<PDX::PDXearch<Q>>(*row_group.index, *row_group.pruner);
	}

	void InitializeSearchForRowGroup(float *const preprocessed_query_embedding, const idx_t limit,
	                                 const idx_t row_group_id, PDX::Heap<PDX::F32> &heap, std::mutex &heap_mutex) {
		PDXRowGroup<Q> &row_group = row_groups[row_group_id];
		row_group.searcher->InitializeSearch(preprocessed_query_embedding, limit, heap, heap_mutex);
	}

	void SearchRowGroup(const idx_t row_group_id, const idx_t num_clusters_to_probe) {
		PDXRowGroup<Q> &row_group = row_groups[row_group_id];
		row_group.searcher->Search(num_clusters_to_probe);
	}

	void InitializeFilteredSearchForRowGroup(float *const preprocessed_query_embedding, const idx_t limit,
	                                         const std::vector<row_t> &passing_row_ids, const idx_t row_group_id,
	                                         PDX::Heap<PDX::F32> &heap, std::mutex &heap_mutex) {
		PDXRowGroup<Q> &row_group = row_groups[row_group_id];

		std::unique_ptr<PDX::PredicateEvaluator> predicate_evaluator =
		    make_uniq<PDX::PredicateEvaluator>(CreatePredicateEvaluatorForRowGroup(passing_row_ids, row_group));

		row_group.searcher->InitializeSearch(preprocessed_query_embedding, limit, heap, heap_mutex,
		                                     std::move(predicate_evaluator));
	}

	void FilteredSearchRowGroup(const idx_t row_group_id, const idx_t num_clusters_to_probe) {
		PDXRowGroup<Q> &row_group = row_groups[row_group_id];
		row_group.searcher->FilteredSearch(num_clusters_to_probe);
	}

	static PDX::PredicateEvaluator CreatePredicateEvaluatorForRowGroup(const std::vector<row_t> &passing_row_ids,
	                                                                   const PDXRowGroup<Q> &row_group) {
		PDX::PredicateEvaluator predicate_evaluator(row_group.index->num_clusters,
		                                            row_group.index->total_num_embeddings);

		for (auto &row_id : passing_row_ids) {
			const auto &[cluster_id, index_in_cluster] = row_group.row_id_metadata[row_id % DEFAULT_ROW_GROUP_SIZE];
			predicate_evaluator.n_passing_tuples[cluster_id]++;
			predicate_evaluator.selection_vector[(row_group.searcher->cluster_offsets[cluster_id]) + index_in_cluster] =
			    1;
		}

		return predicate_evaluator;
	}

	uint32_t GetNumClustersPerRowGroup() const {
		return num_clusters_per_row_group;
	}
	idx_t GetNumRowGroups() const {
		return row_groups.size();
	}
};

using PDXearchWrapperF32 = PDXearchWrapperParallel<PDX::F32>;
using PDXearchWrapperU8 = PDXearchWrapperParallel<PDX::U8>;

// The PDXearchWrapper for the global implementation. The global implementation uses a single PDX index to represent and
// search all embeddings in the DuckDB table.
template <PDX::Quantization Q>
class PDXearchWrapperGlobal : public PDXearchWrapper {
private:
	uint32_t num_clusters {};
	uint64_t total_num_embeddings {};
	std::vector<RowIdClusterMapping> row_id_cluster_mapping;

	std::unique_ptr<PDX::IndexPDXIVF<Q>> index;
	std::unique_ptr<PDX::ADSamplingPruner<Q>> pruner;
	std::unique_ptr<PDX::PDXearch<Q>> searcher;

public:
	PDXearchWrapperGlobal(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe,
	                      int32_t seed, idx_t estimated_cardinality)
	    : PDXearchWrapper(Q, distance_metric, num_dimensions, n_probe, seed),
	      num_clusters(ComputeNumberOfClusters(estimated_cardinality)), total_num_embeddings(estimated_cardinality),
	      row_id_cluster_mapping(estimated_cardinality),
	      pruner(make_uniq<PDX::ADSamplingPruner<Q>>(num_dimensions, rotation_matrix.get())) {
		// Additional constraints on the number of dimensions are enforced in `pdxearch_index_plan.cpp`.
		D_ASSERT(num_dimensions > 0);
		D_ASSERT(estimated_cardinality > 0);
		D_ASSERT(num_clusters > 0);
	}

	// Initialize the wrapper's state. This is called once.
	void SetUpGlobalIndex(const row_t *const row_ids, const float *const embeddings, const idx_t num_embeddings) {
		D_ASSERT(num_embeddings == total_num_embeddings);

		const auto num_dimensions = GetNumDimensions();

		// Compute quantization parameters for U8 from the embeddings' value range.
		float quantization_base = 0.0f;
		float quantization_scale = 1.0f;
		if constexpr (Q == PDX::U8) {
			float global_min = std::numeric_limits<float>::max();
			float global_max = std::numeric_limits<float>::lowest();
			for (idx_t i = 0; i < static_cast<idx_t>(num_embeddings) * num_dimensions; ++i) {
				global_min = std::min(global_min, embeddings[i]);
				global_max = std::max(global_max, embeddings[i]);
			}
			quantization_base = global_min;
			float range = global_max - global_min;
			quantization_scale = (range > 0) ? 255.0f / range : 1.0f;
		}

		if constexpr (Q == PDX::U8) {
			index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings, num_clusters, IsNormalized(),
			                                        quantization_scale, quantization_base);
		} else {
			index = make_uniq<PDX::IndexPDXIVF<Q>>(num_dimensions, num_embeddings, num_clusters, IsNormalized());
		}

		// Compute K-means centroids and embedding-to-centroid assignment (always on float embeddings).
		KMeansResult kmeans_result = ComputeKMeans(embeddings, num_embeddings, num_dimensions, num_clusters,
		                                           GetDistanceMetric(), GetSeed());

		// Store centroids.
		index->centroids = std::move(kmeans_result.centroids);

		// Row-major buffer that the current cluster's embeddings are "gathered" into. This buffer is the source for
		// StoreClusterEmbeddings, the result of which is persistently stored in the index. The buffer is reused across
		// clusters. For F32: buffer is float. For U8: buffer is uint8_t (quantized).
		using EmbeddingStorageType = PDX::DataType_t<Q>;
		size_t max_cluster_size = 0;
		for (size_t i = 0; i < num_clusters; i++) {
			max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
		}
		auto tmp_cluster_embeddings =
		    std::make_unique<EmbeddingStorageType[]>(static_cast<uint64_t>(max_cluster_size * num_dimensions));

		// Set up the IVF clusters' metadata and store the embeddings.
		for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
			const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
			auto &cluster = index->clusters.emplace_back(cluster_size, num_dimensions);

			for (size_t position_in_cluster = 0; position_in_cluster < cluster_size; position_in_cluster++) {
				const auto embedding_idx = kmeans_result.assignments[cluster_idx][position_in_cluster];
				const row_t row_id = row_ids[embedding_idx];

				D_ASSERT(row_id < total_num_embeddings);
				(row_id_cluster_mapping)[row_id] = {static_cast<uint32_t>(cluster_idx),
				                                    static_cast<uint32_t>(position_in_cluster)};
				cluster.indices[position_in_cluster] = row_id;

				if constexpr (Q == PDX::U8) {
					// Quantize float embedding to uint8.
					for (size_t d = 0; d < num_dimensions; d++) {
						float val = embeddings[embedding_idx * num_dimensions + d];
						int rounded = static_cast<int>(std::round((val - quantization_base) * quantization_scale));
						tmp_cluster_embeddings[position_in_cluster * num_dimensions + d] =
						    static_cast<uint8_t>(std::clamp(rounded, 0, 255));
					}
				} else {
					memcpy(tmp_cluster_embeddings.get() + (position_in_cluster * num_dimensions),
					       embeddings + (embedding_idx * num_dimensions), num_dimensions * sizeof(float));
				}
			}

			StoreClusterEmbeddings<Q, EmbeddingStorageType>(cluster, *index, tmp_cluster_embeddings.get(), cluster_size);
		}

		// Note: the searcher depends on a fully initialized index in its constructor.
		searcher = make_uniq<PDX::PDXearch<Q>>(*index, *pruner);
	}

	std::unique_ptr<std::vector<row_t>> Search(const float *const query_embedding, const idx_t limit,
	                                           const uint32_t n_probe) const {
		searcher->SetNProbe(n_probe);
		const std::vector<PDX::KNNCandidate<PDX::F32>> results = searcher->SearchGlobal(query_embedding, limit);
		std::unique_ptr<std::vector<row_t>> row_ids = make_uniq<std::vector<row_t>>(results.size());
		for (size_t i = 0; i < results.size(); i++) {
			(*row_ids)[i] = results[i].index;
		}
		return row_ids;
	}

	PDX::PredicateEvaluator CreatePredicateEvaluator(std::vector<std::pair<Vector, idx_t>> &row_id_vectors) const {
		auto predicate_evaluator = PDX::PredicateEvaluator(num_clusters, total_num_embeddings);

		// Set the number of tuples per cluster that passed the predicate and
		// set up selection vectors using passed row IDs.
		for (auto &[row_id_vector, vector_size] : row_id_vectors) {
			row_id_vector.Flatten(vector_size);
			const auto row_id_data = FlatVector::GetData<row_t>(row_id_vector);
			const auto &validity = FlatVector::Validity(row_id_vector);

			for (idx_t i = 0; i < vector_size; i++) {
				if (validity.RowIsValid(i)) {
					const auto &[cluster_id, index_in_cluster] = (row_id_cluster_mapping)[row_id_data[i]];
					predicate_evaluator.n_passing_tuples[cluster_id]++;
					predicate_evaluator.selection_vector[searcher->cluster_offsets[cluster_id] + index_in_cluster] = 1;
				}
			}
		}

		return predicate_evaluator;
	}

	std::unique_ptr<std::vector<row_t>> FilteredSearch(const float *const query_embedding, const idx_t limit,
	                                                   std::vector<std::pair<Vector, idx_t>> &row_id_vectors,
	                                                   const uint32_t n_probe) const {
		const PDX::PredicateEvaluator predicate_evaluator = CreatePredicateEvaluator(row_id_vectors);

		searcher->SetNProbe(n_probe);
		std::vector<PDX::KNNCandidate<PDX::F32>> results =
		    searcher->FilteredSearchGlobal(query_embedding, limit, predicate_evaluator);
		std::unique_ptr<std::vector<row_t>> row_ids = make_uniq<std::vector<row_t>>(results.size());
		for (size_t i = 0; i < results.size(); i++) {
			(*row_ids)[i] = results[i].index;
		}
		return row_ids;
	}

	idx_t GetNumClusters() const {
		return num_clusters;
	}
};

using PDXearchWrapperGlobalF32 = PDXearchWrapperGlobal<PDX::F32>;
using PDXearchWrapperGlobalU8 = PDXearchWrapperGlobal<PDX::U8>;

} // namespace duckdb
