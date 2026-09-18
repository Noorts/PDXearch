#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/optional_idx.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/indexes/flat.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/ivf_searcher.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "index/pdxearch_index_utils.hpp"

namespace duckdb {

class PDXearchWrapper {
public:
	static constexpr PDX::DistanceMetric DEFAULT_DISTANCE_METRIC = PDX::DistanceMetric::L2SQ;
	static constexpr PDX::Quantization DEFAULT_QUANTIZATION = PDX::Quantization::U8;
	static constexpr int32_t DEFAULT_N_PROBE = 24; // 5% of the data in a full search

private:
	const uint32_t num_dimensions;
	// Whether the embeddings (both the query embeddings and those stored in the
	// index) are normalized. The PDXearch kernel only implements Euclidean
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
	    : num_dimensions(num_dimensions), is_normalized(PDX::DistanceMetricRequiresNormalization(distance_metric)),
	      distance_metric(distance_metric), quantization(quantization), n_probe(n_probe), seed(seed),
	      rotation_matrix(GenerateRandomRotationMatrix(num_dimensions, seed)) {
	}
	virtual ~PDXearchWrapper() = default;

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

	// An approximate lower bound of the index's size in memory.
	virtual uint64_t GetInMemorySizeInBytes() const = 0;
};

// The index of one DuckDB row group: the rows [row_start, row_end).
struct PDXRowGroup {
	row_t row_start;
	row_t row_end;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::IPDXIndex> index;
	// Held while the index is built, so that rows of this row group arriving in a later batch wait for it.
	std::mutex mutex;

	uint64_t GetInMemorySizeInBytes() const {
		return sizeof(*this) + (index ? index->GetInMemorySizeInBytes() : 0);
	}
};

// The PDXearchWrapper for the parallel implementation. The parallel implementation uses a separate index for each row
// group. This allows the creation of the index and searching in it to be parallelized at the row group level.
template <PDX::Quantization Q>
class PDXearchWrapperParallel : public PDXearchWrapper {
private:
	const idx_t row_group_size;
	// Sorted by row_start. Structural changes only happen during the index build and under the index's exclusive lock,
	// so scans read it without locking.
	std::vector<unique_ptr<PDXRowGroup>> row_groups;
	std::mutex row_groups_mutex;

public:
	// Below this many embeddings a row group is not worth clustering and gets an exact Flat index.
	static constexpr size_t MIN_EMBEDDINGS_FOR_CLUSTERING = 2048;

	// The number of clusters depends on the number of embeddings in the row group. We have three levels.
	// In general we aim for a 1:256 ratio of clusters to embeddings.
	static constexpr size_t ComputeNumClustersForRowGroup(const size_t num_embeddings) {
		if (num_embeddings < MIN_EMBEDDINGS_FOR_CLUSTERING) {
			// 1. No clustering: not worth indexing. One cluster minimizes overhead.
			return 1;
		} else if (num_embeddings < static_cast<size_t>(DEFAULT_ROW_GROUP_SIZE * 0.25)) {
			// 2. Small row group: It has less than 30720 embeddings (25% of a full rowgroup).
			return 120;
		} else {
			// 3. Default: As the DuckDB row group size is usually 122880, we set 480 clusters per row group. While some
			//    row groups might be smaller, 480 is still a good number, even if the row group falls down to 40k
			//    embeddings.
			return 480;
		}
	}

	PDXearchWrapperParallel(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe,
	                        int32_t seed, idx_t row_group_size)
	    : PDXearchWrapper(Q, distance_metric, num_dimensions, n_probe, seed), row_group_size(row_group_size) {
	}

	idx_t GetRowGroupSize() const {
		return row_group_size;
	}

	// Builds the index of the row group [row_start, row_start + count) from its (non-NULL) rows. Rows of this row
	// group that arrive in a later batch are appended to the index that was already built.
	void SetUpIndexForRowGroup(const row_t *const row_ids, const float *const embeddings, const idx_t num_embeddings,
	                           const row_t row_start, const idx_t count) {
		D_ASSERT(num_embeddings > 0 && num_embeddings <= count);
		PDXRowGroup *row_group = nullptr;
		std::unique_lock<std::mutex> build_lock;
		{
			const std::lock_guard<std::mutex> lock(row_groups_mutex);
			auto it =
			    std::lower_bound(row_groups.begin(), row_groups.end(), row_start,
			                     [](const unique_ptr<PDXRowGroup> &rg, row_t start) { return rg->row_start < start; });
			if (it != row_groups.end() && (*it)->row_start == row_start) {
				row_group = it->get();
			} else {
				auto new_row_group = make_uniq<PDXRowGroup>();
				new_row_group->row_start = row_start;
				new_row_group->row_end = row_start + static_cast<row_t>(count);
				new_row_group->pruner = make_uniq<PDX::ADSamplingPruner>(GetNumDimensions(), rotation_matrix.get());
				row_group = new_row_group.get();
				build_lock = std::unique_lock<std::mutex>(row_group->mutex);
				row_groups.insert(it, std::move(new_row_group));
			}
		}

		if (build_lock.owns_lock()) {
			row_group->index = BuildRowGroupIndex(*row_group, row_ids, embeddings, num_embeddings);
			return;
		}
		// Only reached when DuckDB splits a row group over several scan tasks (e.g. PRAGMA verify_parallelism hands
		// out one vector per task): the first batch built the index above, the later ones are appended to it.
		const std::lock_guard<std::mutex> lock(row_group->mutex);
		for (idx_t i = 0; i < num_embeddings; i++) {
			row_group->index->Append(static_cast<size_t>(row_ids[i]), embeddings + i * GetNumDimensions());
		}
	}

	unique_ptr<PDX::IIterativeSearch> BeginSearchForRowGroup(const idx_t row_group_idx,
	                                                         const float *const preprocessed_query_embedding,
	                                                         const idx_t limit, PDX::TopKHeap &top_k_heap,
	                                                         const std::vector<size_t> *const passing_row_ids) {
		auto search_cursor = row_groups[row_group_idx]->index->BeginIterativeSearch(
		    preprocessed_query_embedding, static_cast<uint32_t>(limit), top_k_heap, passing_row_ids,
		    /*is_query_transformed=*/true);
		return unique_ptr<PDX::IIterativeSearch>(search_cursor.release());
	}

	// Position in `row_groups` of the row group holding row_id.
	optional_idx LookupRowGroup(const row_t row_id) const {
		auto it = std::upper_bound(row_groups.begin(), row_groups.end(), row_id,
		                           [](row_t id, const unique_ptr<PDXRowGroup> &rg) { return id < rg->row_start; });
		if (it == row_groups.begin()) {
			return optional_idx();
		}
		--it;
		if (row_id >= (*it)->row_end) {
			return optional_idx();
		}
		return optional_idx(static_cast<idx_t>(it - row_groups.begin()));
	}

	const PDXRowGroup &GetRowGroup(const idx_t row_group_idx) const {
		return *row_groups[row_group_idx];
	}

	idx_t GetTotalNumClusters() const {
		idx_t total = 0;
		for (const auto &row_group : row_groups) {
			if (row_group->index) {
				total += row_group->index->GetNumClusters();
			}
		}
		return total;
	}

	idx_t GetNumRowGroups() const {
		return row_groups.size();
	}

	uint64_t GetInMemorySizeInBytes() const override {
		uint64_t in_memory_size_in_bytes = sizeof(*this);
		for (const auto &row_group : row_groups) {
			in_memory_size_in_bytes += row_group->GetInMemorySizeInBytes();
		}
		return in_memory_size_in_bytes;
	}

private:
	unique_ptr<PDX::IPDXIndex> BuildRowGroupIndex(const PDXRowGroup &row_group, const row_t *const row_ids,
	                                              const float *const embeddings, const idx_t num_embeddings) const {
		PDX::PDXIndexConfig config;
		config.num_dimensions = GetNumDimensions();
		config.distance_metric = GetDistanceMetric();
		config.seed = static_cast<uint32_t>(GetSeed());
		config.num_clusters = static_cast<uint32_t>(ComputeNumClustersForRowGroup(num_embeddings));
		config.kmeans_iters = 8;
		config.hierarchical_indexing = true;
		config.n_threads = 1;
		config.is_data_transformed = true;
		config.base_row_id = static_cast<size_t>(row_group.row_start);

		std::vector<size_t> ids(num_embeddings);
		for (idx_t i = 0; i < num_embeddings; i++) {
			ids[i] = static_cast<size_t>(row_ids[i]);
		}

		if (num_embeddings < MIN_EMBEDDINGS_FOR_CLUSTERING) {
			auto flat = make_uniq<PDX::FlatIndex>(config, *row_group.pruner);
			flat->BuildIndex(ids.data(), embeddings, num_embeddings);
			return std::move(flat);
		}
		auto ivf = make_uniq<PDX::PDXIndex<Q>>(config, *row_group.pruner);
		ivf->BuildIndex(ids.data(), embeddings, num_embeddings);
		return std::move(ivf);
	}
};

using PDXearchWrapperF32 = PDXearchWrapperParallel<PDX::F32>;
using PDXearchWrapperU8 = PDXearchWrapperParallel<PDX::U8>;

} // namespace duckdb
