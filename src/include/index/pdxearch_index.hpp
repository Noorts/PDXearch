#pragma once

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_lock.hpp"

#include "pdx/common.hpp"
#include "index/pdxearch_wrapper.hpp"

namespace duckdb {

// A DuckDB row group: the rows [row_start, row_start + count).
struct PDXearchRowGroupBounds {
	row_t row_start;
	idx_t count;
};

struct PDXearchIndexStats {
	string metric;
	string quantization;
	int64_t num_dimensions;
	int64_t n_probe;
	int64_t seed;
	bool is_normalized;
	int64_t approximate_lower_bound_memory_usage_bytes;
};

class PDXearchIndex : public BoundIndex {
public:
	static constexpr const char *TYPE_NAME = "PDXEARCH";

	static const case_insensitive_map_t<PDX::DistanceMetric> DISTANCE_METRIC_MAP;
	static const case_insensitive_map_t<PDX::Quantization> QUANTIZATION_MAP;

private:
	unique_ptr<PDXearchWrapper> pdxearch_wrapper;
	// The indexed table; its row group segment tree is the source of truth for the row groups we mirror.
	optional_ptr<DataTable> table;

	unique_ptr<ExpressionMatcher> function_matcher;
	IndexPointer root_block_ptr;

	// A readers-writers (shared-exclusive; exclusive prioritized) lock that
	// lets any number of index scans run concurrently (shared) while a
	// maintenance operation excludes them all (exclusive). Scans only read
	// index state; their per-query state lives in the PDX::IIterativeSearch
	// cursors they own. DuckDB's IndexLock is also acquired implicitly in
	// BoundIndex, but this is insufficient to guard 1+ index scans from the
	// maintenance operations. Hence, we add this rwlock.
	//
	// See the pull request this was introduced in for more context.
	StorageLock rwlock;

public:
	PDXearchIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	              TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	              AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	              const IndexStorageInfo &info = IndexStorageInfo(), optional_ptr<DataTable> table = nullptr);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	/******************************************************************
	 * Index creation and search methods specific to the parallel implementation
	 ******************************************************************/

	unique_ptr<StorageLockKey> TakeSearchLock();

	idx_t GetRowGroupSize() const;

	// The DuckDB row group that holds row_id. False if the row is not in the table (yet).
	bool TryGetPhysicalRowGroup(row_t row_id, PDXearchRowGroupBounds &result) const;

	vector<PDXearchRowGroupBounds> GetPhysicalRowGroups() const;

	// Position of the index's row group that holds row_id (the row_group_idx of the methods below).
	optional_idx LookupRowGroup(row_t row_id) const;

	void SetUpIndexForRowGroup(const row_t *row_ids, const float *embeddings, idx_t num_embeddings, row_t row_start,
	                           idx_t count);

	// passing_row_ids are size_t because they go straight into PDX's IPDXIndex::BeginIterativeSearch.
	unique_ptr<PDX::IIterativeSearch> BeginSearchForRowGroup(idx_t row_group_idx,
	                                                         const float *preprocessed_query_embedding, idx_t limit,
	                                                         PDX::TopKHeap &top_k_heap,
	                                                         const std::vector<size_t> *passing_row_ids);

	/******************************************************************
	 * Index maintenance
	 ******************************************************************/

	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	void ResetStorage(IndexLock &lock) override;

	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;

	void Vacuum(IndexLock &state) override;

	void Verify(IndexLock &state) override;

	string ToString(IndexLock &state, bool display_ascii) override;

	void VerifyAllocations(IndexLock &state) override;

	idx_t GetInMemorySize(IndexLock &state) override;

	unique_ptr<PDXearchIndexStats> GetStats(const ClientContext &context) const;

	/******************************************************************
	 * Index persistence
	 ******************************************************************/

	void PersistToDisk();

	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;

	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;

	/******************************************************************
	 * Misc.
	 ******************************************************************/

	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override {
		return "Constraint violation in PDXearch index";
	}

	bool TryMatchDistanceFunction(const unique_ptr<Expression> &expr, vector<reference<Expression>> &bindings) const;

	bool TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const;

	unique_ptr<ExpressionMatcher> MakeFunctionMatcher(const PDXearchWrapper &pdxearch_wrapper);

	/******************************************************************
	 * Getters
	 ******************************************************************/

	// Total cluster count summed across all (initialized) row groups. Used for EXPLAIN output.
	idx_t GetTotalNumClusters() const {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetTotalNumClusters();
		}
		return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetTotalNumClusters();
	}

	// Upper bound on the per-row-group cluster count, computed from a full row group. Useful for sizing scan-wide
	// quantities that must apply uniformly across row groups (the searcher clamps to each row group's actual count).
	static constexpr idx_t GetNumClustersForFullRowGroup() {
		return PDXearchWrapperF32::ComputeNumClustersForRowGroup(DEFAULT_ROW_GROUP_SIZE);
	}

	idx_t GetNumRowGroups() const {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetNumRowGroups();
		}
		return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetNumRowGroups();
	}

	string GetQuantization() const;

	idx_t GetNumDimensions() const {
		return pdxearch_wrapper->GetNumDimensions();
	}

	string GetDistanceMetric() const;

	bool IsNormalized() const {
		return pdxearch_wrapper->IsNormalized();
	}

	// N_probe precedence: runtime setting (pdxearch_n_probe) > index setting (n_probe) > default.
	idx_t GetEffectiveNProbe(const ClientContext &context) const {
		auto current_n_probe = static_cast<idx_t>(pdxearch_wrapper->GetNProbe());

		Value pdxearch_n_probe_opt;
		if (context.TryGetCurrentSetting("pdxearch_n_probe", pdxearch_n_probe_opt)) {
			if (!pdxearch_n_probe_opt.IsNull() && pdxearch_n_probe_opt.type() == LogicalType::INTEGER) {
				auto val = pdxearch_n_probe_opt.GetValue<int32_t>();
				if (val >= 0) {
					current_n_probe = static_cast<idx_t>(val);
				}
			}
		}

		return current_n_probe;
	}

	float *GetRotationMatrix() const {
		return pdxearch_wrapper->GetRotationMatrix();
	}
};

} // namespace duckdb
