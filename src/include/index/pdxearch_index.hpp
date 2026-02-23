#pragma once

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "pdxearch/common.hpp"
#include "index/pdxearch_wrapper.hpp"

namespace duckdb {

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

	unique_ptr<ExpressionMatcher> function_matcher;
	IndexPointer root_block_ptr;

public:
	PDXearchIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	              TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	              AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	              const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	/******************************************************************
	 * Index creation and search methods specific to the parallel implementation
	 ******************************************************************/

	void SetUpIndexForRowGroup(const row_t *row_ids, const float *embeddings, idx_t num_embeddings, idx_t row_group_id);

	void InitializeSearchForRowGroup(float *preprocessed_query_embedding, idx_t limit, idx_t row_group_id,
	                                 PDX::Heap &heap, std::mutex &heap_mutex);

	void SearchRowGroup(idx_t row_group_id, idx_t num_clusters_to_probe);

	void InitializeFilteredSearchForRowGroup(float *preprocessed_query_embedding, idx_t limit,
	                                         const std::vector<row_t> &passing_row_ids, idx_t row_group_id,
	                                         PDX::Heap &heap, std::mutex &heap_mutex);

	void FilteredSearchRowGroup(idx_t row_group_id, idx_t num_clusters_to_try_to_probe);

	/******************************************************************
	 * Index creation and search methods specific to the global implementation
	 ******************************************************************/

	void SetUpGlobalIndex(const row_t *row_ids, const float *embeddings, idx_t num_embeddings);

	[[nodiscard]] unique_ptr<IndexScanState> InitializeGlobalScan(const float *query_embedding, idx_t limit,
	                                                              const ClientContext &context);

	idx_t GlobalScan(IndexScanState &state, Vector &result, idx_t result_offset = 0);

	[[nodiscard]] std::unique_ptr<std::vector<row_t>>
	GlobalFilteredSearch(const float *query_embedding, idx_t limit,
	                     std::vector<std::pair<Vector, idx_t>> &collected_embeddings, const ClientContext &context);

	/******************************************************************
	 * Index maintenance
	 ******************************************************************/

	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	void CommitDrop(IndexLock &lock) override;

	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;

	void Vacuum(IndexLock &state) override;

	string VerifyAndToString(IndexLock &state, const bool only_verify) override;

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

	idx_t GetNumClustersPerRowGroup() const {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetNumClustersPerRowGroup();
		}
		return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetNumClustersPerRowGroup();
	}

	idx_t GetNumRowGroups() const {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetNumRowGroups();
		}
		return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetNumRowGroups();
	}

	idx_t GetNumClusters() const {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<PDXearchWrapperGlobalU8 *>(pdxearch_wrapper.get())->GetNumClusters();
		}
		return static_cast<PDXearchWrapperGlobalF32 *>(pdxearch_wrapper.get())->GetNumClusters();
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
