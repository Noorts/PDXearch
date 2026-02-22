#include "pdxearch/common.hpp"
#include "index/pdxearch_index.hpp"

#include "index/pdxearch_module.hpp"

#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

PDXearchIndex::PDXearchIndex(const string &name, IndexConstraintType index_constraint_type,
                             const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                             const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                             const case_insensitive_map_t<Value> &index_creation_options,
                             const IndexStorageInfo &persistence_info, idx_t estimated_cardinality)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("PDXearch indexes do not support unique or primary key constraints");
	}

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	const auto &embedding_type = logical_types[0];
	D_ASSERT(embedding_type.id() == LogicalTypeId::ARRAY);

	const auto num_dimensions = ArrayType::GetSize(embedding_type);

	// Try to get the vector metric from the options, this parameter should be verified during binding.
	auto dist_metric = PDXearchWrapper::DEFAULT_DISTANCE_METRIC;
	const auto dist_metric_opt = index_creation_options.find("metric");
	if (dist_metric_opt != index_creation_options.end()) {
		const auto dist_metric_val =
		    PDXearchIndex::DISTANCE_METRIC_MAP.find(dist_metric_opt->second.GetValue<string>());
		if (dist_metric_val != PDXearchIndex::DISTANCE_METRIC_MAP.end()) {
			dist_metric = dist_metric_val->second;
		}
	}

	auto quantization = PDXearchWrapper::DEFAULT_QUANTIZATION;
	const auto quantization_opt = index_creation_options.find("quantization");
	if (quantization_opt != index_creation_options.end()) {
		const auto quantization_val = PDXearchIndex::QUANTIZATION_MAP.find(quantization_opt->second.GetValue<string>());
		if (quantization_val != PDXearchIndex::QUANTIZATION_MAP.end()) {
			quantization = quantization_val->second;
		}
	}

	auto n_probe = PDXearchWrapper::DEFAULT_N_PROBE;
	const auto n_probe_opt = index_creation_options.find("n_probe");
	if (n_probe_opt != index_creation_options.end()) {
		n_probe = n_probe_opt->second.GetValue<int32_t>();
	}

	// TODO: Confirm the static cast is sound.
	auto seed = static_cast<int32_t>(std::random_device {}());
	const auto seed_opt = index_creation_options.find("seed");
	if (seed_opt != index_creation_options.end()) {
		seed = seed_opt->second.GetValue<int32_t>();
	}

	if (quantization == PDX::Quantization::F32) {
		D_ASSERT(ArrayType::GetChildType(embedding_type).id() == LogicalTypeId::FLOAT);

#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
		pdxearch_wrapper =
		    make_uniq<PDXearchWrapperF32>(dist_metric, num_dimensions, n_probe, seed, estimated_cardinality);
#else
		pdxearch_wrapper =
		    make_uniq<PDXearchWrapperGlobalF32>(dist_metric, num_dimensions, n_probe, seed, estimated_cardinality);
#endif
	} else if (quantization == PDX::Quantization::U8) {
#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
		pdxearch_wrapper =
		    make_uniq<PDXearchWrapperU8>(dist_metric, num_dimensions, n_probe, seed, estimated_cardinality);
#else
		pdxearch_wrapper =
		    make_uniq<PDXearchWrapperGlobalU8>(dist_metric, num_dimensions, n_probe, seed, estimated_cardinality);
#endif
	} else {
		throw InternalException("Unsupported quantization: %s", quantization);
	}

	function_matcher = MakeFunctionMatcher(*pdxearch_wrapper.get());
}

/******************************************************************
 * Index creation and search methods specific to the parallel implementation
 ******************************************************************/

void PDXearchIndex::SetUpIndexForRowGroup(const row_t *const row_ids, const float *const vectors,
                                          const idx_t num_vectors, const idx_t row_group_id) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_group_id);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_group_id);
	}
}

void PDXearchIndex::InitializeSearchForRowGroup(float *const preprocessed_query, const idx_t limit,
                                                const idx_t row_group_id, PDX::Heap &heap, std::mutex &heap_mutex) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->InitializeSearchForRowGroup(preprocessed_query, limit, row_group_id, heap, heap_mutex);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->InitializeSearchForRowGroup(preprocessed_query, limit, row_group_id, heap, heap_mutex);
	}
}

void PDXearchIndex::SearchRowGroup(const idx_t row_group_id, const idx_t num_clusters_to_probe) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->SearchRowGroup(row_group_id, num_clusters_to_probe);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->SearchRowGroup(row_group_id, num_clusters_to_probe);
	}
}

void PDXearchIndex::InitializeFilteredSearchForRowGroup(float *const preprocessed_query, const idx_t limit,
                                                        const std::vector<row_t> &passing_row_ids,
                                                        const idx_t row_group_id, PDX::Heap &heap,
                                                        std::mutex &heap_mutex) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->InitializeFilteredSearchForRowGroup(preprocessed_query, limit, passing_row_ids, row_group_id, heap,
		                                          heap_mutex);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->InitializeFilteredSearchForRowGroup(preprocessed_query, limit, passing_row_ids, row_group_id, heap,
		                                          heap_mutex);
	}
}

void PDXearchIndex::FilteredSearchRowGroup(const idx_t row_group_id, const idx_t num_clusters_to_try_to_probe) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->FilteredSearchRowGroup(row_group_id, num_clusters_to_try_to_probe);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->FilteredSearchRowGroup(row_group_id, num_clusters_to_try_to_probe);
	}
}

/******************************************************************
 * Index creation and search methods specific to the global implementation
 ******************************************************************/

void PDXearchIndex::SetUpGlobalIndex(const row_t *const row_ids, const float *const embeddings,
                                     const idx_t num_embeddings) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperGlobalU8 *>(pdxearch_wrapper.get())
		    ->SetUpGlobalIndex(row_ids, embeddings, num_embeddings);
	} else {
		static_cast<PDXearchWrapperGlobalF32 *>(pdxearch_wrapper.get())
		    ->SetUpGlobalIndex(row_ids, embeddings, num_embeddings);
	}
}

struct PDXearchIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	std::unique_ptr<std::vector<row_t>> row_ids;
};

unique_ptr<IndexScanState> PDXearchIndex::InitializeGlobalScan(const float *const query_embedding, const idx_t limit,
                                                               const ClientContext &context) {
	auto state = make_uniq<PDXearchIndexScanState>();

	const auto n_probe = GetEffectiveNProbe(context);
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		state->row_ids =
		    static_cast<PDXearchWrapperGlobalU8 *>(pdxearch_wrapper.get())->Search(query_embedding, limit, n_probe);
	} else {
		state->row_ids =
		    static_cast<PDXearchWrapperGlobalF32 *>(pdxearch_wrapper.get())->Search(query_embedding, limit, n_probe);
	}

	return std::move(state);
}

idx_t PDXearchIndex::GlobalScan(IndexScanState &state, Vector &result, const idx_t result_offset) {
	auto &scan_state = state.Cast<PDXearchIndexScanState>();

	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;

	// Push the row ids into the result vector, up to STANDARD_VECTOR_SIZE or the
	// end of the result set
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.row_ids->size()) {
		row_ids[count++] = (*scan_state.row_ids)[scan_state.current_row++];
	}

	return count;
}

std::unique_ptr<std::vector<row_t>>
PDXearchIndex::GlobalFilteredSearch(const float *const query_embedding, const idx_t limit,
                                    std::vector<std::pair<Vector, idx_t>> &collected_embeddings,
                                    const ClientContext &context) {
	const auto n_probe = GetEffectiveNProbe(context);
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperGlobalU8 *>(pdxearch_wrapper.get())
		    ->FilteredSearch(query_embedding, limit, collected_embeddings, n_probe);
	}
	return static_cast<PDXearchWrapperGlobalF32 *>(pdxearch_wrapper.get())
	    ->FilteredSearch(query_embedding, limit, collected_embeddings, n_probe);
}

/******************************************************************
 * Index maintenance
 ******************************************************************/

ErrorData PDXearchIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	throw NotImplementedException("PDXearchIndex::Append() not implemented");
}

ErrorData PDXearchIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	throw NotImplementedException("PDXearchIndex::Insert() not implemented");
}

void PDXearchIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	throw NotImplementedException("PDXearchIndex::Delete() not implemented");
}

void PDXearchIndex::CommitDrop(IndexLock &lock) {
}

bool PDXearchIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("PDXearchIndex::MergeIndexes() not implemented");
}

void PDXearchIndex::Vacuum(IndexLock &state) {
}

string PDXearchIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	throw NotImplementedException("PDXearchIndex::VerifyAndToString() not implemented");
}

void PDXearchIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("PDXearchIndex::VerifyAllocations() not implemented");
}

idx_t PDXearchIndex::GetInMemorySize(IndexLock &state) {
	return pdxearch_wrapper->GetInMemorySizeInBytes();
}

/******************************************************************
 * Index persistence
 ******************************************************************/

IndexStorageInfo PDXearchIndex::SerializeToDisk(QueryContext context,
                                                const case_insensitive_map_t<Value> &serialization_options) {
	// For serialization_options see:
	// https://github.com/duckdb/duckdb/blob/32afee3e788394973ce4df4fcae7610832d5550a/src/storage/write_ahead_log.cpp#L374

	IndexStorageInfo info(name);
	case_insensitive_map_t<Value> options;
	options.emplace("testDisk", Value::INTEGER(12));
	info.options = options;

	// Temporary empty FixedSizeAllocatorInfo to satisfy the DuckDB RelDebug build's index_storage_info.IsValid() check.
	info.allocator_infos.push_back(FixedSizeAllocatorInfo {});

	return info;
}

IndexStorageInfo PDXearchIndex::SerializeToWAL(const case_insensitive_map_t<Value> &serialization_options) {
	IndexStorageInfo info(name);
	case_insensitive_map_t<Value> options;
	options.emplace("testWAL", Value::INTEGER(12));
	info.options = options;

	// Temporary empty FixedSizeAllocatorInfo to satisfy the DuckDB RelDebug build's index_storage_info.IsValid() check.
	info.allocator_infos.push_back(FixedSizeAllocatorInfo {});

	return info;
}

// TODO: Implement persistence.
void PDXearchIndex::PersistToDisk() {
}

/******************************************************************
 * Misc.
 ******************************************************************/

bool PDXearchIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                             vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

static void TryBindIndexExpressionInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {

	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		// Rewrite the column reference to fit in the current set of bound column ids
		ref.binding.table_index = table_idx;

		const auto referenced_column = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

bool PDXearchIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

string PDXearchIndex::GetDistanceMetric() const {
	switch (pdxearch_wrapper->GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		return "l2sq";
	case PDX::DistanceMetric::COSINE:
		return "cosine";
	// case PDX::DistanceMetric::IP:
	// 	return "ip";
	default:
		throw InternalException("Unknown distance metric");
	}
}

const case_insensitive_map_t<PDX::DistanceMetric> PDXearchIndex::DISTANCE_METRIC_MAP = {
    {"l2sq", PDX::DistanceMetric::L2SQ}, {"cosine", PDX::DistanceMetric::COSINE},
    // {"ip", PDX::DistanceMetric::IP},
};

const case_insensitive_map_t<PDX::Quantization> PDXearchIndex::QUANTIZATION_MAP = {
    {"f32", PDX::Quantization::F32},
    {"u8", PDX::Quantization::U8},
};

unique_ptr<ExpressionMatcher> PDXearchIndex::MakeFunctionMatcher(const PDXearchWrapper &pdxearch_wrapper) {
	unordered_set<string> distance_functions;

	switch (pdxearch_wrapper.GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		distance_functions = {"array_distance", "<->"};
		break;
	case PDX::DistanceMetric::COSINE:
		distance_functions = {"array_cosine_distance", "<=>"};
		break;
	// case PDX::DistanceMetric::IP:
	// 	distance_functions = {"array_negative_inner_product", "<#>"};
	//  break;
	default:
		throw NotImplementedException("Unknown distance metric");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type =
	    make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, pdxearch_wrapper.GetNumDimensions()));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type =
	    make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, pdxearch_wrapper.GetNumDimensions()));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

void PDXearchModule::RegisterIndex(DatabaseInstance &db) {

	IndexType index_type;

	index_type.name = PDXearchIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<PDXearchIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                    input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = PDXearchIndex::CreatePlan;

	db.config.AddExtensionOption("pdxearch_n_probe",
	                             "override the n_probe parameter when scanning PDXearch indexes (default: " +
	                                 to_string(PDXearchWrapper::DEFAULT_N_PROBE) + ", must be >= 0)",
	                             LogicalType::INTEGER, Value());

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb
