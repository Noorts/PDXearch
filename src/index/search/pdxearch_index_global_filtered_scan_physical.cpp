#include "index/search/pdxearch_index_global_filtered_scan_physical.hpp"
#include "index/pdxearch_index.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

class PhysicalGlobalFilteredScanGlobalSinkState : public GlobalSinkState {
public:
	PhysicalGlobalFilteredScanGlobalSinkState() : collected_vectors(), pdxearch_row_ids(nullptr) {
	}

	std::vector<std::pair<Vector, idx_t>> collected_vectors;
	std::unique_ptr<std::vector<row_t>> pdxearch_row_ids;
};

class PhysicalGlobalFilteredScanLocalSinkState : public LocalSinkState {
public:
	PhysicalGlobalFilteredScanLocalSinkState() : local_vectors() {
	}

	std::vector<std::pair<Vector, idx_t>> local_vectors;
};

class PhysicalGlobalFilteredScanGlobalSourceState : public GlobalSourceState {
public:
	PhysicalGlobalFilteredScanGlobalSourceState()
	    : fetch_state(), local_storage_state(), column_ids(), current_result_idx(0) {
	}

	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	// Current position in pdxearch_row_ids when emitting fetched rows. Used in
	// case the operator has to emit multiple chunks of results.
	idx_t current_result_idx;
};

class PhysicalGlobalFilteredScanLocalSourceState : public LocalSourceState {};

PhysicalGlobalPDXearchIndexFilteredScan::PhysicalGlobalPDXearchIndexFilteredScan(
    PhysicalPlan &physical_plan, vector<LogicalType> types,
    unique_ptr<GlobalPDXearchIndexPhysicalScanBindData> bind_data, vector<ColumnIndex> column_ids,
    idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalGlobalPDXearchIndexFilteredScan::TYPE, std::move(types),
                       estimated_cardinality),
      bind_data(std::move(bind_data)), column_ids(std::move(column_ids)) {
}

// ------------------------------
// Sink interface
// ------------------------------

unique_ptr<GlobalSinkState> PhysicalGlobalPDXearchIndexFilteredScan::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<PhysicalGlobalFilteredScanGlobalSinkState>();
}

unique_ptr<LocalSinkState> PhysicalGlobalPDXearchIndexFilteredScan::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PhysicalGlobalFilteredScanLocalSinkState>();
}

SinkResultType PhysicalGlobalPDXearchIndexFilteredScan::Sink(ExecutionContext &context, DataChunk &input_chunk,
                                                             OperatorSinkInput &input) const {
	auto &l_sink = input.local_state.Cast<PhysicalGlobalFilteredScanLocalSinkState>();

	if (input_chunk.size() == 0) {
		return SinkResultType::FINISHED;
	}

	// Validate input chunk.
	D_ASSERT(input_chunk.ColumnCount() == 1);
	D_ASSERT(input_chunk.data[0].GetType() == LogicalType::ROW_TYPE);

	// Collect row ids into the local state.
	Vector copied_vector(input_chunk.data[0].GetType(), input_chunk.size());
	VectorOperations::Copy(input_chunk.data[0], copied_vector, input_chunk.size(), 0, 0);
	l_sink.local_vectors.emplace_back(std::move(copied_vector), input_chunk.size());

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalGlobalPDXearchIndexFilteredScan::Combine(ExecutionContext &context,
                                                                       OperatorSinkCombineInput &input) const {
	auto &g_sink = input.global_state.Cast<PhysicalGlobalFilteredScanGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<PhysicalGlobalFilteredScanLocalSinkState>();

	// Move all collected Vectors from local state to global state.
	const auto guard = g_sink.Lock();
	g_sink.collected_vectors.reserve(g_sink.collected_vectors.size() + l_sink.local_vectors.size());
	for (auto &vec_pair : l_sink.local_vectors) {
		g_sink.collected_vectors.emplace_back(std::move(vec_pair));
	}

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalGlobalPDXearchIndexFilteredScan::Finalize(Pipeline &pipeline, Event &event,
                                                                   ClientContext &context,
                                                                   OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<PhysicalGlobalFilteredScanGlobalSinkState>();

	if (g_sink.collected_vectors.empty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	g_sink.pdxearch_row_ids = bind_data->index.Cast<PDXearchIndex>().GlobalFilteredSearch(
	    bind_data->query_embedding.get(), bind_data->limit, g_sink.collected_vectors, context);

	return SinkFinalizeType::READY;
}

// ------------------------------
// Source interface
// ------------------------------

unique_ptr<GlobalSourceState>
PhysicalGlobalPDXearchIndexFilteredScan::GetGlobalSourceState(ClientContext &context) const {
	auto g_source = make_uniq<PhysicalGlobalFilteredScanGlobalSourceState>();

	// Set up column ids for fetching data from storage.
	auto &bind_data_ref = *bind_data;
	g_source->column_ids.reserve(column_ids.size());
	for (auto &id : column_ids) {
		StorageIndex storage_id;
		if (id.IsRowIdColumn()) {
			storage_id = StorageIndex();
		} else {
			auto &col = bind_data_ref.table.GetColumn(LogicalIndex(id.GetPrimaryIndex()));
			storage_id = StorageIndex(col.StorageOid());
		}
		g_source->column_ids.emplace_back(storage_id);
	}

	// Initialize the storage scan state.
	g_source->local_storage_state.Initialize(g_source->column_ids, context, nullptr);
	auto &local_storage = LocalStorage::Get(context, bind_data_ref.table.catalog);
	local_storage.InitializeScan(bind_data_ref.table.GetStorage(), g_source->local_storage_state.local_state, nullptr);

	return std::move(g_source);
}

unique_ptr<LocalSourceState>
PhysicalGlobalPDXearchIndexFilteredScan::GetLocalSourceState(ExecutionContext &context,
                                                             GlobalSourceState &g_source) const {
	return make_uniq<PhysicalGlobalFilteredScanLocalSourceState>();
}

SourceResultType PhysicalGlobalPDXearchIndexFilteredScan::GetData(ExecutionContext &context, DataChunk &output_chunk,
                                                                  OperatorSourceInput &input) const {
	auto &g_sink = sink_state->Cast<PhysicalGlobalFilteredScanGlobalSinkState>();
	auto &g_source = input.global_state.Cast<PhysicalGlobalFilteredScanGlobalSourceState>();

	D_ASSERT(g_sink.pdxearch_row_ids);
	D_ASSERT(g_source.current_result_idx <= g_sink.pdxearch_row_ids->size());

	const idx_t num_results_to_emit =
	    MinValue<idx_t>(STANDARD_VECTOR_SIZE, g_sink.pdxearch_row_ids->size() - g_source.current_result_idx);

	if (num_results_to_emit == 0) {
		return SourceResultType::FINISHED;
	}

	// Create vector of row ids that are part of the current output chunk.
	Vector row_ids_vector(LogicalType::ROW_TYPE, num_results_to_emit);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vector);
	for (idx_t i = 0; i < num_results_to_emit; i++) {
		row_ids_data[i] = g_sink.pdxearch_row_ids->at(g_source.current_result_idx + i);
	}
	g_source.current_result_idx += num_results_to_emit;

	// Fetch the data from storage.
	auto &transaction = DuckTransaction::Get(context.client, bind_data->table.catalog);
	bind_data->table.GetStorage().Fetch(transaction, output_chunk, g_source.column_ids, row_ids_vector,
	                                    num_results_to_emit, g_source.fetch_state);
	D_ASSERT(output_chunk.size() == num_results_to_emit);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalGlobalPDXearchIndexFilteredScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Table"] = bind_data->table.name;
	result["PDXearch Index"] = bind_data->index.GetIndexName();
	result["Normalized"] = bind_data->index.Cast<PDXearchIndex>().IsNormalized() ? "true" : "false";
	result["Clusters"] = StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumClusters());
	const idx_t index_in_memory_size = bind_data->index.Cast<BoundIndex>().GetInMemorySize();
	result["Index Size"] = ConvertBytesToHumanReadableString(index_in_memory_size);
	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb
