#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/storage/data_table.hpp"

#include "index/pdxearch_module.hpp"
#include "index/pdxearch_index.hpp"
#include "index/search/pdxearch_index_global_scan.hpp"

namespace duckdb {

BindInfo PDXearchIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<PDXearchIndexScanBindData>();
	return BindInfo(bind_data.table);
}

struct PDXearchIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	unique_ptr<IndexScanState> index_state;
	Vector row_ids_vector = Vector(LogicalType::ROW_TYPE);
};

static unique_ptr<GlobalTableFunctionState> PDXearchIndexScanInitGlobal(ClientContext &context,
                                                                        TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<PDXearchIndexScanBindData>();

	auto result = make_uniq<PDXearchIndexScanGlobalState>();

	// Set up the scan state for the local storage
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	result->column_ids.reserve(input.column_ids.size());

	// Figure out the storage column ids
	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.emplace_back(col_id);
	}

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->column_ids, context, input.filters);
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	// Initialize the scan state for the index
	result->index_state = bind_data.index.Cast<PDXearchIndex>().InitializeGlobalScan(bind_data.query_embedding.get(),
	                                                                                 bind_data.limit, context);

	return std::move(result);
}

static void PDXearchIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	auto &bind_data = data_p.bind_data->Cast<PDXearchIndexScanBindData>();
	auto &state = data_p.global_state->Cast<PDXearchIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	// Scan the index for row ids
	const auto row_count = bind_data.index.Cast<PDXearchIndex>().GlobalScan(*state.index_state, state.row_ids_vector);
	if (row_count == 0) {
		// Short-circuit if the index had no more rows
		output.SetCardinality(0);
		return;
	}

	// Fetch the data from the local storage given the row ids
	bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids_vector, row_count,
	                                   state.fetch_state);
}

unique_ptr<NodeStatistics> PDXearchIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<PDXearchIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit, bind_data.limit);
}

static InsertionOrderPreservingMap<string> PDXearchIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<PDXearchIndexScanBindData>();
	auto &pdxearch_index = bind_data.index.Cast<PDXearchIndex>();

	result["Table"] = bind_data.table.name;
	result["PDXearch Index"] = bind_data.index.GetIndexName();
	result["Clusters"] = StringUtil::Format("%zu", pdxearch_index.GetNumClusters());

	return result;
}

TableFunction PDXearchIndexScanFunction::GetFunction() {
	TableFunction func("pdxearch_index_scan", {}, PDXearchIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = PDXearchIndexScanInitGlobal;
	func.statistics = nullptr;
	func.dependency = nullptr;
	func.cardinality = PDXearchIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = PDXearchIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = PDXearchIndexScanBindInfo;

	return func;
}

void PDXearchModule::RegisterIndexScan(ExtensionLoader &loader) {
	loader.RegisterFunction(PDXearchIndexScanFunction::GetFunction());
}

} // namespace duckdb
