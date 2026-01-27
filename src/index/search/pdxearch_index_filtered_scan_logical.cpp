#include "index/search/pdxearch_index_filtered_scan_logical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
#include "index/search/pdxearch_index_filtered_scan_physical.hpp"
#else
#include "index/search/pdxearch_index_global_filtered_scan_physical.hpp"
#endif

namespace duckdb {

PhysicalOperator &LogicalPDXearchIndexFilteredScan::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
	auto bind_data = make_uniq<PDXearchIndexPhysicalScanBindData>(table, index, limit, std::move(query_embedding));
	auto &physical_op = planner.Make<PhysicalPDXearchIndexFilteredScan>(types, std::move(bind_data), column_ids, limit);
#else
	auto bind_data =
	    make_uniq<GlobalPDXearchIndexPhysicalScanBindData>(table, index, limit, std::move(query_embedding));
	auto &physical_op =
	    planner.Make<PhysicalGlobalPDXearchIndexFilteredScan>(types, std::move(bind_data), column_ids, limit);
#endif

	// The child operator should output row_ids (from a modified Get)
	if (!children.empty()) {
		auto &child = planner.CreatePlan(*children[0]);
		physical_op.children.push_back(child);
	}

	return physical_op;
}

vector<ColumnBinding> LogicalPDXearchIndexFilteredScan::GetColumnBindings() {
	return GenerateColumnBindings(table_index, column_ids.size());
}

void LogicalPDXearchIndexFilteredScan::ResolveTypes() {
	types.clear();
	types.reserve(column_ids.size());
	for (auto &col_idx : column_ids) {
		if (col_idx.IsRowIdColumn()) {
			types.push_back(LogicalType::ROW_TYPE);
		} else {
			auto logical_idx = col_idx.GetPrimaryIndex();
			auto &col = table.GetColumn(LogicalIndex(logical_idx));
			types.push_back(col.Type());
		}
	}
}

} // namespace duckdb
