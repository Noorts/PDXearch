#include "index/search/pdxearch_index_scan_logical.hpp"
#include "index/search/pdxearch_index_scan_physical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {

PhysicalOperator &LogicalPDXearchIndexScan::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	auto bind_data = make_uniq<PDXearchIndexScanBindData>(table, index, limit, std::move(query_embedding));
	auto &physical_op = planner.Make<PhysicalPDXearchIndexScan>(types, std::move(bind_data), column_ids, limit);

	// This is a Source-only operator.
	D_ASSERT(children.empty());

	return physical_op;
}

vector<ColumnBinding> LogicalPDXearchIndexScan::GetColumnBindings() {
	return GenerateColumnBindings(table_index, column_ids.size());
}

void LogicalPDXearchIndexScan::ResolveTypes() {
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
