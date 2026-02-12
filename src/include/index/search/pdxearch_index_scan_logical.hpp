#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

class LogicalPDXearchIndexScan : public LogicalExtensionOperator {
public:
	LogicalPDXearchIndexScan(DuckTableEntry &table, Index &index, idx_t limit,
	                         unsafe_unique_array<float> query_embedding, vector<ColumnIndex> column_ids,
	                         idx_t table_index)
	    : LogicalExtensionOperator(), column_ids(std::move(column_ids)), table_index(table_index), table(table),
	      index(index), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	vector<ColumnIndex> column_ids;
	idx_t table_index;

	DuckTableEntry &table;
	Index &index;
	// The limit (the K in KNN search).
	const idx_t limit;
	unsafe_unique_array<float> query_embedding;

public:
	string GetName() const override {
		return "PDXEARCH_INDEX_SCAN";
	}

	void ResolveTypes() override;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

	vector<ColumnBinding> GetColumnBindings() override;
};

} // namespace duckdb
