#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

class LogicalPDXearchIndexScan : public LogicalExtensionOperator {
public:
	LogicalPDXearchIndexScan(DuckTableEntry &table, Index &index, idx_t limit,
	                         unsafe_unique_array<float> query_embedding, vector<ColumnIndex> column_ids,
	                         vector<ColumnBinding> column_bindings)
	    : LogicalExtensionOperator(), column_ids(std::move(column_ids)), column_bindings(std::move(column_bindings)),
	      table(table), index(index), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	// The table columns fetched by rowid for the results, and the bindings they are emitted under: those the operator
	// above reads. To support View or subquery without a predicate.
	vector<ColumnIndex> column_ids;
	vector<ColumnBinding> column_bindings;

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
