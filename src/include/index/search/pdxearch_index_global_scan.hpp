#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"

namespace duckdb {

class Index;

struct PDXearchIndexScanBindData final : public TableScanBindData {
	explicit PDXearchIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit,
	                                   unsafe_unique_array<float> query_embedding)
	    : TableScanBindData(table), index(index), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	Index &index;
	// The limit (the K in KNN search).
	const idx_t limit;
	unsafe_unique_array<float> query_embedding;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<PDXearchIndexScanBindData>();
		return &other.table == &table;
	}
};

struct PDXearchIndexScanFunction {
	static TableFunction GetFunction();
};

} // namespace duckdb
