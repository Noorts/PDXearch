#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"

#include "index/pdxearch_blob_codec.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "index/pdxearch_module.hpp"
#include "index/pdxearch_index.hpp"
#include "index/search/pdxearch_index_filtered_scan_logical.hpp"
#include "index/search/pdxearch_index_scan_logical.hpp"
#include "index/search/pdxearch_index_scan_physical.hpp"

namespace duckdb {

/**
 * Optimizes various vector similarity search queries, where the table has a
 * matching PDXearch index.
 *
 * Note: this optimizer sets `optimize_function` and thus runs after DuckDB's
 * optimization passes.
 *
 ********************** SCENARIO 1 (non-filtered search): **********************
 *
 * Example query (on a table with an id and a vec column):
 *   SELECT * FROM table
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * Scenario 1 targets the logical query plan shown on the left (this has been
 * optimized by DuckDB's optimization passes, but has not been optimized by this
 * optimizer yet; e.g., print TryOptimize's `plan`). This optimizer replaces the
 * TopN operator and the sequential scan with a PDXearch index scan, turning it
 * into the logical query plan shown on the right.
 *
 *              [IN]                                [OUT]
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │          ~10 rows         │        │        Expressions:       │
 * └─────────────┬─────────────┘        │             id            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │      ~1,000,000 rows      │
 * │             id            │        └─────────────┬─────────────┘
 * │            vec            │        ┌─────────────┴─────────────┐
 * │    array_distance(vec,    │        │    PDXEARCH_INDEX_SCAN    │
 * │     [1.0, 2.0, 3.0])      │        │    ────────────────────   │
 * │                           │        │        Table: table       │
 * │      ~1,000,000 rows      │        │    PDXearch Index: idx    │
 * └─────────────┬─────────────┘        │       Clusters: 4000      │
 * ┌─────────────┴─────────────┐        │                           │
 * │          SEQ_SCAN         │        │        Projections:       │
 * │    ────────────────────   │        │             id            │
 * │        Table: table       │        │            vec            │
 * │   Type: Sequential Scan   │        │                           │
 * │                           │        │      ~1,000,000 rows      │
 * │        Projections:       │        └───────────────────────────┘
 * │             id            │
 * │            vec            │
 * │                           │
 * │      ~1,000,000 rows      │
 * └───────────────────────────┘
 *
 *
 ******************** SCENARIO 2 (simple filtered search): *********************
 *
 * Simple (non-composite) filters pushed down into the table scan.
 *
 * Example query (on a table with an id and a vec column):
 *   SELECT * FROM table
 *   WHERE id > 50
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * Scenario 2 targets the logical query plan shown on the left (this has
 * been optimized by DuckDB's optimization passes, but has not been optimized by
 * this optimizer yet). This optimizer produces the logical query plan shown on
 * the right. Projections are shown for clarity (normally not part of the
 * logical plan).
 *
 *              [IN]                                [OUT]
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │          ~10 rows         │        │        Expressions:       │
 * └─────────────┬─────────────┘        │             id            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │      ~1,000,000 rows      │
 * │             id            │        └─────────────┬─────────────┘
 * │            vec            │        ┌─────────────┴─────────────┐
 * │    array_distance(vec,    │        │  PDXEARCH_INDEX_FILT_SCAN │
 * │     [1.0, 2.0, 3.0])      │        │    ────────────────────   │
 * │                           │        │        Table: table       │
 * │      ~1,000,000 rows      │        │    PDXearch Index: idx    │
 * └─────────────┬─────────────┘        │       Clusters: 4000      │
 * ┌─────────────┴─────────────┐        │                           │
 * │          SEQ_SCAN         │        │        Projections:       │
 * │    ────────────────────   │        │             id            │
 * │        Table: table       │        │            vec            │
 * │   Type: Sequential Scan   │        │                           │
 * │                           │        │      ~1,000,000 rows      │
 * │        Projections:       │        └─────────────┬─────────────┘
 * │             id            │        ┌─────────────┴─────────────┐
 * │            vec            │        │          SEQ_SCAN         │
 * │                           │        │    ────────────────────   │
 * │       Filters: id>50      │        │        Table: table       │
 * │                           │        │   Type: Sequential Scan   │
 * │      ~1,000,000 rows      │        │                           │
 * └───────────────────────────┘        │        Projections:       │
 *                                      │           rowid           │
 *                                      │                           │
 *                                      │       Filters: id>50      │
 *                                      │                           │
 *                                      │      ~1,000,000 rows      │
 *                                      └───────────────────────────┘
 *
 *
 ****************** SCENARIO 3 (residual filtered search): *********************
 *
 * Predicates DuckDB cannot push into the table scan (e.g. an OR across two
 * columns, or an expression over two columns) stay in FILTER operators above
 * it. Scenario 3 handles one or more FILTER operators between the projection
 * and the table scan (which may have pushed-down filters as well).
 *
 * Example query (on a table with an id, a cat and a vec column):
 *   SELECT * FROM table
 *   WHERE id > 50 OR cat = 3
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * The filters move below the PDXEARCH_INDEX_FILT_SCAN unchanged. The table scan
 * and every filter forward the rowid, and a projection on top keeps only the
 * rowid, which the filtered search consumes. When no predicate reads the
 * embedding, the rowid takes the embedding's slot in the table scan, so the
 * filter pipeline does not read embeddings. The filtered search fetches the
 * table scan's original columns by rowid for its K results.
 *
 * Subqueries and views (DuckDB binds a view as a subquery) add projections that
 * forward the table's columns. They may sit anywhere in the chain between the
 * projection and the table scan, mixed with the FILTER operators, and the
 * rowid is forwarded through them as well. The index scans emit the columns
 * the projection reads under the bindings it reads them with, so a chain
 * without FILTER operators is replaced entirely (scenarios 1 and 2). A query
 * that reads a value the chain computes (e.g. `id * 2 AS x` of a subquery) is
 * not optimized. Projections that only forward the distance (`ORDER BY d` over
 * a subquery that computes d) stay above the search.
 *
 * DuckDB turns an IN list of at least five constants that stays in a FILTER
 * into a MARK join between the filter's child and a scan of the constants, and
 * the FILTER tests the join's marker (e.g. `cat = 3 OR id IN (...)`). The MARK
 * join keeps every row of its left child, so it is part of the chain as well:
 * it stays below the search, evaluates the list, and forwards the rowid. A
 * query that reads the marker as a value (IN in the SELECT list) is not
 * optimized.
 *
 * Subqueries used as filters join the chain the same way when the indexed
 * table is the join's left child (the probe side): IN and EXISTS become a SEMI
 * join, NOT EXISTS an ANTI join, and an IN subquery inside a larger predicate
 * (e.g. `NOT IN (SELECT ...)`) a MARK join. These joins keep each row of the
 * table at most once, and keep the runtime filters DuckDB pushes from them into
 * the table scan. When DuckDB builds the hash table on the indexed table
 * instead (RIGHT_SEMI / RIGHT_ANTI), or leaves a correlated subquery as a
 * DELIM_JOIN, the query is not optimized.
 *
 *              IN                                  OUT
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │           #[1.2]          │        │             #2            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │                           │        │        Expressions:       │
 * │          ~10 rows         │        │             id            │
 * └─────────────┬─────────────┘        │            cat            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │          ~10 rows         │
 * │             id            │        └─────────────┬─────────────┘
 * │            cat            │        ┌─────────────┴─────────────┐
 * │            vec            │        │  PDXEARCH_INDEX_FILT_SCAN │
 * │    array_distance(vec,    │        │    ────────────────────   │
 * │      [1.0, 2.0, 3.0])     │        │        Table: table       │
 * │                           │        │    PDXearch Index: idx    │
 * │       ~550,000 rows       │        │                           │
 * └─────────────┬─────────────┘        │        Projections:       │
 * ┌─────────────┴─────────────┐        │             id            │
 * │           FILTER          │        │            cat            │
 * │    ────────────────────   │        │            vec            │
 * │        Expressions:       │        │                           │
 * │  ((id > 50) OR (cat = 3)) │        │          ~10 rows         │
 * │                           │        └─────────────┬─────────────┘
 * │       ~550,000 rows       │        ┌─────────────┴─────────────┐
 * └─────────────┬─────────────┘        │         PROJECTION        │
 * ┌─────────────┴─────────────┐        │    ────────────────────   │
 * │          SEQ_SCAN         │        │        Expressions:       │
 * │    ────────────────────   │        │           rowid           │
 * │        Table: table       │        │                           │
 * │   Type: Sequential Scan   │        │       ~550,000 rows       │
 * │                           │        └─────────────┬─────────────┘
 * │        Projections:       │        ┌─────────────┴─────────────┐
 * │             id            │        │           FILTER          │
 * │            cat            │        │    ────────────────────   │
 * │            vec            │        │        Expressions:       │
 * │                           │        │  ((id > 50) OR (cat = 3)) │
 * │      ~1,000,000 rows      │        │                           │
 * └───────────────────────────┘        │       ~550,000 rows       │
 *                                      └─────────────┬─────────────┘
 *                                      ┌─────────────┴─────────────┐
 *                                      │          SEQ_SCAN         │
 *                                      │    ────────────────────   │
 *                                      │        Table: table       │
 *                                      │   Type: Sequential Scan   │
 *                                      │                           │
 *                                      │        Projections:       │
 *                                      │             id            │
 *                                      │            cat            │
 *                                      │           rowid           │
 *                                      │                           │
 *                                      │      ~1,000,000 rows      │
 *                                      └───────────────────────────┘
 *
 * TODO: Leave out the PROJECTION at the top? Our pattern recognition starts at
 * the TopN operator.
 * TODO: Fix cardinality of after-optimization plan (K instead of 1M?).
 *       Also see filtered scan optimizer.
 */
class PDXearchIndexScanOptimizer : public OptimizerExtension {
public:
	PDXearchIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	// The operators the chain between the projection and the table scan may hold, each continuing into children[0]:
	// FILTER operators, projections (subqueries and views), and the joins that keep each row of their left child at
	// most once. DuckDB plans IN and EXISTS subqueries as SEMI joins and NOT EXISTS as an ANTI join. A MARK join keeps
	// every row of its left child and adds a marker column, which a FILTER above it tests: DuckDB makes one of an IN
	// list of at least five constants (the right child is then a column data scan) and of an IN subquery inside a
	// larger predicate (e.g. `NOT IN (SELECT ...)`, `cat = 3 OR id IN (SELECT ...)`).
	static bool IsChainOperator(const LogicalOperator &op) {
		switch (op.type) {
		// To support: OR or sparse IN on one column, Predicates over two or more columns, Conjunction left after
		// pushdown, Volatile predicate, Selective residual filter, Filter reading the embedding, Filter on the rowid,
		// Subquery or view with a residual filter.
		case LogicalOperatorType::LOGICAL_FILTER:
		// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate,
		// Renamed embedding.
		case LogicalOperatorType::LOGICAL_PROJECTION:
			return op.children.size() == 1;
		// To support: IN list of at least five constants and IN lists combined with other predicates (MARK), Subqueries
		// as filters, our table on the probe side (SEMI, ANTI, MARK).
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
			const auto join_type = op.Cast<LogicalComparisonJoin>().join_type;
			return op.children.size() == 2 &&
			       (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK);
		}
		default:
			return false;
		}
	}

	// Follows a binding read above chain.front() down the chain (ordered from the top down to the table scan) to the
	// table scan's binding it comes from. A filter forwards its child's bindings, and a projection forwards a binding
	// when its expression is a plain column reference. False when a projection computes the value instead.
	// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate,
	// Renamed embedding.
	static bool TryTraceToTableScan(const vector<reference<LogicalOperator>> &chain, ColumnBinding &binding) {
		for (auto &op : chain) {
			if (op.get().type != LogicalOperatorType::LOGICAL_PROJECTION) {
				continue;
			}
			auto &projection = op.get().Cast<LogicalProjection>();
			if (binding.table_index != projection.table_index ||
			    binding.column_index >= projection.expressions.size()) {
				return false;
			}
			auto &expression = *projection.expressions[binding.column_index];
			if (expression.type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			binding = expression.Cast<BoundColumnRefExpression>().binding;
		}
		return true;
	}

	// Collects the bindings the projection reads from its child, and the table scan's column each one traces to: an
	// index scan replacing the projection's child emits those columns, fetched by rowid, under those bindings. False
	// when the projection reads a value that the chain computes.
	// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate.
	static bool TryCollectOutputColumns(LogicalProjection &projection, const vector<reference<LogicalOperator>> &chain,
	                                    const LogicalGet &get, vector<ColumnBinding> &bindings,
	                                    vector<ColumnIndex> &column_ids) {
		bool all_traced = true;
		for (auto &expression : projection.expressions) {
			ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
				if (child.type != ExpressionType::BOUND_COLUMN_REF) {
					return;
				}
				const auto binding = child.Cast<BoundColumnRefExpression>().binding;
				if (std::find(bindings.begin(), bindings.end(), binding) != bindings.end()) {
					return;
				}
				auto traced_binding = binding;
				if (!TryTraceToTableScan(chain, traced_binding) || traced_binding.table_index != get.table_index) {
					all_traced = false;
					return;
				}
				bindings.push_back(binding);
				column_ids.push_back(get.GetColumnIds()[traced_binding.column_index]);
			});
		}
		return all_traced;
	}

	// Whether the rowid can take the embedding's slot in the table scan: the scan has no rowid yet, no pushed-down
	// filter reads the embedding, and the chain only forwards it (no predicate and no computed value reads it).
	// `forwards` receives the projections' column references that forward it.
	static bool CanReplaceEmbeddingByRowId(const LogicalGet &get, const vector<reference<LogicalOperator>> &chain,
	                                       const idx_t embedding_column_position,
	                                       vector<reference<BoundColumnRefExpression>> &forwards) {
		const auto &column_ids = get.GetColumnIds();
		// To support Filter on the rowid (the scan already emits the rowid, which is reused instead).
		for (auto &column_id : column_ids) {
			if (column_id.IsRowIdColumn()) {
				return false;
			}
		}
		if (get.table_filters.filters.find(column_ids[embedding_column_position].GetPrimaryIndex()) !=
		    get.table_filters.filters.end()) {
			return false;
		}
		// The bindings that carry the embedding, from the table scan up the chain.
		vector<ColumnBinding> embedding_bindings {ColumnBinding(get.table_index, embedding_column_position)};
		const auto reads_embedding = [&](unique_ptr<Expression> &expression) {
			bool reads = false;
			ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
				if (child.type == ExpressionType::BOUND_COLUMN_REF &&
				    std::find(embedding_bindings.begin(), embedding_bindings.end(),
				              child.Cast<BoundColumnRefExpression>().binding) != embedding_bindings.end()) {
					reads = true;
				}
			});
			return reads;
		};
		for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
			auto &op = it->get();
			// To support Filter reading the embedding (the scan keeps the embedding and appends the rowid instead).
			if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
				for (auto &expression : op.expressions) {
					if (reads_embedding(expression)) {
						return false;
					}
				}
				continue;
			}
			// To support: IN list of at least five constants, IN lists combined with other predicates, Subqueries as
			// filters, our table on the probe side.
			if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				// A join's predicate is its condition, whose left side reads the chain below it.
				for (auto &condition : op.Cast<LogicalComparisonJoin>().conditions) {
					if (reads_embedding(condition.left)) {
						return false;
					}
				}
				continue;
			}
			// To support: Subquery or view with a residual filter, Renamed embedding (projections that forward it).
			auto &projection = op.Cast<LogicalProjection>();
			vector<ColumnBinding> forwarded_bindings;
			for (idx_t i = 0; i < projection.expressions.size(); i++) {
				auto &expression = projection.expressions[i];
				if (expression->type == ExpressionType::BOUND_COLUMN_REF && reads_embedding(expression)) {
					forwards.push_back(expression->Cast<BoundColumnRefExpression>());
					forwarded_bindings.emplace_back(projection.table_index, i);
				} else if (reads_embedding(expression)) {
					return false;
				}
			}
			embedding_bindings = std::move(forwarded_bindings);
		}
		return true;
	}

	// Makes the rowid of the rows that pass the chain (ordered from the top down to `get`) come out of the chain's top
	// operator, and returns its binding there. `embedding_binding` is the embedding's binding at the top.
	static ColumnBinding PassRowIdThroughChain(LogicalGet &get, const vector<reference<LogicalOperator>> &chain,
	                                           const ColumnBinding embedding_binding,
	                                           const optional_idx embedding_column_position) {
		auto &column_ids = get.GetMutableColumnIds();
		// Above the filtered search the embedding is fetched again by rowid, so below it the scan only needs it for
		// predicates. If no predicate reads it and the scan has no rowid yet, the rowid takes the embedding's slot:
		// the chain forwards it along the embedding's path, and the filter pipeline no longer reads embeddings.
		// To support the chain cases whose predicates do not read the embedding: OR or sparse IN on one column,
		// Predicates over two or more columns, Conjunction left after pushdown, Volatile predicate, Selective residual
		// filter, Subquery or view with a residual filter, IN list of at least five constants, IN lists combined with
		// other predicates, Subqueries as filters, our table on the probe side.
		vector<reference<BoundColumnRefExpression>> forwards;
		if (embedding_column_position.IsValid() &&
		    CanReplaceEmbeddingByRowId(get, chain, embedding_column_position.GetIndex(), forwards)) {
			column_ids[embedding_column_position.GetIndex()] = ColumnIndex(COLUMN_IDENTIFIER_ROW_ID);
			for (auto &forward : forwards) {
				forward.get().return_type = LogicalType::ROW_TYPE;
			}
			return embedding_binding;
		}
		// Otherwise the scan emits the rowid as well, and every operator of the chain forwards it. Appending keeps the
		// positions of the other columns, which bindings and projection maps refer to.
		// To support: Filter reading the embedding (rowid appended), Filter on the rowid (existing rowid reused).
		idx_t rowid_position = column_ids.size();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].IsRowIdColumn()) {
				rowid_position = i;
				break;
			}
		}
		if (rowid_position == column_ids.size()) {
			get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
		}
		if (!get.projection_ids.empty() && std::find(get.projection_ids.begin(), get.projection_ids.end(),
		                                             rowid_position) == get.projection_ids.end()) {
			get.projection_ids.push_back(rowid_position);
		}
		ColumnBinding rowid_binding(get.table_index, rowid_position);
		for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
			// To support Subquery or view with a residual filter.
			if (it->get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &projection = it->get().Cast<LogicalProjection>();
				projection.expressions.push_back(
				    make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, rowid_binding));
				rowid_binding = ColumnBinding(projection.table_index, projection.expressions.size() - 1);
				continue;
			}
			// A FILTER and a join forward the bindings of children[0], trimmed by a projection map when it is set.
			// To support: the FILTER cases listed in IsChainOperator, IN list of at least five constants, IN lists
			// combined with other predicates, Subqueries as filters, our table on the probe side.
			auto &projection_map = it->get().type == LogicalOperatorType::LOGICAL_FILTER
			                           ? it->get().Cast<LogicalFilter>().projection_map
			                           : it->get().Cast<LogicalComparisonJoin>().left_projection_map;
			if (projection_map.empty()) {
				continue;
			}
			const auto child_bindings = it->get().children[0]->GetColumnBindings();
			const auto child_position = std::find(child_bindings.begin(), child_bindings.end(), rowid_binding);
			D_ASSERT(child_position != child_bindings.end());
			const auto position = static_cast<idx_t>(child_position - child_bindings.begin());
			if (std::find(projection_map.begin(), projection_map.end(), position) == projection_map.end()) {
				projection_map.push_back(position);
			}
		}
		return rowid_binding;
	}

	static bool TryOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto &context = input.context;
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.null_order != OrderByNullType::NULLS_LAST) {
			// Rows without an embedding have a NULL distance and are not in the index, so the index can only produce
			// them last. The binder has already resolved an unspecified null order (default_null_order).
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		// Follow the TopN key through projections that only forward it (e.g. `ORDER BY d` over a subquery that computes
		// d) to the projection that computes the distance. The forwarding projections stay above the search.
		// To support Distance aliased in a subquery.
		reference<LogicalProjection> distance_projection = top_n.children.front()->Cast<LogicalProjection>();
		ColumnBinding distance_binding = bound_column_ref.binding;
		while (distance_binding.column_index < distance_projection.get().expressions.size() &&
		       distance_projection.get().expressions[distance_binding.column_index]->type ==
		           ExpressionType::BOUND_COLUMN_REF) {
			auto &forwarding_projection = distance_projection.get();
			distance_binding = forwarding_projection.expressions[distance_binding.column_index]
			                       ->Cast<BoundColumnRefExpression>()
			                       .binding;
			if (forwarding_projection.children.size() != 1 ||
			    forwarding_projection.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
				return false;
			}
			distance_projection = forwarding_projection.children.front()->Cast<LogicalProjection>();
			if (distance_binding.table_index != distance_projection.get().table_index) {
				return false;
			}
		}
		auto &projection = distance_projection.get();
		if (distance_binding.column_index >= projection.expressions.size()) {
			return false;
		}

		// This the expression that is referenced by the order by expression
		const auto &projection_expr = projection.expressions[distance_binding.column_index];

		// The projection must sit on top of a get, possibly with a chain of FILTER operators (the predicates DuckDB
		// could not push into the table scan), projections (subqueries and views) and joins (IN lists and subqueries)
		// in between. IsChainOperator lists which case each kind of operator supports.
		if (projection.children.size() != 1) {
			return false;
		}
		vector<reference<LogicalOperator>> chain; // From the projection's child down to the table scan (excluded).
		// Whether the chain decides which rows pass: FILTER operators and joins do, forwarding projections do not.
		// To support Subqueries as filters, our table on the probe side (a SEMI or ANTI join without a FILTER above it
		// must stay below the search).
		bool chain_filters_rows = false;
		auto *get_ptr_ptr = &projection.children.front();
		while (IsChainOperator(**get_ptr_ptr)) {
			chain_filters_rows |= (*get_ptr_ptr)->type != LogicalOperatorType::LOGICAL_PROJECTION;
			chain.push_back(**get_ptr_ptr);
			get_ptr_ptr = &(*get_ptr_ptr)->children.front();
		}
		if ((*get_ptr_ptr)->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = *get_ptr_ptr;
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<PDXearchIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;
		// When the index expression is a plain column: the embedding's binding as the projection reads it, and its
		// position in the table scan.
		ColumnBinding embedding_binding;
		optional_idx embedding_column_position;

		table_info.BindIndexes(context, PDXearchIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || PDXearchIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<PDXearchIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				continue;
			}
			// Check that the PDXearch index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			// The index expression is bound to the table scan, the argument reads it through the chain: compare the
			// argument traced down to the table scan.
			// To support Renamed embedding (and every case with a projection between the distance and the scan).
			const auto is_index_expression = [&](const Expression &argument) {
				if (argument.type != ExpressionType::BOUND_COLUMN_REF) {
					return index_expr->Equals(argument);
				}
				auto binding = argument.Cast<BoundColumnRefExpression>().binding;
				return TryTraceToTableScan(chain, binding) &&
				       index_expr->Equals(BoundColumnRefExpression(argument.return_type, binding));
			};

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !is_index_expression(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !is_index_expression(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					continue;
				}
			}
			if (index_expr_ref.get().type == ExpressionType::BOUND_COLUMN_REF) {
				embedding_binding = index_expr_ref.get().Cast<BoundColumnRefExpression>().binding;
				auto traced_binding = embedding_binding;
				TryTraceToTableScan(chain, traced_binding);
				embedding_column_position = traced_binding.column_index;
			}

			const auto num_dimensions = cast_index.GetNumDimensions();
			const auto &matched_embedding = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_embedding = make_unsafe_uniq_array<float>(num_dimensions);

			if (matched_embedding.type().id() == LogicalTypeId::BLOB) {
				// BLOB path: decode the quantized blob to float array
				auto blob = StringValue::Get(matched_embedding);
				auto blob_dims = BlobDimensionCount(blob.size());
				if (blob_dims != num_dimensions) {
					return false;
				}
				DecodeBlobToFloatArray(const_data_ptr_cast(blob.data()), blob.size(), query_embedding.get());
			} else {
				// ARRAY path: existing logic
				auto embedding_elements = ArrayValue::GetChildren(matched_embedding);
				for (idx_t i = 0; i < num_dimensions; i++) {
					query_embedding[i] = embedding_elements[i].GetValue<float>();
				}
			}

			// With an OFFSET the TopN stays in the plan and skips the first `offset` of the rows we return.
			bind_data = make_uniq<PDXearchIndexScanBindData>(duck_table, cast_index, top_n.limit + top_n.offset,
			                                                 std::move(query_embedding));
			break;
		}

		if (!bind_data) {
			// No index found
			return false;
		}

		// The columns the index scan emits: those the projection reads. Checked before the plan is changed.
		// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate.
		vector<ColumnBinding> output_bindings;
		vector<ColumnIndex> output_column_ids;
		if (!TryCollectOutputColumns(projection, chain, get, output_bindings, output_column_ids)) {
			return false;
		}

		// A join above this search may drop some of the K rows the search returns, but must not decide which K rows
		// those are. DuckDB's join filter pushdown also hands the join's runtime filters to this table scan, through
		// the TopN the search replaces, so the scan is detached from them. The join keeps its own reference and fills
		// a filter set nothing reads. For example (other = {1, 3, 7, 11, 4474}):
		//
		//   SELECT id FROM (SELECT id FROM t WHERE id < 15000 ORDER BY array_distance(emb, q) LIMIT 10) s
		//   WHERE s.id IN (SELECT x FROM other);
		//
		// keeps the rows of `other` among the 10 nearest rows with id < 15000 (only 4474). With the join's filters on
		// the scan, the search would pick its neighbours among those five rows only and return all of them. The same
		// IN inside the subquery (`WHERE id < 15000 AND id IN (SELECT x FROM other)`) is a predicate of the search:
		// the joins inside the chain keep their runtime filters, moved to a filter set that only they fill.
		// To support: Join above the search (the scan leaves its old filter set), Subqueries as filters, our table on
		// the probe side (the joins inside the chain move to the new one).
		auto chain_dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
		for (auto &op : chain) {
			if (op.get().type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				continue;
			}
			auto &join = op.get().Cast<LogicalComparisonJoin>();
			if (!join.filter_pushdown || !get.dynamic_filters) {
				continue;
			}
			for (auto &probe_filter : join.filter_pushdown->probe_info) {
				if (probe_filter.dynamic_filters == get.dynamic_filters) {
					probe_filter.dynamic_filters = chain_dynamic_filters;
				}
			}
		}
		get.dynamic_filters = std::move(chain_dynamic_filters);

		bool has_pushed_down_filters = !get.table_filters.filters.empty();

		if (!chain_filters_rows && !has_pushed_down_filters) {
			// Scenario 1: Non-filtered search.
			// To support View or subquery without a predicate (the index scan replaces the forwarding projections too).

			// 1. Replace the table scan, and the projections above it that forward its columns, with the index scan.
			auto pdxearch_index_scan = make_uniq<LogicalPDXearchIndexScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    std::move(output_column_ids), std::move(output_bindings));

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// 2. Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		} else if (!chain_filters_rows) {
			// Scenario 2: Simple filtered search.
			// To support Filtered subquery (filter pushed into the scan; the forwarding projections are dropped).

			// We have a top-n operator on top of a table scan that has pushed down filters, possibly below projections
			// that forward its columns. The PDXearchIndexFilteredScan replaces those projections: it emits the columns
			// the projection above reads (output_bindings), fetched by rowid. We do the following:
			// 1. The PDXearchIndexFilteredScan goes directly above the table scan, which the next steps change.

			// 2. Set the table scan's column_ids to include only those needed for the filters and the projected rowid
			//    column.
			get.ClearColumnIds();
			for (auto &filter_entry : get.table_filters.filters) {
				get.AddColumnId(filter_entry.first);
			}
			// Add the rowid column (if it has not already been added because it is relied on by pushed-down filters).
			idx_t rowid_pos = DConstants::INVALID_INDEX;
			for (idx_t i = 0; i < get.GetColumnIds().size(); i++) {
				if (get.GetColumnIds()[i].GetPrimaryIndex() == COLUMN_IDENTIFIER_ROW_ID) {
					rowid_pos = i;
					break;
				}
			}
			if (rowid_pos == DConstants::INVALID_INDEX) {
				get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
				rowid_pos = get.GetColumnIds().size() - 1;
			}

			// 3. Set the projection to only include the rowid column.
			get.projection_ids.clear();
			get.projection_ids.push_back(rowid_pos);

			// Re-resolve the get operator types after changing projection
			get.ResolveOperatorTypes();

			// 4. Insert a PDXearchIndexFilteredScan operator above the table scan.
			auto pdxearch_index_filtered_scan = make_uniq<LogicalPDXearchIndexFilteredScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    std::move(output_column_ids), std::move(output_bindings));
			// Bind and push back only the row_id column that the table scan should emit
			// The table scan is configured to only output the rowid column at index rowid_pos
			auto row_id_column =
			    make_uniq<BoundColumnRefExpression>(get.types[0], ColumnBinding(get.table_index, rowid_pos));
			pdxearch_index_filtered_scan->expressions.push_back(std::move(row_id_column));

			pdxearch_index_filtered_scan->children.push_back(std::move(get_ptr));
			pdxearch_index_filtered_scan->ResolveOperatorTypes();

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_filtered_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// 5. Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		} else {
			// Scenario 3: Filtered search where FILTER operators between the projection and the table scan hold (part
			// of) the predicate, possibly mixed with projections that forward the table scan's columns. The table scan
			// may have pushed-down filters as well.
			// To support every case whose chain holds a FILTER or a join: the FILTER and join cases listed in
			// IsChainOperator, including Subquery or view with a residual filter.

			// 1. The PDXearchIndexFilteredScan emits the columns the projection reads (output_bindings), fetched by
			//    rowid, so the operators above it keep working unchanged.

			// 2. Make the rowid of every row that passes the predicate come out of the top of the chain.
			const auto rowid_binding = PassRowIdThroughChain(get, chain, embedding_binding, embedding_column_position);

			// 3. Keep only that rowid on top of the chain: the filtered search's sink takes a single rowid column.
			vector<unique_ptr<Expression>> rowid_expressions;
			rowid_expressions.push_back(make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, rowid_binding));
			auto rowid_projection =
			    make_uniq<LogicalProjection>(input.optimizer.binder.GenerateTableIndex(), std::move(rowid_expressions));
			rowid_projection->children.push_back(std::move(projection.children.front()));

			// 4. Insert a PDXearchIndexFilteredScan operator above the rowid projection.
			auto pdxearch_index_filtered_scan = make_uniq<LogicalPDXearchIndexFilteredScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    std::move(output_column_ids), std::move(output_bindings));
			pdxearch_index_filtered_scan->expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    LogicalType::ROW_TYPE, ColumnBinding(rowid_projection->table_index, 0)));
			pdxearch_index_filtered_scan->children.push_back(std::move(rowid_projection));
			pdxearch_index_filtered_scan->ResolveOperatorTypes();

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_filtered_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// 5. Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		}
	}

	static bool OptimizeChildren(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto ok = TryOptimize(input, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(input, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
				    ( // Targets non-filtered.
				        child->children[0]->GetName() == "PDXEARCH_INDEX_SCAN" ||
				        // Targets filtered.
				        child->children[0]->GetName() == "PDXEARCH_INDEX_FILT_SCAN")) {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_pdxearch_scan = OptimizeChildren(input, plan);
		if (did_use_pdxearch_scan) {
			MergeProjections(plan);
		}
	}
};

void PDXearchModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	OptimizerExtension::Register(db.config, PDXearchIndexScanOptimizer());
}

} // namespace duckdb
