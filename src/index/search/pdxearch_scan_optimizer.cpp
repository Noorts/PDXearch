#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
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

	static bool CanReplaceEmbeddingByRowId(const LogicalGet &get, const vector<reference<LogicalFilter>> &filters,
	                                       const idx_t embedding_column_position) {
		const auto &column_ids = get.GetColumnIds();
		for (auto &column_id : column_ids) {
			if (column_id.IsRowIdColumn()) {
				return false;
			}
		}
		if (get.table_filters.filters.find(column_ids[embedding_column_position].GetPrimaryIndex()) !=
		    get.table_filters.filters.end()) {
			return false;
		}
		const ColumnBinding embedding_binding(get.table_index, embedding_column_position);
		bool is_read_by_a_filter = false;
		for (auto &filter : filters) {
			for (auto &expression : filter.get().expressions) {
				ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
					if (child.type == ExpressionType::BOUND_COLUMN_REF &&
					    child.Cast<BoundColumnRefExpression>().binding == embedding_binding) {
						is_read_by_a_filter = true;
					}
				});
			}
		}
		return !is_read_by_a_filter;
	}

	// Makes the rowid of the rows that pass `filters` (ordered from the top down to `get`) come out of the top filter,
	// and returns its binding there.
	static ColumnBinding PassRowIdThroughFilters(LogicalGet &get, const vector<reference<LogicalFilter>> &filters,
	                                             const optional_idx embedding_column_position) {
		auto &column_ids = get.GetMutableColumnIds();
		// Above the filtered search the embedding is fetched again by rowid, so below it the scan only needs it for
		// predicates. If no predicate reads it and the scan has no rowid yet, the rowid takes the embedding's slot:
		// every filter forwards it under the embedding's binding, and the filter pipeline no longer reads embeddings.
		if (embedding_column_position.IsValid() &&
		    CanReplaceEmbeddingByRowId(get, filters, embedding_column_position.GetIndex())) {
			column_ids[embedding_column_position.GetIndex()] = ColumnIndex(COLUMN_IDENTIFIER_ROW_ID);
			return ColumnBinding(get.table_index, embedding_column_position.GetIndex());
		}
		// Otherwise the scan emits the rowid as well, and every filter that projects its output forwards it. Appending
		// keeps the positions of the other columns, which bindings and projection maps refer to.
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
		const ColumnBinding rowid_binding(get.table_index, rowid_position);
		for (auto it = filters.rbegin(); it != filters.rend(); ++it) {
			auto &projection_map = it->get().projection_map;
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

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get, possibly with FILTER operators in between: they hold the predicates
		// DuckDB could not push into the table scan.
		if (projection.children.size() != 1) {
			return false;
		}
		vector<reference<LogicalFilter>> filters; // From the projection down to the table scan.
		auto *get_ptr_ptr = &projection.children.front();
		while ((*get_ptr_ptr)->type == LogicalOperatorType::LOGICAL_FILTER && (*get_ptr_ptr)->children.size() == 1) {
			filters.push_back((*get_ptr_ptr)->Cast<LogicalFilter>());
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
		// The embedding column's position in the table scan, when the index expression is a plain column.
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

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					continue;
				}
			}
			if (index_expr_ref.get().type == ExpressionType::BOUND_COLUMN_REF) {
				embedding_column_position = index_expr_ref.get().Cast<BoundColumnRefExpression>().binding.column_index;
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
		// IN inside the subquery (`WHERE id < 15000 AND id IN (SELECT x FROM other)`) is a predicate of the search.
		get.dynamic_filters.reset();

		bool has_pushed_down_filters = !get.table_filters.filters.empty();

		if (filters.empty() && !has_pushed_down_filters) {
			// Scenario 1: Non-filtered search.

			// 1. Store the table scan's column ids. The replacement PDXearchIndexFilteredScan will need to emit the
			//    same columns as the table scan did.
			const auto table_scan_column_ids = get.GetColumnIds();

			// 2. Replace the table scan with the index scan.
			auto pdxearch_index_scan = make_uniq<LogicalPDXearchIndexScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    table_scan_column_ids, get.table_index);

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// 3. Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		} else if (filters.empty()) {
			// Scenario 2: Simple filtered search.

			// We have a top-n operator on top of a table scan that has pushed down filters.
			// We do the following:
			// 1. Store the table scan's column ids. The replacement PDXearchIndexFilteredScan will need to emit the
			//    same columns as the table scan did.
			const auto table_scan_column_ids = get.GetColumnIds();

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
			    table_scan_column_ids, get.table_index);
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
			// of) the predicate. The table scan may have pushed-down filters as well.

			// 1. Store the table scan's column ids. The PDXearchIndexFilteredScan emits the same columns under the same
			//    bindings, fetched by rowid, so the operators above it keep working unchanged.
			const auto table_scan_column_ids = get.GetColumnIds();

			// 2. Make the rowid of every row that passes the predicate come out of the top filter.
			const auto rowid_binding = PassRowIdThroughFilters(get, filters, embedding_column_position);

			// 3. Keep only that rowid on top of the filters: the filtered search's sink takes a single rowid column.
			vector<unique_ptr<Expression>> rowid_expressions;
			rowid_expressions.push_back(make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, rowid_binding));
			auto rowid_projection =
			    make_uniq<LogicalProjection>(input.optimizer.binder.GenerateTableIndex(), std::move(rowid_expressions));
			rowid_projection->children.push_back(std::move(projection.children.front()));

			// 4. Insert a PDXearchIndexFilteredScan operator above the rowid projection.
			auto pdxearch_index_filtered_scan = make_uniq<LogicalPDXearchIndexFilteredScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    table_scan_column_ids, get.table_index);
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
