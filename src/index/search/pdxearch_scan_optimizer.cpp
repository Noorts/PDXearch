#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "index/pdxearch_module.hpp"
#include "index/pdxearch_index.hpp"
#include "index/search/pdxearch_index_filtered_scan_logical.hpp"

#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
#include "index/search/pdxearch_index_scan_logical.hpp"
#include "index/search/pdxearch_index_scan_physical.hpp"
#else
#include "index/search/pdxearch_index_global_scan.hpp"
#endif

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
 ****************** SCENARIO 3 (composite filtered search): ********************
 *
 *  TODO: Ensure that composite filters (thus plans with a LogicalFilter
 *  operator on top of the table scan) are supported (also update description).
 *  Note that the table scan will have to project out whatever columns are
 *  needed for the LogicalFilter + the rowid.
 *  TODO: Leave out the PROJECTION at the top? Our pattern recognition starts at
 *  the TopN operator.
 *  TODO: Fix cardinality of after-optimization plan (K instead of 1M?).
 *        Also see filtered scan optimizer.
 *
 * ┌───────────────────────────┐
 * │         PROJECTION        │
 * │    ────────────────────   │
 * │        Expressions:       │
 * │           #[1.0]          │
 * │           #[1.1]          │
 * │                           │
 * │          ~10 rows         │
 * └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐
 * │           TOP_N           │
 * │    ────────────────────   │
 * │          ~10 rows         │
 * └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐
 * │         PROJECTION        │
 * │    ────────────────────   │
 * │        Expressions:       │
 * │             id            │
 * │            vec            │
 * │ array_distance(vec, [1.0, │
 * │  2.0, 3.0, 4.0, 1.0, 2.0, │
 * │  3.0, 4.0, 1.0, 2.0, 3.0, │
 * │       2.0, 3.0, 4.0])     │
 * │                           │
 * │        ~30,775 rows       │
 * └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐
 * │           FILTER          │
 * │    ────────────────────   │
 * │        Expressions:       │
 * │ ((id < 1000) OR (vec = [1 │
 * │ .0, 2.0, 3.0, 4.0, 1.0, 2 │
 * │ .0, 1.0, 2.0, 3.0, 4.0, 1 │
 * │    .0, 2.0, 3.0, 4.0]))   │
 * │                           │
 * │        ~30,775 rows       │
 * └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐
 * │          SEQ_SCAN         │
 * │    ────────────────────   │
 * │       Filters: id>10      │
 * │        Table: mxbai       │
 * │   Type: Sequential Scan   │
 * │                           │
 * │       ~153,876 rows       │
 * └───────────────────────────┘
 */
class PDXearchIndexScanOptimizer : public OptimizerExtension {
public:
	PDXearchIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
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

		// The projection must sit on top of a get
		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
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

		table_info.BindIndexes(context, PDXearchIndex::TYPE_NAME);
		table_info.GetIndexes().Scan([&](Index &index) {
			if (!index.IsBound() || PDXearchIndex::TYPE_NAME != index.GetIndexType()) {
				return false;
			}
			auto &cast_index = index.Cast<PDXearchIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			// Check that the PDXearch index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				return false;
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
					return false;
				}
			}

			const auto num_dimensions = cast_index.GetNumDimensions();
			const auto &matched_embedding = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_embedding = make_unsafe_uniq_array<float>(num_dimensions);
			auto embedding_elements = ArrayValue::GetChildren(matched_embedding);
			for (idx_t i = 0; i < num_dimensions; i++) {
				query_embedding[i] = embedding_elements[i].GetValue<float>();
			}

			bind_data =
			    make_uniq<PDXearchIndexScanBindData>(duck_table, cast_index, top_n.limit, std::move(query_embedding));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		bool has_pushed_down_filters = !get.table_filters.filters.empty();

		if (!has_pushed_down_filters) {
#ifndef PDX_USE_ALTERNATIVE_GLOBAL_VERSION
			// Scenario 1: Non-filtered search (parallel version).

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

			// 3. Remove the TopN operator.
			plan = std::move(top_n.children[0]);
			return true;
#else
			// Scenario 1: Non-filtered search (global version).

			// 1. Replace the table scan with the index scan.
			const auto cardinality = get.function.cardinality(context, bind_data.get());
			get.function = PDXearchIndexScanFunction::GetFunction();
			get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
			get.estimated_cardinality = cardinality->estimated_cardinality;
			get.bind_data = std::move(bind_data);

			// 2. Remove the TopN operator
			plan = std::move(top_n.children[0]);
			return true;
#endif
		} else {
			// Scenario 2: Simple filtered search (parallel and global version)
			// The logic inside LogicalPDXearchIndexFilteredScan determines
			// whether to use the parallel or global implementation.

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

			// 5. Remove the TopN operator.
			plan = std::move(top_n.children[0]);
			return true;
		}
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if ( // Non-filtered search
				    (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				     child->children[0]->Cast<LogicalGet>().function.name == "pdxearch_index_scan") ||
				    // Filtered search
				    (child->children[0]->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
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
		auto did_use_pdxearch_scan = OptimizeChildren(input.context, plan);
		if (did_use_pdxearch_scan) {
			MergeProjections(plan);
		}
	}
};

void PDXearchModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(PDXearchIndexScanOptimizer());
}

} // namespace duckdb
