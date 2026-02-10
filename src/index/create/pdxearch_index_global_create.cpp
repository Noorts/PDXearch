#include "index/create/pdxearch_index_global_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "index/pdxearch_index.hpp"

namespace duckdb {

PhysicalCreateGlobalPDXearchIndex::PhysicalCreateGlobalPDXearchIndex(
    PhysicalPlan &physical_plan, const vector<LogicalType> &types_p, TableCatalogEntry &table_p,
    const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
    vector<unique_ptr<Expression>> unbound_expressions, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {

	for (auto &virtual_column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(virtual_column_id)).index);
	}
}

class CreateGlobalPDXearchIndexLocalSinkState : public LocalSinkState {};

unique_ptr<LocalSinkState> PhysicalCreateGlobalPDXearchIndex::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreateGlobalPDXearchIndexLocalSinkState>();
}

class CreateGlobalPDXearchIndexGlobalSinkState : public GlobalSinkState {
public:
	explicit CreateGlobalPDXearchIndexGlobalSinkState(const PhysicalCreateGlobalPDXearchIndex &op)
	    : global_index(make_uniq<PDXearchIndex>(op.info->index_name, op.info->constraint_type, op.storage_ids,
	                                            TableIOManager::Get(op.table.GetStorage()), op.unbound_expressions,
	                                            op.table.GetStorage().db, op.info->options, IndexStorageInfo(),
	                                            op.estimated_cardinality)),
	      num_dimensions(ArrayType::GetSize(op.unbound_expressions[0]->return_type)),
	      max_num_embeddings(op.estimated_cardinality),
	      embeddings(make_uniq_array<float>(max_num_embeddings * num_dimensions)),
	      row_ids(make_uniq_array<row_t>(max_num_embeddings)),
	      embedding_preprocessor(make_uniq<EmbeddingPreprocessor>(
	          num_dimensions, global_index->Cast<PDXearchIndex>().GetRotationMatrix(), PDXearchWrapper::EPSILON0)),
	      is_normalized(global_index->Cast<PDXearchIndex>().IsNormalized()) {
	}
	unique_ptr<BoundIndex> global_index;

	const idx_t num_dimensions;
	idx_t current_embedding_count {0};
	// Contiguous allocations based on the estimated cardinality.
	const idx_t max_num_embeddings;
	unique_ptr<float[]> embeddings;
	unique_ptr<row_t[]> row_ids;

	const unique_ptr<EmbeddingPreprocessor> embedding_preprocessor;
	const bool is_normalized {false};
};

unique_ptr<GlobalSinkState> PhysicalCreateGlobalPDXearchIndex::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CreateGlobalPDXearchIndexGlobalSinkState>(*this);
}

SinkResultType PhysicalCreateGlobalPDXearchIndex::Sink(ExecutionContext &context, DataChunk &input_chunk,
                                                       OperatorSinkInput &input) const {
	auto &g_sink = input.global_state.Cast<CreateGlobalPDXearchIndexGlobalSinkState>();

	// Validate input chunk structure.
	D_ASSERT(input_chunk.ColumnCount() == 2);
	auto &embedding_column = input_chunk.data[0];
	auto &row_id_column = input_chunk.data[1];
	D_ASSERT(embedding_column.GetType().id() == LogicalTypeId::ARRAY);
	D_ASSERT(ArrayType::GetSize(embedding_column.GetType()) == g_sink.num_dimensions);
	D_ASSERT(row_id_column.GetType() == LogicalType::ROW_TYPE);

	// Process the current chunk's embeddings and row ids.
	const idx_t num_embeddings = input_chunk.size();
	D_ASSERT(g_sink.current_embedding_count + num_embeddings <= g_sink.max_num_embeddings);

	g_sink.embedding_preprocessor->PreprocessEmbeddings(
	    FlatVector::GetData<float>(ArrayVector::GetEntry(embedding_column)),
	    g_sink.embeddings.get() + (g_sink.current_embedding_count * g_sink.num_dimensions), num_embeddings,
	    g_sink.is_normalized);

	row_id_column.Flatten(num_embeddings);
	const auto row_id_data = FlatVector::GetData<row_t>(row_id_column);
	memcpy(g_sink.row_ids.get() + g_sink.current_embedding_count, row_id_data, num_embeddings * sizeof(row_t));
	g_sink.current_embedding_count += num_embeddings;

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateGlobalPDXearchIndex::Combine(ExecutionContext &context,
                                                                 OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreateGlobalPDXearchIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                             OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<CreateGlobalPDXearchIndexGlobalSinkState>();
	D_ASSERT(g_sink.current_embedding_count > 0);

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException(
		    "Transaction conflict: cannot add an index to a table that has been altered or dropped");
	}

	auto &schema = table.schema;
	info->column_ids = storage_ids;

	// Ensure that the index does not yet exist in the catalog.
	auto entry = schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info->index_name);
	if (entry) {
		if (info->on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
			throw CatalogException("Index with name \"%s\" already exists!", info->index_name);
		}
		// IF NOT EXISTS on existing index. We are done.
		return SinkFinalizeType::READY;
	}

	auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), *info, table).get();
	D_ASSERT(index_entry);
	auto &index = index_entry->Cast<DuckIndexEntry>();

	auto &pdxearch_index = g_sink.global_index->Cast<PDXearchIndex>();
	pdxearch_index.SetUpGlobalIndex(g_sink.row_ids.get(), g_sink.embeddings.get(), g_sink.current_embedding_count);

	index.initial_index_size = g_sink.global_index->GetInMemorySize();

	// Add the index to the storage.
	storage.AddIndex(std::move(g_sink.global_index));
	return SinkFinalizeType::READY;
}

} // namespace duckdb
