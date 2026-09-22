#include "index/create/pdxearch_index_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "index/pdxearch_index.hpp"

namespace duckdb {

PhysicalCreatePDXearchIndex::PhysicalCreatePDXearchIndex(PhysicalPlan &physical_plan,
                                                         const vector<LogicalType> &types_p, TableCatalogEntry &table_p,
                                                         const vector<column_t> &column_ids,
                                                         unique_ptr<CreateIndexInfo> info,
                                                         vector<unique_ptr<Expression>> unbound_expressions,
                                                         idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {
	for (auto &virtual_column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(virtual_column_id)).index);
	}
}

class CreatePDXearchIndexGlobalSinkState : public GlobalSinkState {
public:
	explicit CreatePDXearchIndexGlobalSinkState(const PhysicalCreatePDXearchIndex &op)
	    : global_index(make_uniq<PDXearchIndex>(op.info->index_name, op.info->constraint_type, op.storage_ids,
	                                            TableIOManager::Get(op.table.GetStorage()), op.unbound_expressions,
	                                            op.table.GetStorage().db, op.info->options, IndexStorageInfo())),
	      num_dimensions(ArrayType::GetSize(op.unbound_expressions[0]->return_type)),
	      embedding_preprocessor(make_uniq<EmbeddingPreprocessor>(
	          num_dimensions, global_index->Cast<PDXearchIndex>().GetRotationMatrix())),
	      is_normalized(global_index->Cast<PDXearchIndex>().IsNormalized()) {
	}

	unique_ptr<BoundIndex> global_index;
	const idx_t num_dimensions;
	const unique_ptr<EmbeddingPreprocessor> embedding_preprocessor;
	const bool is_normalized {false};
};

unique_ptr<GlobalSinkState> PhysicalCreatePDXearchIndex::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CreatePDXearchIndexGlobalSinkState>(*this);
}

class CreatePDXearchIndexLocalSinkState : public LocalSinkState {
public:
	explicit CreatePDXearchIndexLocalSinkState(const PhysicalCreatePDXearchIndex &op, ClientContext &context,
	                                           CreatePDXearchIndexGlobalSinkState &g_sink) {
		const auto row_group_size = g_sink.global_index->Cast<PDXearchIndex>().GetRowGroupSize();
		row_group_embeddings_buffer.resize(row_group_size * g_sink.num_dimensions);
		row_group_row_ids.resize(row_group_size);
	}

	// The DuckDB row group whose rows are currently buffered.
	PDXearchRowGroupBounds row_group {0, 0};
	bool has_row_group {false};
	// Number of embeddings currently buffered in the row group.
	idx_t row_group_embeddings_count {0};
	std::vector<float> row_group_embeddings_buffer;
	// Row IDs of the embeddings currently buffered in the row group.
	std::vector<row_t> row_group_row_ids;
};

static void FlushRowGroup(PDXearchIndex &pdxearch_index, CreatePDXearchIndexLocalSinkState &l_sink) {
	if (l_sink.row_group_embeddings_count > 0) {
		pdxearch_index.SetUpIndexForRowGroup(l_sink.row_group_row_ids.data(), l_sink.row_group_embeddings_buffer.data(),
		                                     l_sink.row_group_embeddings_count, l_sink.row_group.row_start,
		                                     l_sink.row_group.count);
		l_sink.row_group_embeddings_count = 0;
	}
}

unique_ptr<LocalSinkState> PhysicalCreatePDXearchIndex::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreatePDXearchIndexLocalSinkState>(*this, context.client,
	                                                    sink_state->Cast<CreatePDXearchIndexGlobalSinkState>());
}

SinkResultType PhysicalCreatePDXearchIndex::Sink(ExecutionContext &context, DataChunk &input_chunk,
                                                 OperatorSinkInput &input) const {
	auto &g_sink = input.global_state.Cast<CreatePDXearchIndexGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<CreatePDXearchIndexLocalSinkState>();
	auto &pdxearch_index = g_sink.global_index->Cast<PDXearchIndex>();

	// Early exit if the chunk is empty.
	if (input_chunk.size() == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	// Validate input chunk structure.
	D_ASSERT(input_chunk.ColumnCount() == 2);
	auto &embedding_column = input_chunk.data[0];
	auto &row_id_column = input_chunk.data[1];
	D_ASSERT(embedding_column.GetType().id() == LogicalTypeId::ARRAY);
	D_ASSERT(ArrayType::GetSize(embedding_column.GetType()) == g_sink.num_dimensions);
	D_ASSERT(row_id_column.GetType() == LogicalType::ROW_TYPE);

	// Chunks arrive with a selection vector when the scan skipped deleted rows or the filter dropped NULL embeddings.
	const idx_t num_embeddings = input_chunk.size();
	embedding_column.Flatten(num_embeddings);
	row_id_column.Flatten(num_embeddings);
	const auto row_id_data = FlatVector::GetData<row_t>(row_id_column);

	// A chunk never spans two DuckDB row groups, so the first row id tells which one this chunk belongs to.
	PDXearchRowGroupBounds row_group;
	if (!pdxearch_index.TryGetPhysicalRowGroup(table.GetStorage(), row_id_data[0], row_group)) {
		throw InternalException("PDXearch: row id %lld is not in any row group of the table", row_id_data[0]);
	}
	D_ASSERT(!l_sink.has_row_group || l_sink.row_group.row_start <= row_group.row_start);

	// If we detect a new row group, then finalize the previous row group and prepare to process the new one.
	if (l_sink.has_row_group && row_group.row_start != l_sink.row_group.row_start) {
		FlushRowGroup(pdxearch_index, l_sink);
	}
	l_sink.row_group = row_group;
	l_sink.has_row_group = true;

	// Preprocess and accumulate the embeddings into the temporary row group buffer.
	D_ASSERT(l_sink.row_group_embeddings_count + num_embeddings <= row_group.count);

	g_sink.embedding_preprocessor->PreprocessEmbeddings(
	    FlatVector::GetData<float>(ArrayVector::GetEntry(embedding_column)),
	    l_sink.row_group_embeddings_buffer.data() + (l_sink.row_group_embeddings_count * g_sink.num_dimensions),
	    num_embeddings, g_sink.is_normalized);

	memcpy(l_sink.row_group_row_ids.data() + l_sink.row_group_embeddings_count, row_id_data,
	       num_embeddings * sizeof(row_t));

	l_sink.row_group_embeddings_count += num_embeddings;

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreatePDXearchIndex::Combine(ExecutionContext &context,
                                                           OperatorSinkCombineInput &input) const {
	auto &l_sink = input.local_state.Cast<CreatePDXearchIndexLocalSinkState>();
	auto &g_sink = input.global_state.Cast<CreatePDXearchIndexGlobalSinkState>();
	auto &pdxearch_index = g_sink.global_index->Cast<PDXearchIndex>();

	// Finalize this thread's last row group.
	FlushRowGroup(pdxearch_index, l_sink);

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreatePDXearchIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                       OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<CreatePDXearchIndexGlobalSinkState>();

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

	index.initial_index_size = g_sink.global_index->GetInMemorySize();

	// Add the index to the storage.
	storage.AddIndex(std::move(g_sink.global_index));
	return SinkFinalizeType::READY;
}

} // namespace duckdb
