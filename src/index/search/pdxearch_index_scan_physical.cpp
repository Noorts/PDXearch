#include "index/search/pdxearch_index_scan_physical.hpp"
#include "duckdb/parallel/event.hpp"
#include "index/pdxearch_index.hpp"
#include "index/pdxearch_index_utils.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/execution/executor.hpp"

namespace duckdb {

PhysicalPDXearchIndexScan::PhysicalPDXearchIndexScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                     unique_ptr<PDXearchIndexScanBindData> bind_data,
                                                     vector<ColumnIndex> column_ids, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalPDXearchIndexScan::TYPE, std::move(types), estimated_cardinality),
      bind_data(std::move(bind_data)), column_ids(std::move(column_ids)) {
}

class PDXearchScanGlobalSourceState : public GlobalSourceState {
public:
	PDXearchScanGlobalSourceState(ClientContext &context, const PhysicalPDXearchIndexScan &op,
	                              const PDXearchIndexScanBindData &bind_data,
	                              const vector<ColumnIndex> &operator_column_ids)
	    : context(context), op(op), limit(bind_data.limit), index(bind_data.index.Cast<PDXearchIndex>()),
	      preprocessed_query_embedding(make_uniq_array<float>(index.GetNumDimensions())), search_started(false),
	      search_completed(false), pdxearch_row_ids(nullptr), pdxearch_row_ids_idx(0) {

		// Preprocess the query embedding.
		const EmbeddingPreprocessor embedding_preprocessor =
		    EmbeddingPreprocessor(index.GetNumDimensions(), index.GetRotationMatrix());
		embedding_preprocessor.PreprocessEmbedding(bind_data.query_embedding.get(), preprocessed_query_embedding.get(),
		                                           index.IsNormalized());

		// Initialize the global heap.
		{
			const std::lock_guard<std::mutex> lock(global_heap_mutex);
			global_heap = make_uniq<PDX::Heap>();
			global_heap->push(HEAP_INITIALIZATION_ELEMENT);
		}

		// Determine number of clusters to probe per row group.
		const auto n_probe = index.GetEffectiveNProbe(context);
		const auto num_clusters_per_row_group = index.GetNumClustersPerRowGroup();
		num_clusters_to_probe_per_row_group =
		    (n_probe == 0 || n_probe > num_clusters_per_row_group) ? num_clusters_per_row_group : n_probe;

		// Set up column IDs for fetching data from storage.
		column_ids.reserve(operator_column_ids.size());
		for (auto &id : operator_column_ids) {
			StorageIndex storage_id;
			if (id.IsRowIdColumn()) {
				storage_id = StorageIndex();
			} else {
				auto &col = bind_data.table.GetColumn(LogicalIndex(id.GetPrimaryIndex()));
				storage_id = StorageIndex(col.StorageOid());
			}
			column_ids.emplace_back(storage_id);
		}

		// Initialize the storage scan state.
		local_storage_state.Initialize(column_ids, context, nullptr);
		auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
		local_storage.InitializeScan(bind_data.table.GetStorage(), local_storage_state.local_state, nullptr);
	}

	ClientContext &context;
	const PhysicalPDXearchIndexScan &op;
	// The limit (the K in KNN search).
	const idx_t limit;
	PDXearchIndex &index;

	const unique_ptr<float[]> preprocessed_query_embedding;

	// Global heap shared by all threads (and row groups). Assumes the `global_heap_mutex` is used.
	std::unique_ptr<PDX::Heap> global_heap;
	std::mutex global_heap_mutex;
	// Element used to initialize the global heap. PDXearch uses the top of the heap in its operations, thus there must
	// be an initial element. This element is always filtered out when the heap is transformed into a result set.
	static constexpr PDX::KNNCandidate HEAP_INITIALIZATION_ELEMENT = {1337, std::numeric_limits<float>::max()};

	idx_t num_clusters_to_probe_per_row_group {0};

	// For Source blocking coordination.
	std::atomic<bool> search_started;
	std::atomic<bool> search_completed;

	// Row ids of the final result of the search. For these rows the projected columns are fetched from local storage
	// and emitted through the Source interface.
	std::unique_ptr<std::vector<row_t>> pdxearch_row_ids;
	//! Current position in pdxearch_rowids when emitting fetched rows. Used in case the operator has to emit multiple
	//! chunks of results.
	idx_t pdxearch_row_ids_idx;

	// For fetching the result rows from storage.
	vector<StorageIndex> column_ids;
	TableScanState local_storage_state;
	ColumnFetchState fetch_state;
};

class PDXearchScanSearchTask : public ExecutorTask {
public:
	PDXearchScanSearchTask(shared_ptr<Event> event_p, ClientContext &context, PDXearchScanGlobalSourceState &g_state_p,
	                       const PhysicalOperator &op_p, idx_t row_group_id_p)
	    : ExecutorTask(context, std::move(event_p), op_p), g_state(g_state_p), row_group_id(row_group_id_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		auto &index = g_state.index;

		index.InitializeSearchForRowGroup(g_state.preprocessed_query_embedding.get(), g_state.limit, row_group_id,
		                                  *g_state.global_heap, g_state.global_heap_mutex);
		index.SearchRowGroup(row_group_id, g_state.num_clusters_to_probe_per_row_group);

		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

	string TaskType() const override {
		return "PDXearchScanSearchTask";
	}

private:
	PDXearchScanGlobalSourceState &g_state;
	idx_t row_group_id;
};

class PDXearchScanSearchEvent : public Event {
public:
	PDXearchScanSearchEvent(Executor &executor_p, PDXearchScanGlobalSourceState &g_state_p)
	    : Event(executor_p), g_state(g_state_p) {
	}

	PDXearchScanGlobalSourceState &g_state;

public:
	void Schedule() override {
		auto &context = GetClientContext();
		vector<shared_ptr<Task>> tasks;
		for (idx_t row_group_id = 0; row_group_id < g_state.index.GetNumRowGroups(); row_group_id++) {
			tasks.push_back(
			    make_uniq<PDXearchScanSearchTask>(shared_from_this(), context, g_state, g_state.op, row_group_id));
		}
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		// Store PDXearch result into the source state.
		const auto result_rowids = PDX::PDXearch<PDX::F32>::BuildResultSetFromHeap(g_state.limit, *g_state.global_heap);
		g_state.pdxearch_row_ids = make_uniq<std::vector<row_t>>(result_rowids.size());
		for (size_t i = 0; i < result_rowids.size(); i++) {
			(*g_state.pdxearch_row_ids)[i] = result_rowids[i].index;
		}

		g_state.search_completed = true;

		const auto guard = g_state.Lock();
		g_state.UnblockTasks(guard);
	}
};

class PDXearchScanLocalSourceState : public LocalSourceState {};

unique_ptr<GlobalSourceState> PhysicalPDXearchIndexScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PDXearchScanGlobalSourceState>(context, *this, *bind_data, column_ids);
}

unique_ptr<LocalSourceState> PhysicalPDXearchIndexScan::GetLocalSourceState(ExecutionContext &context,
                                                                            GlobalSourceState &g_state) const {
	return make_uniq<PDXearchScanLocalSourceState>();
}

SourceResultType PhysicalPDXearchIndexScan::GetData(ExecutionContext &context, DataChunk &output_chunk,
                                                    OperatorSourceInput &input) const {
	auto &g_state = input.global_state.Cast<PDXearchScanGlobalSourceState>();

	// Trigger parallel search of all row groups.
	if (!g_state.search_started.exchange(true)) {
		auto &executor = Executor::Get(context.client);
		auto event = make_shared_ptr<PDXearchScanSearchEvent>(executor, g_state);
		executor.AddEvent(event);
		event->Schedule();
	}

	// Block until the search is done.
	const auto guard = g_state.Lock();
	if (!g_state.search_completed) {
		return g_state.BlockSource(guard, input.interrupt_state);
	}

	// Search is done, now fetch and emit results. Note that if K > STANDARD_VECTOR_SIZE, then GetData will be called
	// multiple times and multiple chunks of results will be emitted.
	D_ASSERT(g_state.pdxearch_row_ids);
	D_ASSERT(g_state.pdxearch_row_ids_idx <= g_state.pdxearch_row_ids->size());

	const idx_t num_results_to_emit =
	    MinValue<idx_t>(STANDARD_VECTOR_SIZE, g_state.pdxearch_row_ids->size() - g_state.pdxearch_row_ids_idx);

	if (num_results_to_emit == 0) {
		return SourceResultType::FINISHED;
	}

	// Create vector of row ids that are part of the current output chunk.
	Vector row_ids_vector(LogicalType::ROW_TYPE, num_results_to_emit);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vector);
	for (idx_t i = 0; i < num_results_to_emit; i++) {
		row_ids_data[i] = g_state.pdxearch_row_ids->at(g_state.pdxearch_row_ids_idx + i);
	}
	g_state.pdxearch_row_ids_idx += num_results_to_emit;

	// Fetch the data from storage.
	auto &transaction = DuckTransaction::Get(context.client, bind_data->table.catalog);
	bind_data->table.GetStorage().Fetch(transaction, output_chunk, g_state.column_ids, row_ids_vector,
	                                    num_results_to_emit, g_state.fetch_state);
	D_ASSERT(output_chunk.size() == num_results_to_emit);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalPDXearchIndexScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Table"] = bind_data->table.name;
	result["PDXearch Index"] = bind_data->index.GetIndexName();
	result["Total Clusters"] =
	    StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumClustersPerRowGroup() *
	                                  bind_data->index.Cast<PDXearchIndex>().GetNumRowGroups());
	result["Row Groups"] = StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumRowGroups());
	const idx_t index_in_memory_size = bind_data->index.Cast<BoundIndex>().GetInMemorySize();
	result["Index Size"] = ConvertBytesToHumanReadableString(index_in_memory_size);
	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb
