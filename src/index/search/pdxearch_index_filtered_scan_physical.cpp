#include "index/search/pdxearch_index_filtered_scan_physical.hpp"
#include "duckdb/parallel/event.hpp"
#include "index/pdxearch_index.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"

#include "pdx/searcher.hpp"

#include <algorithm>

namespace duckdb {

PhysicalPDXearchIndexFilteredScan::PhysicalPDXearchIndexFilteredScan(
    PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<PDXearchIndexPhysicalScanBindData> bind_data,
    vector<ColumnIndex> column_ids, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalPDXearchIndexFilteredScan::TYPE, std::move(types), estimated_cardinality),
      bind_data(std::move(bind_data)), column_ids(std::move(column_ids)) {
}

// ------------------------------
// Sink: State, and Sink and Combine methods.
// ------------------------------

class PhysicalFilteredScanGlobalSinkState : public GlobalSinkState {
public:
	PhysicalFilteredScanGlobalSinkState(ClientContext &context, const PhysicalPDXearchIndexFilteredScan &op,
	                                    const PDXearchIndexPhysicalScanBindData &bind_data)
	    : search_lock(bind_data.index.Cast<PDXearchIndex>().TakeSearchLock()), context(context), op(op),
	      limit(bind_data.limit), index(bind_data.index.Cast<PDXearchIndex>()),
	      preprocessed_query_embedding(make_uniq_array<float>(index.GetNumDimensions())), pdxearch_row_ids(nullptr) {
		// Preprocess the query embedding.
		EmbeddingPreprocessor embedding_preprocessor(index.GetNumDimensions(), index.GetRotationMatrix());
		embedding_preprocessor.PreprocessEmbedding(bind_data.query_embedding.get(), preprocessed_query_embedding.get(),
		                                           index.IsNormalized());

		auto n_probe = index.GetEffectiveNProbe(context);
		// The cluster count varies per row group, but the iteration loop uses a single value across all row groups. We
		// use the count for a full row group as the upper bound: for full row groups it matches exactly, and for
		// smaller row groups (e.g., the last one) the per-row-group searcher clamps internally.
		const idx_t num_clusters_for_full_row_group = PDXearchIndex::GetNumClustersForFullRowGroup();
		partitions_to_probe_per_row_group_on_first_iteration =
		    (n_probe == 0 || n_probe > num_clusters_for_full_row_group) ? num_clusters_for_full_row_group : n_probe;

		search_cursors.reserve(index.GetNumRowGroups());
	}

	// Held for the duration of execution to serialize searches against index
	// maintenance. Declared first so it is constructed first and destroyed
	// last, ensuring the lock is held while all other members (which reference
	// index state) are torn down.
	unique_ptr<StorageLockKey> search_lock;

	const ClientContext &context;
	const PhysicalPDXearchIndexFilteredScan &op;
	// The limit (the K in KNN search).
	const idx_t limit;
	PDXearchIndex &index;

	const unique_ptr<float[]> preprocessed_query_embedding;

	// Top-k heap shared by the searches of all row groups (and threads).
	PDX::TopKHeap top_k_heap {/*thread_safe=*/true};

	// For iteration support:
	// Based on the n_probe.
	idx_t partitions_to_probe_per_row_group_on_first_iteration {0};
	// The partitions to probe per iteration for each row group for all iterations except the first one. Note: the
	// number of partitions to probe on the first iteration is determined by n_probe.
	static constexpr idx_t PARTITIONS_TO_PROBE_PER_ROW_GROUP_PER_FOLLOW_UP_ITERATION = 5;
	// One search cursor per row group that had tuples passing the filter. A cursor tracks which of its clusters were
	// already probed and reports `Done()` once no cluster with passing tuples is left, so follow-up iterations only
	// schedule the cursors that still have work. Filled by Combine(), read-only afterwards.
	std::vector<unique_ptr<PDX::IIterativeSearch>> search_cursors;
	void TryFinalizeSinkPhase(Pipeline &pipeline, Event &event);

	// Row ids of the final result of the filtered search. For these rows, during the Source phase, the projected
	// columns are fetched from local storage and emitted to the next operator.
	std::unique_ptr<std::vector<row_t>> pdxearch_row_ids;
	//! Current position in pdxearch_rowids when emitting fetched rows. Used in case the operator has to emit multiple
	//! chunks of results.
	idx_t pdxearch_row_ids_idx {0};
};

unique_ptr<GlobalSinkState> PhysicalPDXearchIndexFilteredScan::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<PhysicalFilteredScanGlobalSinkState>(context, *this, *bind_data);
}

class PhysicalFilteredScanLocalSinkState : public LocalSinkState {
public:
	PhysicalFilteredScanLocalSinkState() : current_row_group_passing_rowids() {
		current_row_group_passing_rowids.reserve(DEFAULT_ROW_GROUP_SIZE);
	}

	// Temporary row group staging area.
	idx_t current_row_group_id {0};
	// The row ids of the current row group that passed the predicate and were thus emitted by the child operator (e.g.,
	// sequential scan operator).
	std::vector<row_t> current_row_group_passing_rowids;

	// The search cursors started by this thread. Moved into the global sink state's `search_cursors` in Combine().
	std::vector<unique_ptr<PDX::IIterativeSearch>> search_cursors;
};

unique_ptr<LocalSinkState> PhysicalPDXearchIndexFilteredScan::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PhysicalFilteredScanLocalSinkState>();
}

// Starts the filtered search of the row group staged in the local state, with the row ids that passed the SQL
// predicate, and runs its first iteration: the n_probe nearest clusters that hold passing tuples.
static void BeginSearchForStagedRowGroup(PDXearchIndex &index, PhysicalFilteredScanGlobalSinkState &g_sink,
                                         PhysicalFilteredScanLocalSinkState &l_sink) {
	auto search_cursor =
	    index.BeginSearchForRowGroup(l_sink.current_row_group_id, g_sink.preprocessed_query_embedding.get(),
	                                 g_sink.limit, g_sink.top_k_heap, &l_sink.current_row_group_passing_rowids);
	search_cursor->Next(g_sink.partitions_to_probe_per_row_group_on_first_iteration);
	l_sink.search_cursors.push_back(std::move(search_cursor));
}

SinkResultType PhysicalPDXearchIndexFilteredScan::Sink(ExecutionContext &context, DataChunk &input_chunk,
                                                       OperatorSinkInput &input) const {
	auto &l_sink = input.local_state.Cast<PhysicalFilteredScanLocalSinkState>();
	auto &g_sink = input.global_state.Cast<PhysicalFilteredScanGlobalSinkState>();
	auto &index = bind_data->index.Cast<PDXearchIndex>();

	if (input_chunk.size() == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	// We control the query plan, so the input chunk format should always be valid.
	D_ASSERT(input_chunk.ColumnCount() == 1);
	D_ASSERT(input_chunk.data[0].GetType() == LogicalType::ROW_TYPE);

	input_chunk.data[0].Flatten(input_chunk.size());
	const auto input_chunk_row_ids = FlatVector::GetData<row_t>(input_chunk.data[0]);
	const idx_t row_group_id = GetRowGroupId(input_chunk_row_ids[0]);
	D_ASSERT(l_sink.current_row_group_id <= row_group_id);

	// If we encounter a new row group, then start the filtered search of the previous row group.
	if (row_group_id > l_sink.current_row_group_id && !l_sink.current_row_group_passing_rowids.empty()) {
		BeginSearchForStagedRowGroup(index, g_sink, l_sink);
		// Clear the local state to process the next row group.
		l_sink.current_row_group_passing_rowids.clear();
	}
	l_sink.current_row_group_id = row_group_id;

	// Collect row ids of the current row group into the local state.
	for (idx_t i = 0; i < input_chunk.size(); i++) {
		l_sink.current_row_group_passing_rowids.push_back(input_chunk_row_ids[i]);
	}
	D_ASSERT(l_sink.current_row_group_passing_rowids.size() <= DEFAULT_ROW_GROUP_SIZE);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalPDXearchIndexFilteredScan::Combine(ExecutionContext &context,
                                                                 OperatorSinkCombineInput &input) const {
	auto &g_sink = input.global_state.Cast<PhysicalFilteredScanGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<PhysicalFilteredScanLocalSinkState>();
	auto &index = g_sink.index;

	// If this thread's last row group has not been searched (for one iteration), do so now.
	if (!l_sink.current_row_group_passing_rowids.empty()) {
		BeginSearchForStagedRowGroup(index, g_sink, l_sink);
	}

	// Merge this thread's search cursors into the global sink state.
	const auto guard = g_sink.Lock();
	for (auto &search_cursor : l_sink.search_cursors) {
		g_sink.search_cursors.push_back(std::move(search_cursor));
	}
	l_sink.search_cursors.clear();

	return SinkCombineResultType::FINISHED;
}

// ------------------------------
// Sink: Finalize and search iteration mechanism.
// ------------------------------

// A task that performs one iteration of the filtered search for a single row group. An iteration means that the next X
// clusters with passing tuples of this row group are probed.
class PhysicalFilteredScanSearchIterationTask : public ExecutorTask {
public:
	PhysicalFilteredScanSearchIterationTask(shared_ptr<Event> event_p, ClientContext &context,
	                                        PDX::IIterativeSearch &search_cursor_p, const PhysicalOperator &op_p)
	    : ExecutorTask(context, std::move(event_p), op_p), search_cursor(search_cursor_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		search_cursor.Next(
		    PhysicalFilteredScanGlobalSinkState::PARTITIONS_TO_PROBE_PER_ROW_GROUP_PER_FOLLOW_UP_ITERATION);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

	string TaskType() const override {
		return "PhysicalFilteredScanSearchIterationTask";
	}

private:
	PDX::IIterativeSearch &search_cursor;
};

// An event that executes one search iteration in all row groups that still have clusters to probe. This means that
// the next X clusters of each of those row groups are probed.
class PhysicalFilteredScanSearchIterationEvent : public BasePipelineEvent {
public:
	PhysicalFilteredScanSearchIterationEvent(Pipeline &pipeline_p, PhysicalFilteredScanGlobalSinkState &g_sink_p)
	    : BasePipelineEvent(pipeline_p), g_sink(g_sink_p) {
	}

	PhysicalFilteredScanGlobalSinkState &g_sink;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		vector<shared_ptr<Task>> tasks;
		for (auto &search_cursor : g_sink.search_cursors) {
			if (search_cursor->Done()) {
				continue;
			}
			tasks.push_back(make_uniq<PhysicalFilteredScanSearchIterationTask>(shared_from_this(), context,
			                                                                   *search_cursor, g_sink.op));
		}
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		g_sink.TryFinalizeSinkPhase(*pipeline, *this);
	}
};

SinkFinalizeType PhysicalPDXearchIndexFilteredScan::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                             OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<PhysicalFilteredScanGlobalSinkState>();

	g_sink.TryFinalizeSinkPhase(pipeline, event);

	return SinkFinalizeType::READY;
}

// Move from the Sink phase to the Source phase if the operator is ready to begin emitting results, that is, there are K
// results in the heap or every row group has probed all of its clusters with passing tuples. Else, stay in the Sink
// phase and run another search iteration, which will probe the next X clusters of each row group that has some left.
void PhysicalFilteredScanGlobalSinkState::TryFinalizeSinkPhase(Pipeline &pipeline, Event &event) {
	// All search tasks of the previous iteration have finished, so nothing else touches the heap or the cursors here.
	D_ASSERT(top_k_heap.heap.size() <= limit);

	const bool is_heap_filled_with_k_results = top_k_heap.heap.size() == limit;
	const bool are_all_partitions_probed =
	    std::all_of(search_cursors.begin(), search_cursors.end(),
	                [](const unique_ptr<PDX::IIterativeSearch> &search_cursor) { return search_cursor->Done(); });

	if (is_heap_filled_with_k_results || are_all_partitions_probed) {
		// If we are done, then prepare emission of the results by moving the result row ids into the Source state.
		const auto result_rowids = PDX::BuildResultSetFromHeap(limit, top_k_heap.heap);
		this->pdxearch_row_ids = make_uniq<std::vector<row_t>>(result_rowids.size());
		for (size_t i = 0; i < result_rowids.size(); i++) {
			(*this->pdxearch_row_ids)[i] = result_rowids[i].index;
		}
		// We return such that the Sink phase completes, allowing the Source phase to start.
		return;
	}

	// Else, run another iteration of filtered search on the row groups that still have clusters to probe.
	auto new_search_iteration_event = make_shared_ptr<PhysicalFilteredScanSearchIterationEvent>(pipeline, *this);
	event.InsertEvent(new_search_iteration_event);
}

// ------------------------------
// Source interface
// ------------------------------

class PhysicalFilteredScanGlobalSourceState : public GlobalSourceState {
public:
	PhysicalFilteredScanGlobalSourceState(ClientContext &context, const PDXearchIndexPhysicalScanBindData &bind_data,
	                                      const vector<ColumnIndex> &operator_column_ids) {
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

	vector<StorageIndex> column_ids;
	TableScanState local_storage_state;
	ColumnFetchState fetch_state;
};

unique_ptr<GlobalSourceState> PhysicalPDXearchIndexFilteredScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PhysicalFilteredScanGlobalSourceState>(context, *bind_data, column_ids);
}

class PhysicalFilteredScanLocalSourceState : public LocalSourceState {};

unique_ptr<LocalSourceState> PhysicalPDXearchIndexFilteredScan::GetLocalSourceState(ExecutionContext &context,
                                                                                    GlobalSourceState &gstate) const {
	return make_uniq<PhysicalFilteredScanLocalSourceState>();
}

SourceResultType PhysicalPDXearchIndexFilteredScan::GetDataInternal(ExecutionContext &context, DataChunk &output_chunk,
                                                                    OperatorSourceInput &input) const {
	auto &g_sink = sink_state->Cast<PhysicalFilteredScanGlobalSinkState>();
	auto &g_source = input.global_state.Cast<PhysicalFilteredScanGlobalSourceState>();

	D_ASSERT(g_sink.pdxearch_row_ids);
	D_ASSERT(g_sink.pdxearch_row_ids_idx <= g_sink.pdxearch_row_ids->size());

	const idx_t num_results_to_emit =
	    MinValue<idx_t>(STANDARD_VECTOR_SIZE, g_sink.pdxearch_row_ids->size() - g_sink.pdxearch_row_ids_idx);

	if (num_results_to_emit == 0) {
		return SourceResultType::FINISHED;
	}

	// Create vector of row ids that are part of the current output chunk.
	Vector row_ids_vector(LogicalType::ROW_TYPE, num_results_to_emit);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vector);
	for (idx_t i = 0; i < num_results_to_emit; i++) {
		row_ids_data[i] = g_sink.pdxearch_row_ids->at(g_sink.pdxearch_row_ids_idx + i);
	}
	g_sink.pdxearch_row_ids_idx += num_results_to_emit;

	// Fetch the data from storage.
	auto &transaction = DuckTransaction::Get(context.client, bind_data->table.catalog);
	bind_data->table.GetStorage().Fetch(transaction, output_chunk, g_source.column_ids, row_ids_vector,
	                                    num_results_to_emit, g_source.fetch_state);
	D_ASSERT(output_chunk.size() == num_results_to_emit);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

// Defines this operator's details shown in the query plan.
InsertionOrderPreservingMap<string> PhysicalPDXearchIndexFilteredScan::ParamsToString() const {
	auto &index = bind_data->index.Cast<PDXearchIndex>();
	InsertionOrderPreservingMap<string> result;
	result["Table"] = bind_data->table.name;
	result["PDXearch Index"] = bind_data->index.GetIndexName();
	result["Total Clusters"] = StringUtil::Format("%zu", index.GetTotalNumClusters());
	result["Row Groups"] = StringUtil::Format("%zu", index.GetNumRowGroups());
	const idx_t index_in_memory_size = bind_data->index.Cast<BoundIndex>().GetInMemorySize();
	result["Index Size"] = ConvertBytesToHumanReadableString(index_in_memory_size);
	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb
