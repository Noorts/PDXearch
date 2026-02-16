#include "index/search/pdxearch_index_filtered_scan_physical.hpp"
#include "duckdb/parallel/event.hpp"
#include "index/pdxearch_index.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"

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
	    : context(context), op(op), limit(bind_data.limit), index(bind_data.index.Cast<PDXearchIndex>()),
	      preprocessed_query_embedding(make_uniq_array<float>(index.GetNumDimensions())), pdxearch_row_ids(nullptr) {

		// Preprocess the query embedding.
		EmbeddingPreprocessor embedding_preprocessor(index.GetNumDimensions(), index.GetRotationMatrix());
		embedding_preprocessor.PreprocessEmbedding(bind_data.query_embedding.get(), preprocessed_query_embedding.get(),
		                                           index.IsNormalized());

		{
			// Initialize the global heap.
			const std::lock_guard<std::mutex> lock(global_heap_mutex);
			global_heap = make_uniq<PDX::Heap<PDX::F32>>();
			global_heap->push(HEAP_INITIALIZATION_ELEMENT);
		}

		auto n_probe = index.GetEffectiveNProbe(context);
		// Assumption: all row groups have the same number of clusters.
		auto num_clusters_per_row_group = index.GetNumClustersPerRowGroup();
		partitions_to_probe_per_row_group_on_first_iteration =
		    (n_probe == 0 || n_probe > num_clusters_per_row_group) ? num_clusters_per_row_group : n_probe;

		row_group_ids_of_row_groups_with_passing_tuples.reserve(index.GetNumRowGroups());
	}

	const ClientContext &context;
	const PhysicalPDXearchIndexFilteredScan &op;
	// The limit (the K in KNN search).
	const idx_t limit;
	PDXearchIndex &index;

	const unique_ptr<float[]> preprocessed_query_embedding;

	// Global heap shared by all threads (and row groups). Assumes the `global_heap_mutex` is used.
	std::unique_ptr<PDX::Heap<PDX::F32>> global_heap;
	std::mutex global_heap_mutex;
	// Element used to initialize the global heap. PDXearch uses the top of the heap in its operations, thus there must
	// be an initial element. This element is always filtered out when the heap is transformed into a result set.
	static constexpr PDX::KNNCandidate<PDX::F32> HEAP_INITIALIZATION_ELEMENT = {1337,
	                                                                            std::numeric_limits<float>::max()};

	// For iteration support:
	// Based on the n_probe.
	idx_t partitions_to_probe_per_row_group_on_first_iteration {0};
	// The partitions to probe per iteration for each row group for all iterations except the first one. Note: the
	// number of partitions to probe on the first iteration is determined by n_probe.
	static constexpr idx_t PARTITIONS_TO_PROBE_PER_ROW_GROUP_PER_FOLLOW_UP_ITERATION = 5;
	idx_t partitions_per_row_group_probed_thus_far {0};
	// Tracked so we can avoid probing a row group with no tuples that passed the filter in the follow up iterations.
	std::vector<idx_t> row_group_ids_of_row_groups_with_passing_tuples;
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

	// Merged into the global state's `row_group_ids_of_row_groups_with_passing_tuples`. See that for more.
	std::vector<idx_t> row_group_ids_of_row_groups_with_passing_tuples;
};

unique_ptr<LocalSinkState> PhysicalPDXearchIndexFilteredScan::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PhysicalFilteredScanLocalSinkState>();
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

	const auto input_chunk_row_ids = FlatVector::GetData<row_t>(input_chunk.data[0]);
	const idx_t row_group_id = GetRowGroupId(input_chunk_row_ids[0]);
	D_ASSERT(l_sink.current_row_group_id <= row_group_id);

	// If we encounter a new row group, then initialize and perform one iteration of the filtered search for the
	// previous row group.
	if (row_group_id > l_sink.current_row_group_id && !l_sink.current_row_group_passing_rowids.empty()) {
		// Use the row ids of the rows that passed the SQL predicate and which belong to this row group to initialize
		// the filtered search for this row group.
		index.InitializeFilteredSearchForRowGroup(g_sink.preprocessed_query_embedding.get(), bind_data->limit,
		                                          l_sink.current_row_group_passing_rowids, l_sink.current_row_group_id,
		                                          *g_sink.global_heap, g_sink.global_heap_mutex);

		// Perform one iteration of filtered search (probing the next X clusters) for this row group.
		index.FilteredSearchRowGroup(l_sink.current_row_group_id,
		                             g_sink.partitions_to_probe_per_row_group_on_first_iteration);

		l_sink.row_group_ids_of_row_groups_with_passing_tuples.push_back(l_sink.current_row_group_id);

		// Clear the local state to process the next rowgroup.
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

	// If this thread's last row group has not been initialized and searched (for one iteration), do so now.
	if (!l_sink.current_row_group_passing_rowids.empty()) {
		index.InitializeFilteredSearchForRowGroup(g_sink.preprocessed_query_embedding.get(), bind_data->limit,
		                                          l_sink.current_row_group_passing_rowids, l_sink.current_row_group_id,
		                                          *g_sink.global_heap, g_sink.global_heap_mutex);
		index.FilteredSearchRowGroup(l_sink.current_row_group_id,
		                             g_sink.partitions_to_probe_per_row_group_on_first_iteration);

		l_sink.row_group_ids_of_row_groups_with_passing_tuples.push_back(l_sink.current_row_group_id);
	}

	// Merge this thread's local state into the global sink state.
	const auto guard = g_sink.Lock();
	g_sink.row_group_ids_of_row_groups_with_passing_tuples.insert(
	    g_sink.row_group_ids_of_row_groups_with_passing_tuples.end(),
	    l_sink.row_group_ids_of_row_groups_with_passing_tuples.begin(),
	    l_sink.row_group_ids_of_row_groups_with_passing_tuples.end());

	return SinkCombineResultType::FINISHED;
}

// ------------------------------
// Sink: Finalize and search iteration mechanism.
// ------------------------------

// A task that performs one iteration of the filtered search for a single row group. An iteration means that the next X
// clusters for this row group are probed.
class PhysicalFilteredScanSearchIterationTask : public ExecutorTask {
public:
	PhysicalFilteredScanSearchIterationTask(shared_ptr<Event> event_p, ClientContext &context,
	                                        PhysicalFilteredScanGlobalSinkState &g_sink_p, const PhysicalOperator &op_p,
	                                        idx_t row_group_id_p)
	    : ExecutorTask(context, std::move(event_p), op_p), g_sink(g_sink_p), row_group_id(row_group_id_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		g_sink.index.FilteredSearchRowGroup(
		    row_group_id,
		    PhysicalFilteredScanGlobalSinkState::PARTITIONS_TO_PROBE_PER_ROW_GROUP_PER_FOLLOW_UP_ITERATION);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

	string TaskType() const override {
		return "PhysicalFilteredScanSearchIterationTask";
	}

private:
	PhysicalFilteredScanGlobalSinkState &g_sink;
	idx_t row_group_id;
};

// An event that executes one search iteration in all row groups. This means that the next X clusters of each row group
// are probed.
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
		for (const idx_t &row_group_id : g_sink.row_group_ids_of_row_groups_with_passing_tuples) {
			tasks.push_back(make_uniq<PhysicalFilteredScanSearchIterationTask>(shared_from_this(), context, g_sink,
			                                                                   g_sink.op, row_group_id));
		}
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		g_sink.partitions_per_row_group_probed_thus_far +=
		    PhysicalFilteredScanGlobalSinkState::PARTITIONS_TO_PROBE_PER_ROW_GROUP_PER_FOLLOW_UP_ITERATION;
		g_sink.TryFinalizeSinkPhase(*pipeline, *this);
	}
};

SinkFinalizeType PhysicalPDXearchIndexFilteredScan::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                             OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<PhysicalFilteredScanGlobalSinkState>();

	g_sink.partitions_per_row_group_probed_thus_far += g_sink.partitions_to_probe_per_row_group_on_first_iteration;
	g_sink.TryFinalizeSinkPhase(pipeline, event);

	return SinkFinalizeType::READY;
}

// Move from the Sink phase to the Source phase if the operator is ready to begin emitting results, that is, there are K
// valid elements in the heap or if all clusters have been probed. Else, stay in the Sink phase and run another search
// iteration, which will probe the next X clusters for each row group.
void PhysicalFilteredScanGlobalSinkState::TryFinalizeSinkPhase(Pipeline &pipeline, Event &event) {
	D_ASSERT(global_heap->size() <= limit);
	D_ASSERT(partitions_per_row_group_probed_thus_far <= index.GetNumClustersPerRowGroup());

	// The heap (and thus pruning threshold) is initialized with a max float element. This float element should not be
	// part of the result (it is not valid). There is an edge case where this element is the Kth item (at the top of the
	// heap), thus we should run another search iteration, as it might find a valid Kth item from the next search
	// iteration.
	const bool is_initialization_element_at_top_of_heap =
	    this->global_heap->top().distance == HEAP_INITIALIZATION_ELEMENT.distance;
	const bool is_heap_filled_with_k_valid_results =
	    this->global_heap->size() == limit && !is_initialization_element_at_top_of_heap;
	const bool are_all_partitions_probed =
	    partitions_per_row_group_probed_thus_far >= index.GetNumClustersPerRowGroup();

	if (is_heap_filled_with_k_valid_results || are_all_partitions_probed) {
		// If we are done, then prepare emission of the results by moving the result row ids into the Source state.
		const auto result_rowids = PDX::PDXearch<PDX::F32>::BuildResultSetFromHeap(limit, *this->global_heap);
		this->pdxearch_row_ids = make_uniq<std::vector<row_t>>(result_rowids.size());
		for (size_t i = 0; i < result_rowids.size(); i++) {
			(*this->pdxearch_row_ids)[i] = result_rowids[i].index;
		}
		// We return such that the Sink phase completes, allowing the Source phase to start.
		return;
	}

	// Else, run another iteration of filtered search on all rowgroups. This visits the next clusters of each rowgroup.
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

SourceResultType PhysicalPDXearchIndexFilteredScan::GetData(ExecutionContext &context, DataChunk &output_chunk,
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
	InsertionOrderPreservingMap<string> result;
	result["Table"] = bind_data->table.name;
	result["PDXearch Index"] = bind_data->index.GetIndexName();
	result["Total Clusters"] =
	    StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumClustersPerRowGroup() *
	                                  bind_data->index.Cast<PDXearchIndex>().GetNumRowGroups());
	result["Row Groups"] = StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumRowGroups());
	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb
