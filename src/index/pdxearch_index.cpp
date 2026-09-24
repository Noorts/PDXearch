#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "pdx/common.hpp"
#include "index/pdxearch_index.hpp"

#include "index/pdxearch_module.hpp"

#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

PDXearchIndex::PDXearchIndex(const string &name, IndexConstraintType index_constraint_type,
                             const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                             const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                             const case_insensitive_map_t<Value> &index_creation_options,
                             const IndexStorageInfo &persistence_info)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {
	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("PDXearch indexes do not support unique or primary key constraints");
	}
	if (persistence_info.IsValid()) {
		throw InternalException(
		    "Something went wrong: the PDXearch index was created without its table. This is likely because a "
		    "malformed persisted index was loaded. Index persistence is not supported yet, but DuckDB will still "
		    "try to persist it. Open your database file manually (duckdb test.db) and drop the index(es). Run "
		    "'SELECT sql FROM duckdb_indexes();' to see the indexes, and then 'DROP INDEX index_name;' to drop the "
		    "unused index(es).");
	}

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	const auto &embedding_type = logical_types[0];
	D_ASSERT(embedding_type.id() == LogicalTypeId::ARRAY);

	const auto num_dimensions = ArrayType::GetSize(embedding_type);

	// Try to get the vector metric from the options, this parameter should be verified during binding.
	auto dist_metric = PDXearchWrapper::DEFAULT_DISTANCE_METRIC;
	const auto dist_metric_opt = index_creation_options.find("metric");
	if (dist_metric_opt != index_creation_options.end()) {
		const auto dist_metric_val =
		    PDXearchIndex::DISTANCE_METRIC_MAP.find(dist_metric_opt->second.GetValue<string>());
		if (dist_metric_val != PDXearchIndex::DISTANCE_METRIC_MAP.end()) {
			dist_metric = dist_metric_val->second;
		}
	}

	auto quantization = PDXearchWrapper::DEFAULT_QUANTIZATION;
	const auto quantization_opt = index_creation_options.find("quantization");
	if (quantization_opt != index_creation_options.end()) {
		const auto quantization_val = PDXearchIndex::QUANTIZATION_MAP.find(quantization_opt->second.GetValue<string>());
		if (quantization_val != PDXearchIndex::QUANTIZATION_MAP.end()) {
			quantization = quantization_val->second;
		}
	}

	auto n_probe = PDXearchWrapper::DEFAULT_N_PROBE;
	const auto n_probe_opt = index_creation_options.find("n_probe");
	if (n_probe_opt != index_creation_options.end()) {
		n_probe = n_probe_opt->second.GetValue<int32_t>();
	}

	// TODO: Confirm the static cast is sound.
	auto seed = static_cast<int32_t>(std::random_device {}());
	const auto seed_opt = index_creation_options.find("seed");
	if (seed_opt != index_creation_options.end()) {
		seed = seed_opt->second.GetValue<int32_t>();
	}

	const idx_t row_group_size = table_io_manager.GetRowGroupSize();
	if (quantization == PDX::Quantization::F32) {
		D_ASSERT(ArrayType::GetChildType(embedding_type).id() == LogicalTypeId::FLOAT);

		pdxearch_wrapper = make_uniq<PDXearchWrapperF32>(dist_metric, num_dimensions, n_probe, seed, row_group_size);
	} else if (quantization == PDX::Quantization::U8) {
		pdxearch_wrapper = make_uniq<PDXearchWrapperU8>(dist_metric, num_dimensions, n_probe, seed, row_group_size);
	} else {
		throw InternalException("Unsupported quantization: %s", quantization);
	}

	function_matcher = MakeFunctionMatcher(*pdxearch_wrapper.get());
	embedding_preprocessor = make_uniq<EmbeddingPreprocessor>(num_dimensions, pdxearch_wrapper->GetRotationMatrix());
}

/******************************************************************
 * Index creation and search methods specific to the parallel implementation
 ******************************************************************/

bool PDXearchIndex::IsInSyncWithTable(DataTable &table) const {
	return !has_unindexed_rows && FindStaleRowGroups(table).empty();
}

unique_ptr<StorageLockKey> PDXearchIndex::SyncAndLockForSearch(DataTable &table) {
	bool in_sync;
	{
		auto _lock = rwlock.GetSharedLock();
		in_sync = IsInSyncWithTable(table);
	}
	if (!in_sync) {
		SyncWithTable(table);
	}
	return rwlock.GetSharedLock();
}

idx_t PDXearchIndex::GetRowGroupSize() const {
	return table_io_manager.GetRowGroupSize();
}

bool PDXearchIndex::TryGetPhysicalRowGroup(DataTable &table, const row_t row_id, PDXearchRowGroupBounds &result) const {
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	idx_t segment_index;
	if (!row_groups->TryGetSegmentIndex(lock, static_cast<idx_t>(row_id), segment_index)) {
		return false;
	}
	auto node = row_groups->GetSegmentByIndex(lock, static_cast<int64_t>(segment_index));
	result = {static_cast<row_t>(node->GetRowStart()), node->GetCount()};
	return true;
}

vector<PDXearchRowGroupBounds> PDXearchIndex::GetPhysicalRowGroups(DataTable &table) const {
	vector<PDXearchRowGroupBounds> result;
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	for (auto node = row_groups->GetRootSegment(lock); node; node = row_groups->GetNextSegment(lock, *node)) {
		result.push_back({static_cast<row_t>(node->GetRowStart()), node->GetCount()});
	}
	return result;
}

optional_idx PDXearchIndex::LookupRowGroup(const row_t row_id) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->LookupRowGroup(row_id);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->LookupRowGroup(row_id);
}

PDXearchRowRange PDXearchIndex::GetRowGroupRange(const idx_t row_group_idx) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		const auto &row_group = static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetRowGroup(row_group_idx);
		return {row_group.row_start, row_group.row_end};
	}
	const auto &row_group = static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetRowGroup(row_group_idx);
	return {row_group.row_start, row_group.row_end};
}

void PDXearchIndex::AppendRow(const idx_t row_group_idx, const row_t row_id, const float *const transformed_embedding) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->AppendRow(row_group_idx, row_id, transformed_embedding);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->AppendRow(row_group_idx, row_id, transformed_embedding);
	}
}

void PDXearchIndex::DeleteRow(const row_t row_id) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->DeleteRow(row_id);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->DeleteRow(row_id);
	}
}

vector<PDXearchRowGroupBounds> PDXearchIndex::FindStaleRowGroups(DataTable &table) const {
	const auto physical_row_groups = GetPhysicalRowGroups(table);
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->FindStaleRowGroups(physical_row_groups);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->FindStaleRowGroups(physical_row_groups);
}

void PDXearchIndex::RemoveRowGroupsOverlapping(const row_t start, const row_t end) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->RemoveRowGroupsOverlapping(start, end);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->RemoveRowGroupsOverlapping(start, end);
	}
}

idx_t PDXearchIndex::FetchRows(DataTable &table, const row_t start, const row_t end, row_t *const row_ids,
                               float *const embeddings, idx_t &rows_returned) {
	rows_returned = 0;
	// Fetches every committed row, whatever the calling transaction's snapshot. The mirror
	// holds the committed state, and rows a query must not see are dropped when it fetches its results.
	const TransactionData committed(MAX_TRANSACTION_ID, DuckTransactionManager::Get(db).GetLastCommit() + 1);
	const vector<StorageIndex> fetch_column_ids {StorageIndex(GetColumnIds()[0]), StorageIndex()};
	DataChunk fetched;
	fetched.Initialize(Allocator::Get(db), {logical_types[0], LogicalType::ROW_TYPE});
	Vector fetch_row_ids(LogicalType::ROW_TYPE, STANDARD_VECTOR_SIZE);
	const auto fetch_row_ids_data = FlatVector::GetData<row_t>(fetch_row_ids);
	ColumnFetchState fetch_state;
	const auto num_dimensions = GetNumDimensions();
	auto raw_embeddings = make_uniq_array<float>(STANDARD_VECTOR_SIZE * num_dimensions);

	idx_t count = 0;
	for (row_t batch_start = start; batch_start < end; batch_start += STANDARD_VECTOR_SIZE) {
		const idx_t batch_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, static_cast<idx_t>(end - batch_start));
		for (idx_t i = 0; i < batch_size; i++) {
			fetch_row_ids_data[i] = batch_start + static_cast<row_t>(i);
		}
		fetched.Reset();
		// Deleted rows are gone, NULL embeddings are NULL.
		table.GetRowGroupCollection()->Fetch(committed, fetched, fetch_column_ids, fetch_row_ids, batch_size,
		                                     fetch_state);
		const idx_t fetched_count = fetched.size();
		rows_returned += fetched_count;
		if (fetched_count == 0) {
			continue;
		}
		auto &embedding_column = fetched.data[0];
		embedding_column.Flatten(fetched_count);
		fetched.data[1].Flatten(fetched_count);
		const auto &validity = FlatVector::Validity(embedding_column);
		const auto fetched_embeddings = FlatVector::GetData<float>(ArrayVector::GetEntry(embedding_column));
		const auto fetched_row_ids = FlatVector::GetData<row_t>(fetched.data[1]);
		idx_t batch_count = 0;
		for (idx_t i = 0; i < fetched_count; i++) {
			if (!validity.RowIsValid(i)) {
				continue;
			}
			memcpy(raw_embeddings.get() + batch_count * num_dimensions, fetched_embeddings + i * num_dimensions,
			       num_dimensions * sizeof(float));
			row_ids[count + batch_count] = fetched_row_ids[i];
			batch_count++;
		}
		if (batch_count > 0) {
			embedding_preprocessor->PreprocessEmbeddings(raw_embeddings.get(), embeddings + count * num_dimensions,
			                                             batch_count, IsNormalized());
			count += batch_count;
		}
	}
	return count;
}

void PDXearchIndex::SyncWithTable(DataTable &table) {
	auto _lock = rwlock.GetExclusiveLock();

	// One DuckDB row group of transformed embeddings at a time, like the create sink.
	const auto num_dimensions = GetNumDimensions();
	auto embeddings = make_uniq_array<float>(GetRowGroupSize() * num_dimensions);
	auto row_ids = make_uniq_array<row_t>(GetRowGroupSize());

	// Row groups a checkpoint merged or dropped: their mirrors go, and the ones still in the table are rebuilt.
	idx_t rows_returned;
	for (const auto &stale : FindStaleRowGroups(table)) {
		RemoveRowGroupsOverlapping(stale.row_start, stale.row_start + static_cast<row_t>(stale.count));
		PDXearchRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, stale.row_start, bounds)) {
			continue;
		}
		const idx_t count = FetchRows(table, bounds.row_start, bounds.row_start + static_cast<row_t>(bounds.count),
		                              row_ids.get(), embeddings.get(), rows_returned);
		if (count > 0) {
			SetUpIndexForRowGroup(row_ids.get(), embeddings.get(), count, bounds.row_start, bounds.count);
		}
	}

	// Unindexed rows, gathered per DuckDB row group: a row group without a mirror is clustered once from all of its
	// rows. Each range is retired on its own because rows of one commit become visible together.
	std::vector<PDXearchRowRange> still_unindexed;
	idx_t range_idx = 0;
	while (range_idx < unindexed_row_ranges.size()) {
		PDXearchRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, unindexed_row_ranges[range_idx].start, bounds)) {
			// The rows are not in the table yet: their commit is still in flight.
			still_unindexed.push_back(unindexed_row_ranges[range_idx++]);
			continue;
		}
		const row_t bounds_end = bounds.row_start + static_cast<row_t>(bounds.count);
		idx_t count = 0;
		while (range_idx < unindexed_row_ranges.size() && unindexed_row_ranges[range_idx].start < bounds_end) {
			auto &range = unindexed_row_ranges[range_idx];
			const row_t sub_end = MinValue<row_t>(range.end, bounds_end);
			const idx_t range_count = FetchRows(table, range.start, sub_end, row_ids.get() + count,
			                                    embeddings.get() + count * num_dimensions, rows_returned);
			if (rows_returned == 0) {
				// Not committed yet: a later sync indexes them.
				still_unindexed.push_back({range.start, sub_end});
			}
			count += range_count;
			if (sub_end < range.end) {
				range.start = sub_end;
				break;
			}
			range_idx++;
		}
		if (count == 0) {
			continue;
		}
		const auto row_group_idx = LookupRowGroup(bounds.row_start);
		if (row_group_idx.IsValid()) {
			for (idx_t i = 0; i < count; i++) {
				AppendRow(row_group_idx.GetIndex(), row_ids[i], embeddings.get() + i * num_dimensions);
			}
		} else {
			SetUpIndexForRowGroup(row_ids.get(), embeddings.get(), count, bounds.row_start, bounds.count);
		}
	}
	unindexed_row_ranges = std::move(still_unindexed);
	has_unindexed_rows = !unindexed_row_ranges.empty();
}

void PDXearchIndex::SetUpIndexForRowGroup(const row_t *const row_ids, const float *const vectors,
                                          const idx_t num_vectors, const row_t row_start, const idx_t count) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_start, count);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_start, count);
	}
}

unique_ptr<PDX::IIterativeSearch>
PDXearchIndex::BeginSearchForRowGroup(const idx_t row_group_idx, const float *const preprocessed_query,
                                      const idx_t limit, PDX::TopKHeap &top_k_heap,
                                      const std::vector<size_t> *const passing_row_ids) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->BeginSearchForRowGroup(row_group_idx, preprocessed_query, limit, top_k_heap, passing_row_ids);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
	    ->BeginSearchForRowGroup(row_group_idx, preprocessed_query, limit, top_k_heap, passing_row_ids);
}

/******************************************************************
 * Index maintenance
 ******************************************************************/

ErrorData PDXearchIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	// Only the row ids are needed, so the index expressions are not evaluated.
	return Insert(lock, entries, row_ids);
}

// Called by DuckDB at commit with the final row ids, before the rows enter the table's row groups. Which row group
// they land in is not knowable here, so only the ids are recorded; SyncWithTable indexes them once the table knows.
ErrorData PDXearchIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	auto _lock = rwlock.GetExclusiveLock();

	const idx_t count = data.size();
	if (count == 0) {
		return ErrorData();
	}
	row_ids.Flatten(count);
	const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
	// Determine consecutive ranges of row ids.
	// For example: `1, 2, 3, 5, 6 --> [1, 3] + [5, 6]`
	PDXearchRowRange range {row_id_data[0], row_id_data[0] + 1};
	for (idx_t i = 1; i < count; i++) {
		if (row_id_data[i] == range.end) {
			range.end++;
			continue;
		}
		unindexed_row_ranges.push_back(range);
		range = {row_id_data[i], row_id_data[i] + 1};
	}
	unindexed_row_ranges.push_back(range);
	has_unindexed_rows = true;
	return ErrorData();
}

// Called by DuckDB at commit. Row ids that are not in the index (never indexed, or reverted before they were) are
// skipped.
void PDXearchIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	auto _lock = rwlock.GetExclusiveLock();

	const idx_t count = entries.size();
	row_ids.Flatten(count);
	const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
	for (idx_t i = 0; i < count; i++) {
		// PDX Delete is idempotent: if the row was never indexed, it is a no-op
		DeleteRow(row_id_data[i]);
		RemoveUnindexedRow(row_id_data[i]);
	}
	has_unindexed_rows = !unindexed_row_ranges.empty();
}

// A deleted row that was never indexed has nothing left to index: take it out of its staged range.
void PDXearchIndex::RemoveUnindexedRow(const row_t row_id) {
	// Find staged range [start, end) that contains row_id, if any
	auto it = std::upper_bound(unindexed_row_ranges.begin(), unindexed_row_ranges.end(), row_id,
	                           [](row_t id, const PDXearchRowRange &range) { return id < range.start; });
	if (it == unindexed_row_ranges.begin()) {
		return;
	}
	--it;
	// row_id is not staged if it falls in a gap
	if (row_id >= it->end) {
		return;
	}
	// row_id is the first of the range, we just shrink the range by moving the start up one
	if (it->start == row_id) {
		it->start++;
	} else if (it->end == row_id + 1) { // idem (with the tail)
		it->end--;
	} else {
		// row_id is in the middle of the range, we split it into two ranges
		const PDXearchRowRange tail {row_id + 1, it->end};
		it->end = row_id;
		unindexed_row_ranges.insert(it + 1, tail);
		return;
	}
	// If the range is now empty, remove it
	if (it->start == it->end) {
		unindexed_row_ranges.erase(it);
	}
}

void PDXearchIndex::ResetStorage(IndexLock &lock) {
	auto _lock = rwlock.GetExclusiveLock();

	// TODO: Implement when we implement persistence.
}

bool PDXearchIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("PDXearchIndex::MergeIndexes() not implemented");
}

void PDXearchIndex::Vacuum(IndexLock &state) {
}

void PDXearchIndex::Verify(IndexLock &state) {
	throw NotImplementedException("PDXearchIndex::Verify() not implemented");
}

string PDXearchIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("PDXearchIndex::ToString() not implemented");
}

void PDXearchIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("PDXearchIndex::VerifyAllocations() not implemented");
}

idx_t PDXearchIndex::GetInMemorySize(IndexLock &state) {
	auto _lock = rwlock.GetSharedLock();

	return pdxearch_wrapper->GetInMemorySizeInBytes();
}

unique_ptr<PDXearchIndexStats> PDXearchIndex::GetStats(const ClientContext &context) const {
	auto result = make_uniq<PDXearchIndexStats>();

	// Make sure to sync any changes made here to `pdxearch_index_info_function.cpp`.
	result->metric = GetDistanceMetric();
	result->quantization = GetQuantization();
	result->num_dimensions = static_cast<int64_t>(GetNumDimensions());
	result->n_probe = static_cast<int64_t>(pdxearch_wrapper->GetNProbe());
	result->seed = static_cast<int64_t>(pdxearch_wrapper->GetSeed());
	result->is_normalized = IsNormalized();
	result->approximate_lower_bound_memory_usage_bytes =
	    static_cast<int64_t>(pdxearch_wrapper->GetInMemorySizeInBytes());

	return result;
}

/******************************************************************
 * Index persistence
 ******************************************************************/

IndexStorageInfo PDXearchIndex::SerializeToDisk(QueryContext context,
                                                const case_insensitive_map_t<Value> &serialization_options) {
	// For serialization_options see:
	// https://github.com/duckdb/duckdb/blob/32afee3e788394973ce4df4fcae7610832d5550a/src/storage/write_ahead_log.cpp#L374

	PersistToDisk();

	IndexStorageInfo info(name);
	case_insensitive_map_t<Value> options;
	options.emplace("testDisk", Value::INTEGER(12));
	info.options = options;

	// Temporary empty FixedSizeAllocatorInfo to satisfy the DuckDB RelDebug build's index_storage_info.IsValid() check.
	info.allocator_infos.push_back(FixedSizeAllocatorInfo {});

	return info;
}

IndexStorageInfo PDXearchIndex::SerializeToWAL(const case_insensitive_map_t<Value> &serialization_options) {
	PersistToDisk();

	IndexStorageInfo info(name);
	case_insensitive_map_t<Value> options;
	options.emplace("testWAL", Value::INTEGER(12));
	info.options = options;

	// Temporary empty FixedSizeAllocatorInfo to satisfy the DuckDB RelDebug build's index_storage_info.IsValid() check.
	info.allocator_infos.push_back(FixedSizeAllocatorInfo {});

	return info;
}

// TODO: Implement persistence.
void PDXearchIndex::PersistToDisk() {
	auto _lock = rwlock.GetExclusiveLock();
}

/******************************************************************
 * Misc.
 ******************************************************************/

bool PDXearchIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                             vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

static void TryBindIndexExpressionInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		// Rewrite the column reference to fit in the current set of bound column ids
		ref.binding.table_index = table_idx;

		const auto referenced_column = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

bool PDXearchIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

string PDXearchIndex::GetQuantization() const {
	switch (pdxearch_wrapper->GetQuantization()) {
	case PDX::Quantization::F32:
		return "f32";
	case PDX::Quantization::U8:
		return "u8";
	default:
		throw InternalException("Unknown quantization");
	}
}

const case_insensitive_map_t<PDX::Quantization> PDXearchIndex::QUANTIZATION_MAP = {
    {"f32", PDX::Quantization::F32},
    {"u8", PDX::Quantization::U8},
};

string PDXearchIndex::GetDistanceMetric() const {
	switch (pdxearch_wrapper->GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		return "l2sq";
	case PDX::DistanceMetric::COSINE:
		return "cosine";
	// case PDX::DistanceMetric::IP:
	// 	return "ip";
	default:
		throw InternalException("Unknown distance metric");
	}
}

const case_insensitive_map_t<PDX::DistanceMetric> PDXearchIndex::DISTANCE_METRIC_MAP = {
    {"l2sq", PDX::DistanceMetric::L2SQ}, {"cosine", PDX::DistanceMetric::COSINE},
    // {"ip", PDX::DistanceMetric::IP},
};

unique_ptr<ExpressionMatcher> PDXearchIndex::MakeFunctionMatcher(const PDXearchWrapper &pdxearch_wrapper) {
	unordered_set<string> distance_functions;

	switch (pdxearch_wrapper.GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		distance_functions = {"array_distance", "<->"};
		break;
	case PDX::DistanceMetric::COSINE:
		distance_functions = {"array_cosine_distance", "<=>"};
		break;
	// case PDX::DistanceMetric::IP:
	// 	distance_functions = {"array_negative_inner_product", "<#>"};
	//  break;
	default:
		throw NotImplementedException("Unknown distance metric");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto array_type = LogicalType::ARRAY(LogicalType::FLOAT, pdxearch_wrapper.GetNumDimensions());

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

void PDXearchModule::RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;

	index_type.name = PDXearchIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<PDXearchIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                    input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = PDXearchIndex::CreatePlan;

	db.config.AddExtensionOption("pdxearch_n_probe",
	                             "override the n_probe parameter when scanning PDXearch indexes (default: " +
	                                 to_string(PDXearchWrapper::DEFAULT_N_PROBE) + ", must be >= 0)",
	                             LogicalType::INTEGER, Value());

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb
