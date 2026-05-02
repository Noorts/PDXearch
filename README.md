<h1 align="center">
  The PDXearch DuckDB Extension
</h1>
<h3 align="center">
  A state-of-the-art IVF index for lightweight but fast (filtered) vector similarity search.
</h3>
<br>

- [Why PDXearch?](#why-pdxearch)
- [Install](#install)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Configuration](#configuration)
    - [Index Creation](#index-creation)
    - [Index Search](#index-search)
    - [Index Metadata](#index-metadata)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Why PDXearch?

DuckDB offers vector similarity search (VSS) out of the box, through its
fixed-size `ARRAY` column type and distance functions
([docs](https://duckdb.org/docs/stable/sql/data_types/array#functions)). These
functions return exact results, but are often too slow on large datasets.

The official DuckDB [VSS extension](https://duckdb.org/docs/stable/core_extensions/vss)
introduces a graph-based (HNSW) VSS index. You can create this index on your
table to speed up the vector search queries. Unfortunately, although
these graph-based indexes deliver fast search, they take up a considerable
amount of memory and take long to construct.

The PDXearch extension aims to address these drawbacks. It achieves competitive
search performance, while using less memory and being significantly faster to
construct. This is made possible by a state-of-the-art partition-based (IVF)
index. To be precise, we rely on the CWI's [PDX](https://github.com/cwida/pdx)
data layout and the accompanying search framework called PDXearch. Furthermore,
this extension integrates tightly with DuckDB's internals to parallelize across
row groups, allowing us to squeeze more performance out of modern hardware.

## Install

> [!WARNING]
> The extension is unstable and experimental. We're actively working on adding
> features and improving stability. The extension will be made available as a
> community extension once it's ready. For now the extension has to be built
> locally.

To build the extension locally, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Usage

### Getting Started

Our syntax is almost identical to that of the
official VSS extension ([VSS docs](https://duckdb.org/docs/stable/core_extensions/vss)).

1. Start a DuckDB instance with an in-memory database and allow loading unsigned extensions.

    ```bash
    duckdb -unsigned
    ```

2. Load the locally built extension by providing a full path to it.

    ```sql
    LOAD '<Fill in>/PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension';
    ```

3. Ensure you have a table with a fixed-sized `FLOAT[num-dims]` column for your embeddings (below called `embedding`). The table can have any number of other columns. We currently do not support `NULL` values in the `embedding` column.

    ```sql
    CREATE TABLE t1 (id INTEGER, embedding FLOAT[512]);
    ```

    ```sql
    INSERT INTO t1 (id, embedding) SELECT i as id, repeat([i], 512) FROM range(20000) t(i);
    ```

4. Create the PDXearch index to speed up vector similarity search queries. Optionally, configure the index (e.g., the `metric` option). For all options see the [configuration](#configuration) section.

    ```sql
    CREATE INDEX t1_idx ON t1 USING PDXEARCH (embedding) WITH (metric = 'l2sq');
    ```

5. Run an approximate filtered vector similarity search where the top 100 rows are returned.

    ```sql
    SELECT * FROM t1 WHERE id < 500
        ORDER BY array_distance(embedding, repeat([1000.51], 512)::FLOAT[512]) LIMIT 100;
    ```

> [!WARNING]
> If you're executing (filtered) search queries where `K <= 50`, then please
> disable DuckDB's late materialization optimization by running the following
> statement prior to your search: `SET late_materialization_max_rows = 0;`. Due
> to the query's low LIMIT (K), DuckDB will apply a late materialization
> optimization. Unfortunately, the extension does not handle this case optimally
> yet, leading to a suboptimal query plan when a `K <= 50` VSS query is
> optimized. We aim to address this behavior in the near future.

### Configuration

#### Index Creation

As shown below, during index creation you can set index creation options in the `WITH` clause. These options cannot be modified after index creation. Drop and recreate the index instead.

```sql
CREATE INDEX t1_idx ON t1 USING PDXEARCH (vec) WITH (metric = 'l2sq', quantization = 'f32');
```

Available options:

- `metric`
  - The distance metric this index speeds up. One index can only optimize one distance metric. If you want two or more distance metrics to be optimized for the same column, then create multiple indexes, where the `metric` option differs.
  - `'l2sq'` (*default*; Euclidean distance, optimizes `array_distance`), `'cosine'` (Cosine similarity distance, optimizes `array_cosine_distance`). Inner product distance is not supported yet.
- `quantization`
  - The precision of the embeddings stored inside the index. Using quantization decreases search latency and index size, but also slightly decreases recall.
  - `f32` (full precision, 4 bytes), `u8` (*default*; scalar quantization, 1 byte).
- `n_probe`
  - Determines the number of partitions/clusters/lists that are explored. Increasing `n_probe` increases the effort spent during a search, thus likely increasing recall, but also increasing search latency. Setting this `n_probe` option will store this value in the index. It can be temporarily overwritten at search time using `pdxearch_n_probe` (see below).
  - `[0, 2147483647]`. *Default* is `24`. This is per row group, which likely has 480 lists. Set `n_probe` to `0` to ensure all clusters are probed. If `n_probe` exceeds the index's number of lists, then all clusters will be probed.
- `seed`
  - The index uses RNG for some internal mechanisms. Set the seed to make behavior reproducible (e.g., for tests or bugs).
  - `[-2147483647, 2147483647]`. *Default* is random.

> [!NOTE]
> There is currently no option to set the number of lists/partitions/clusters manually. We set this [automatically](https://github.com/Noorts/PDXearch/blob/e994ac5f8a99fa3467e18670f4e8056fd5ad9572/src/include/index/pdxearch_wrapper.hpp#L131-L146) based on the row group's size.

#### Index Search

DuckDB by default will use all available threads. To set the number of threads to 1, use `SET threads = 1;`. See the [DuckDB documentation](https://duckdb.org/docs/stable/sql/statements/set) for more information.

Before running a search query, you can set the number of clusters to probe to 48 using `SET pdxearch_n_probe = 48`. This will temporarily overwrite the `n_probe` value stored in the index. Use `RESET pdxearch_n_probe` to unset this overwrite. For more information, see the `n_probe` description above.

Prepend your search query with `EXPLAIN` to show the optimized query plan of your vector search query. Optimized query plans will include a `PDXEARCH_INDEX_SCAN` or a `PDXEARCH_INDEX_FILT_SCAN` operator. For more information about `EXPLAIN` see the [DuckDB documentation](https://duckdb.org/docs/stable/guides/meta/explain).

#### Index Metadata

Execute `CALL pdxearch_index_info();` to print metadata about all PDXearch indexes. This includes an approximate lower bound on the index's size in memory.

Execute `FROM duckdb_indexes();` for general information about all indexes.

## Limitations

The extension's functionality is limited, as it is still in early development.
As mentioned above, we aim to address all of these limitations soon.

- **No persistence**: The index should only be created in in-memory DuckDB
  databases. For disk-resident databases you'll have to manually drop and
  rebuild the index when you reload the database (to avoid loading a malformed
  index from storage).

- **No maintenance**: We currently only support creating an index on static
  collections. This means the index does not yet support updating the index when
  a `INSERT INTO` or `DELETE FROM` statement is invoked on the table.

- **No concurrency**: We do not support concurrent index access yet.

- **Requires full row groups**: The extension currently requires all but the
  last row group to be completely filled with rows. For example, three row
  groups where they have 122880, 122880, 4000 rows respectively is valid.
  Inserting rows in batches of 122880 can help to create such a layout. This is
  a limitation we aim to address very soon.

- **Late materialization and filter types**: As noted above, we don't optimally
  handle DuckDB's late materialization optimizer rule yet. Furthermore, on a
  related note, we currently only support filtered vector similarity queries
  where DuckDB pushes the entire filter down into the sequential scan. This is
  not a limitation of our design. We plan to adjust our scan optimizer such that
  we can process SQL queries with arbitrary predicates. You can check whether
  your query is currently being optimized by prepending the `EXPLAIN` keyword to
  your search query and checking if a PDXearch operator is part of the query
  plan.

- **Configuration options**: The available [configuration options](#configuration) are currently
  limited (e.g., quantization, distance functions, normalization).

- **Stability**

## Acknowledgements

The extension would not be possible without the underlying technologies and the
lessons learned from other extensions.

- **[PDX](https://github.com/cwida/pdx)**: We use the PDX data layout and
  PDXearch framework.

- **[Super K-Means](https://github.com/lkuffo/SuperKMeans)**: We use
  the Super K-Means library for fast k-means clustering.

- **[VSS](https://github.com/duckdb/duckdb-vss)**: We've taken inspiration from
  the VSS interface and we reuse parts of the VSS extension's code.

## License

The extension is licensed under the [MIT license](LICENSE).
