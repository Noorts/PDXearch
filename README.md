<h1 align="center">
  The PDXearch DuckDB Extension
</h1>
<h3 align="center">
  A state-of-the-art IVF index for lightweight but fast (filtered) vector similarity search.
</h3>
<br>

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

To create an index and run a search, we provide an interface similar to the
official VSS extension ([VSS docs](https://duckdb.org/docs/stable/core_extensions/vss)).

1. Start a DuckDB instance with an in-memory database and allow loading unsigned extensions.

    ```bash
    duckdb -unsigned
    ```

2. Load the locally built extension by providing a full path to it.

    ```sql
    LOAD '<Fill in>/PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension';
    ```

3. Set up a table.

    ```sql
    CREATE TABLE t1 (id INTEGER, vec FLOAT[512]);
    ```

    ```sql
    INSERT INTO t1 (id, vec) SELECT i as id, repeat([i], 512) FROM range(20000) t(i);
    ```

4. Create the PDXearch index and set one of the index's options (n_probe).

    ```sql
    CREATE INDEX t1_idx ON t1 USING PDXEARCH (vec) WITH (n_probe = 64);
    ```

5. Run an approximate filtered vector similarity search where the top 100 rows are returned.

    ```sql
    SELECT * FROM t1 WHERE id < 500
        ORDER BY array_distance(vec, repeat([1000.51], 512)::FLOAT[512]) LIMIT 100;
    ```

> [!WARNING]
> If you're executing (filtered) search queries where `K <= 50`, then please
> disable DuckDB's late materialization optimization by running the following
> statement prior to your search: `SET late_materialization_max_rows = 0;`. Due
> to the query's low LIMIT (K), DuckDB will apply a late materialization
> optimization. Unfortunately, the extension does not handle this case optimally
> yet, leading to a suboptimal query plan when a `K <= 50` VSS query is
> optimized. We aim to address this behavior in the near future.

## Limitations

The extension's functionality is limited, as it is still in early development.
As mentioned above, we aim to address these limitations soon.

- **No persistence**: The index should only created in in-memory DuckDB
  databases. For disk-resident databases you'll have to manually drop and
  rebuild an index when you reload a database.

- **No maintenance**: We currently only support creating an index on static
  collections. This means the index does not yet support updating the index when
  a `INSERT INTO` or `DELETE FROM` statement is invoked on the table.

- **Configuration options**: The available configuration options are currently
  limited (e.g., quantization, distance functions, normalization).

- **Stability**

## Acknowledgements

The extension would not be possible without the underlying technologies and the
lessons learned from other extensions.

- **[PDX](https://github.com/cwida/pdx)**: We use the PDX data layout and
  PDXearch framework.

- **[VSS](https://github.com/duckdb/duckdb-vss)**: We've taken inspiration from
  the VSS interface and we reuse parts of the VSS extension's code.

- **[Faiss](https://github.com/facebookresearch/faiss)**: We currently use
  Meta's Faiss library for its K-means capabilities.

- **[DuckDB-Faiss](https://github.com/duckdb-faiss-ext/duckdb-faiss-ext)**:
  We've taken inspiration from its build setup to include Faiss in the
  extension. This is yet another alternative DuckDB VSS extension.

## License

The extension is licensed under the [MIT license](LICENSE).
