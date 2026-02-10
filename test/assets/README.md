# Test Assets

## Data Sets

The SIFT dataset included in this repository has 128 dimensions and 245_760 rows
(2 row groups). The parquet file has the `id` (0, 1, ...) and `emb` column. Note
that the dataset's 128-dimensional `emb` array column is composed integers.
These integers are converted to float when loaded into DuckDB. Details about the
original dataset can be found at
[corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/).
