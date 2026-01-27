# This file is included by DuckDB's build system. It specifies which extension to load

# Force CXX_STANDARD from 11 to 17 as PDXearch currently uses C++17 features.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to enforce" FORCE)

# Extension from this repo
duckdb_extension_load(pdxearch
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Any extra extensions that should be built
# e.g.: duckdb_extension_load(json)