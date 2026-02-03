# Development

## Build

The extension relies on a handful of dependencies. They are included in
a variety of ways, including being copied in (PDXearch headers; AKA vendoring), included as a git submodule (Faiss, DuckDB), and included through the [vcpkg dependency manager](vcpkg.io/) (see [vcpkg.json](vcpkg.json); Eigen3, BLAS, LAPACK). Some need to be available in the environment (OpenMP).

### Building the extension

1. Clone the repo:

    ```sh
    git clone --recurse-submodules https://github.com/Noorts/PDXearch.git
    ```

2. Install vcpkg ([detailed instructions](https://github.com/duckdb/extension-template?tab=readme-ov-file#managing-dependencies)):

    ```sh
    git clone https://github.com/Microsoft/vcpkg.git
    ```

    ```sh
    ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
    ```

    ```sh
    export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake
    ```

3. On macOS make sure you have `libomp` installed (for OpenMP):

    ```sh
    brew install libomp
    ```

    Note: You might have to set `OpenMP_ROOT` in your `.zshrc` file.

    ```sh
    export OpenMP_ROOT=$(brew --prefix)/opt/libomp
    ```

4. Build the extension (the default is an optimized `release` build):

    ```sh
    cd PDXearch
    ```

    ```sh
    make
    ```

    Other build modes include `debug` and `reldebug`.

    ```sh
    make debug
    ```

5. [Recommended] For faster builds, install [ccache](https://ccache.dev/) and [ninja](https://ninja-build.org/), and then set the generator:

    ```sh
    GEN=ninja make
    ```

The built extension can be found at `PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension`.

For extension usage instructions, please refer to the [README.md](README.md).

### PDXearch variants

The extension's internals have been implemented in two variants. These variants
both implement index creation, and filtered AND non-filtered search.

**1. Row group**: The first variant creates a separate internal IVF index for each
row group. This allows the extension to parallelize across row groups, which
speeds up both the creation of the index and using it for search.

**2. Global**: The second variant uses a single internal IVF index for all
embeddings in the targeted table column.

From the user's perspective the creation of an index (`CREATE INDEX`) and using
it for search is the same. The user always interacts with a single database
index.

You decide which variant to use at compile time. Row group is the default. Build
the extension with the following argument to use the global variant:

```sh
EXT_FLAGS="-DPDX_USE_ALTERNATIVE_GLOBAL_VERSION=1" make
```

### Miscellaneous commands

```sh
make clean
```

### Clangd language server support

For [clangd](https://open-vsx.org/extension/llvm-vs-code-extensions/vscode-clangd) support in VSCode-based editors you can build the compilation database.

Include the following argument in your build command (each time you build):

```sh
EXTRA_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" make
```

Then in the PDXearch root directory, symlink the generated compilation database (once).

```sh
ln -s ./build/release/compile_commands.json ./
```

You might have to manually set the clangd executable path in the VSCode clangd extension configuration.

## Test

After building the extension you can run the tests. See
[test/README.md](test/README.md) for more details.

```sh
make test
```

```sh
make test_debug
```

## Code quality

```sh
make format
```

```sh
make tidy-check
```

## Example Setup

Here we declare a system setup that has successfully built and used the extension.

- Apple M4 Pro
- macOS Sequoia 15.6.1
- [LLVM 18](https://formulae.brew.sh/formula/llvm@18#default)

Build:

```sh
GEN=ninja DISABLE_SANITIZER=1 CC=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang CXX=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang++ EXTRA_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" make
```

Portable `make format` alternative:

```sh
uv run --with black --with clang_format==11.0.1 --with cmake-format duckdb/scripts/format.py --all --fix --noconfirm --directories src test
```

`make tidy-check`:

```sh
make tidy-check TIDY_BINARY=/opt/homebrew/opt/llvm@18/bin/clang-tidy
```
