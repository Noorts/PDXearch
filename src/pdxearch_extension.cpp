#include "pdxearch_extension.hpp"

#include "index/pdxearch_module.hpp"

#ifdef PDXEARCH_USES_OPENBLAS
extern "C" void openblas_set_num_threads(int num_threads);
#endif

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
#ifdef PDXEARCH_USES_OPENBLAS
	// DuckDB parallelizes the searches and index builds, so each BLAS call runs on the thread that makes it.
	openblas_set_num_threads(1);
#endif
	PDXearchModule::Register(loader);
}

void PdxearchExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string PdxearchExtension::Name() {
	return "pdxearch";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(pdxearch, loader) {
	duckdb::LoadInternal(loader);
}
}
