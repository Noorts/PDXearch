#pragma once

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

struct PDXearchModule {
public:
	static void Register(ExtensionLoader &loader) {

		auto &db = loader.GetDatabaseInstance();

		RegisterIndex(db);
		RegisterIndexScan(loader);

		RegisterScanOptimizer(db);

		RegisterIndexInfo(loader);
		RegisterBlobFunctions(loader);
	}

private:
	static void RegisterIndex(DatabaseInstance &ldb);
	static void RegisterIndexScan(ExtensionLoader &loader);

	static void RegisterScanOptimizer(DatabaseInstance &db);

	static void RegisterIndexInfo(ExtensionLoader &loader);
	static void RegisterBlobFunctions(ExtensionLoader &loader);
};

} // namespace duckdb
