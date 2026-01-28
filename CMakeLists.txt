cmake_minimum_required(VERSION 3.12)

# Set extension name here
set(TARGET_NAME pdxearch)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

include_directories(src/include)
set(EXTENSION_SOURCES src/pdxearch_extension.cpp)
add_subdirectory(src/index)

# Add `EXT_FLAGS="-DPDX_USE_ALTERNATIVE_GLOBAL_VERSION=1"` to an extension build command to enable.
if(PDX_USE_ALTERNATIVE_GLOBAL_VERSION)
  add_compile_definitions(PDX_USE_ALTERNATIVE_GLOBAL_VERSION)
endif()

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(BUILD_TESTING OFF)
add_subdirectory(third_party/faiss)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenMP::OpenMP_CXX)

add_definitions(-DEIGEN_USE_BLAS)
target_link_libraries(${EXTENSION_NAME} Eigen3::Eigen)
target_link_libraries(${LOADABLE_EXTENSION_NAME} Eigen3::Eigen)

target_link_libraries(${EXTENSION_NAME} faiss)
target_link_libraries(${LOADABLE_EXTENSION_NAME} faiss)

install(
  TARGETS ${EXTENSION_NAME} faiss
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
