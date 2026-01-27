cmake_minimum_required(VERSION 3.12)

# Set extension name here
set(TARGET_NAME pdxearch)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(Eigen3 REQUIRED)

add_definitions(-DEIGEN_USE_BLAS)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(third_party/faiss)

# Add `EXT_FLAGS="-DPDX_USE_ALTERNATIVE_GLOBAL_VERSION=1"` to an extension build command to enable.
if(PDX_USE_ALTERNATIVE_GLOBAL_VERSION)
  add_compile_definitions(PDX_USE_ALTERNATIVE_GLOBAL_VERSION)
endif()

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(BUILD_TESTING OFF)

if(APPLE)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES omp)
  set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
  set(OpenMP_C_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
  set(OpenMP_C_LIBRARY_DIR "/opt/homebrew/opt/libomp/lib")

  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES omp)
  set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
  set(OpenMP_CXX_LIBRARY_DIR "/opt/homebrew/opt/libomp/lib")

  link_directories("/opt/homebrew/opt/libomp/lib")
endif()

set(EXTENSION_SOURCES src/pdxearch_extension.cpp)
add_subdirectory(src/index)
add_subdirectory(third_party/faiss)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} Eigen3::Eigen)
target_link_libraries(${LOADABLE_EXTENSION_NAME} Eigen3::Eigen)

target_link_libraries(${EXTENSION_NAME} faiss)
target_link_libraries(${LOADABLE_EXTENSION_NAME} faiss)

find_package(OpenMP REQUIRED)
if(APPLE)
  target_link_libraries(${EXTENSION_NAME} "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  target_link_libraries(${LOADABLE_EXTENSION_NAME} "/opt/homebrew/opt/libomp/lib/libomp.dylib")
else()
  target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
  target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenMP::OpenMP_CXX)
endif()

install(
  TARGETS ${EXTENSION_NAME} faiss
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
