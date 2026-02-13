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

# Remove OpenMP dependency for CI clang-tidy job. This workaround can be removed if we remove our Eigen dependency.
# Defining EIGEN_DONT_PARALLELIZE prevents Eigen from including the <omp.h>, which the clang-tidy CI is not properly set
# up to use. The CI's build and test job will ensure the extension remains buildable across platforms and architectures.
if(CLANG_TIDY)
  add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# SuperKMeans is a header-only library.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/SuperKMeans/include)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenMP::OpenMP_CXX)

# Based on SuperKMeans CMakeLists:
# https://github.com/lkuffo/SuperKMeans/blob/8d64661520845d587f52283962bf482233788f4c/CMakeLists.txt#L31-L55
set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG QUIET)
if (MKL_FOUND)
  message(STATUS "MKL library found")
  message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
  get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "MKL includes: ${mkl_includes}")
  add_definitions(-DEIGEN_USE_MKL_ALL)
  set(MKL_COMMON_LIBS MKL::MKL OpenMP::OpenMP_CXX m dl)
  set(BLAS_LINK_LIBRARIES "")
else()
  set(MKL_COMMON_LIBS "")
  message(STATUS "MKL not found. Trying to find a BLAS implementation")

  # On macOS, prefer Apple's Accelerate framework over other BLAS implementations
  if(APPLE)
      set(BLA_VENDOR Apple)
      message(STATUS "macOS detected: prioritizing Apple Accelerate framework")
  endif()

  find_package(BLAS REQUIRED)
  message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
  add_definitions(-DEIGEN_USE_BLAS)
  set(BLAS_LINK_LIBRARIES ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

if(MKL_FOUND)
  target_link_libraries(${EXTENSION_NAME} Eigen3::Eigen ${MKL_COMMON_LIBS})
  target_link_libraries(${LOADABLE_EXTENSION_NAME} Eigen3::Eigen ${MKL_COMMON_LIBS})
else()
  target_link_libraries(${EXTENSION_NAME} Eigen3::Eigen ${BLAS_LINK_LIBRARIES})
  target_link_libraries(${LOADABLE_EXTENSION_NAME} Eigen3::Eigen ${BLAS_LINK_LIBRARIES})
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
