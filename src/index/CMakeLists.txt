include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/pdxearch_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/create/pdxearch_index_create_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/create/pdxearch_index_create.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/create/pdxearch_index_global_create.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_filtered_scan_logical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_filtered_scan_physical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_global_filtered_scan_physical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_global_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_scan_logical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_index_scan_physical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search/pdxearch_scan_optimizer.cpp
    PARENT_SCOPE)
