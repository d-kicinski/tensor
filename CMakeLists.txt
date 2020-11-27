cmake_minimum_required(VERSION 3.12)
project(tensor LANGUAGES CXX VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Layout build dir like install dir so it's easier to find binaries and libs (bin/ dir)
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Force default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug, Release (default), RelWithDebInfo and MinSizeRel."
            FORCE)
endif (NOT CMAKE_BUILD_TYPE)

option(TENSOR_USE_BLAS "" OFF)
option(TENSOR_ENABLE_COVERAGE "" OFF)
option(TENSOR_BUILD_EXAMPLES "" ON)
option(TENSOR_BUILD_TESTS "" ON)
option(TENSOR_BUILD_SHARED_LIBS "" ON)

message(STATUS "USE_BLAS: ${TENSOR_USE_BLAS}")
message(STATUS "ENABLE_COVERAGE: ${TENSOR_ENABLE_COVERAGE}")
message(STATUS "BUILD_EXAMPLES: ${TENSOR_BUILD_EXAMPLES}")
message(STATUS "BUILD_TESTS: ${TENSOR_BUILD_TESTS}")
message(STATUS "BUILD_SHARED_LIBS: ${TENSOR_BUILD_SHARED_LIBS}")

if (TENSOR_BUILD_TESTS)
    add_subdirectory(lib/Catch2)
endif ()

if (TENSOR_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

set(COMPILE_OPTIONS -fno-exceptions -fno-rtti)

add_library(tensor
        src/tensor/ts.hpp
        src/tensor/tensor.hpp
        src/tensor/tensor.cpp
        src/tensor/ops_dot.cpp
        src/tensor/ops_common.hpp
        src/tensor/ops_common.cpp
        )
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(tensor PRIVATE ${COMPILE_OPTIONS})
target_include_directories(tensor
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

if (TENSOR_USE_BLAS)
    message("Using BLAS for matrix operations.")
    find_package(BLAS REQUIRED)
    target_compile_definitions(tensor PUBLIC USE_BLAS)
    target_link_libraries(tensor ${BLAS_LIBRARIES})
endif ()


add_library(nn
        src/tensor/nn/feed_forward.cpp
        src/tensor/nn/cross_entropy_loss.cpp
        src/tensor/nn/softmax.hpp
        src/tensor/nn/planar_dataset.cpp
        src/tensor/nn/dataset_iterator.cpp
        )
target_link_libraries(nn tensor)
set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(nn PRIVATE ${COMPILE_OPTIONS})
target_include_directories(nn
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

# Enable tests only if project is not imported via add_subdirectory()
if (TENSOR_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(tests
            tests/main_catch2.cpp
            tests/tensor/test_tensor.cpp
            tests/tensor/test_ops_common.cpp
            tests/tensor/test_ops_dot.cpp
            tests/nn/test_feed_forward.cpp
            tests/nn/test_cross_entropy_loss.cpp
            tests/nn/test_softmax.cpp
            tests/nn/test_planar_dataset.cpp)

    target_link_libraries(tests tensor nn Catch2::Catch2)

    include(lib/Catch2/contrib/Catch.cmake)
    enable_testing()
    catch_discover_tests(tests)

    if (TENSOR_ENABLE_COVERAGE)
        set(ENABLE_COVERAGE ON CACHE BOOL "Enable coverage build." FORCE)
        find_package(codecov)
        add_coverage(tests)
        list(APPEND LCOV_REMOVE_PATTERNS "'/usr/*'")
        coverage_evaluate()
    endif ()

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/resources DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif ()


if (TENSOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()