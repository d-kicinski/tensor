cmake_minimum_required(VERSION 3.12)
project(TensorOps LANGUAGES CXX)
set(LIBRARY_NAME tensor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

add_subdirectory(lib/Catch2)

option(TENSOR_USE_BLAS "" OFF)
option(TENSOR_ENABLE_COVERAGE "" OFF)

message(STATUS "USE_BLAS: ${TENSOR_USE_BLAS}")
message(STATUS "ENABLE_COVERAGE: ${TENSOR_ENABLE_COVERAGE}")

add_library(${LIBRARY_NAME} SHARED
        src/tensor/exceptions.hpp
        src/tensor/ts.hpp
        src/tensor/tensor.hpp
        src/tensor/tensor.cpp
        src/tensor/iterator.hpp
        src/tensor/dimensions.hpp
        src/tensor/ops_dot.cpp
        src/tensor/ops_common.hpp
        src/tensor/ops_common.cpp
        include/tensor/tensor.hpp
        include/tensor/ops.hpp)
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(tensor
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

if(TENSOR_USE_BLAS)
    message("Using BLAS for matrix operations.")
    find_package(BLAS REQUIRED)
    target_compile_definitions(tensor PUBLIC USE_BLAS)
    target_link_libraries(tensor ${BLAS_LIBRARIES})
endif()


add_library(nn SHARED
        src/nn/feedforward.cpp
        src/nn/feedforward.hpp)
target_link_libraries(nn tensor)

set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(nn
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

# Enable tests only if project is not imported via add_subdirectory()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(tests
            tests/main_catch2.cpp
            tests/tensor/test_tensor.cpp
            tests/tensor/test_dimensions.cpp
            tests/tensor/test_ops_common.cpp
            tests/tensor/test_ops_dot.cpp)

    target_link_libraries(tests tensor Catch2::Catch2)

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
endif()

