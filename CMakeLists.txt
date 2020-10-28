cmake_minimum_required(VERSION 3.15)
project(LinearAlgebra)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/Catch2)
add_subdirectory(lib/fmt)
add_subdirectory(lib/GSL)
add_subdirectory(lib/range-v3)

find_package(BLAS REQUIRED)

add_library(linalg STATIC
        src/exceptions.hpp
        src/flatarray/flatarray.hpp
        src/flatarray/iterator.hpp
        src/flatarray/dimensions.hpp
        src/flatarray/ops.hpp)
set_target_properties(linalg PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(linalg
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_link_libraries(linalg
        fmt::fmt
        Microsoft.GSL::GSL
        range-v3::range-v3
        ${BLAS_LIBRARIES})

add_executable(tests
        tests/main_catch2.cpp
        tests/flatarray/test_flatarray.cpp
        tests/flatarray/test_dimensions.cpp
        tests/flatarray/test_dot.cpp)
target_link_libraries(tests linalg Catch2::Catch2)

include(lib/Catch2/contrib/Catch.cmake)
catch_discover_tests(tests)
