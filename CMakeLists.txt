cmake_minimum_required(VERSION 3.15)
project(algorithms)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/Catch2)
add_subdirectory(lib/fmt)
add_subdirectory(lib/GSL)
add_subdirectory(lib/range-v3)

add_library(algorithms STATIC
        src/matrix.cpp
        src/matrixiterator.cpp
        src/exceptions.hpp
        src/matrixview.cpp src/matrixview.hpp
        src/flatarray.hpp
        )
set_target_properties(algorithms PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(algorithms
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_link_libraries(algorithms
        fmt::fmt
        Microsoft.GSL::GSL
        range-v3::range-v3
        )

add_executable(tests
        tests/test_matrix.cpp
        tests/main_catch2.cpp tests/test_plain_arrays.cpp tests/test_flatarray.cpp)
target_link_libraries(tests algorithms Catch2::Catch2)

include(lib/Catch2/contrib/Catch.cmake)
catch_discover_tests(tests)
