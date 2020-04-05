cmake_minimum_required(VERSION 3.15)
project(algorithms)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/Catch2)

add_library(algorithms STATIC src/matrix.hpp src/tensor1d.hpp)
set_target_properties(algorithms PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(algorithms
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

add_executable(tests tests/matrix.cpp tests/tensor1d.cpp tests/main_catch2.cpp)
target_link_libraries(tests algorithms Catch2::Catch2)

include(lib/Catch2/contrib/Catch.cmake)
catch_discover_tests(tests)
