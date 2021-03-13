cmake_minimum_required(VERSION 3.12)
project(tensor LANGUAGES CXX VERSION 0.2.0)
set(CMAKE_CXX_STANDARD 17)

# Layout build dir like install dir so it's easier to find binaries and libs (bin/ dir)
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# TODO: in case of error look at this
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra)

# Force default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug, Release (default), RelWithDebInfo and MinSizeRel."
            FORCE)
endif (NOT CMAKE_BUILD_TYPE)

option(TENSOR_USE_BLAS "" OFF)
option(TENSOR_ENABLE_COVERAGE "" OFF)
option(TENSOR_BUILD_EXAMPLES "" OFF)
option(TENSOR_BUILD_TESTS "" OFF)
option(TENSOR_BUILD_SHARED_LIBS "" ON)
option(TENSOR_BUILD_PYTHON_WRAPPER "" OFF)
option(TENSOR_BUILD_BENCHMARK "" OFF)

option(TENSOR_USE_OPENMP "" OFF)

message(STATUS "USE_BLAS: ${TENSOR_USE_BLAS}")
message(STATUS "USE_OPENMP: ${TENSOR_USE_OPENMP}")
message(STATUS "ENABLE_COVERAGE: ${TENSOR_ENABLE_COVERAGE}")
message(STATUS "BUILD_EXAMPLES: ${TENSOR_BUILD_EXAMPLES}")
message(STATUS "BUILD_TESTS: ${TENSOR_BUILD_TESTS}")
message(STATUS "BUILD_SHARED_LIBS: ${TENSOR_BUILD_SHARED_LIBS}")
message(STATUS "BUILD_PYTHON_WRAPPER: ${TENSOR_BUILD_PYTHON_WRAPPER}")

if (TENSOR_BUILD_TESTS)
    add_subdirectory(third-party/Catch2)
endif ()

if (TENSOR_BUILD_PYTHON_WRAPPER)
    add_subdirectory(third-party/pybind11)
endif()

if (TENSOR_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

add_subdirectory(cpp)

if (TENSOR_USE_BLAS)
    if(TENSOR_BUILD_BENCHMARK)
        add_compile_definitions(BUILD_BENCHMARK)
        add_subdirectory(benchmark)
    endif()
endif ()

if (TENSOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
