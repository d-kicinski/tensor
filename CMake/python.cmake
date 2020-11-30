# Use latest UseSWIG module
cmake_minimum_required(VERSION 3.14)

set(PYTHON_PROJECT ${CMAKE_SOURCE_DIR}/pytensor)
set(SWIG_SRC ${CMAKE_SOURCE_DIR}/swig/tensor.i)

# Find SWIG
set(CMAKE_SWIG_FLAGS)
list(APPEND CMAKE_SWIG_FLAGS "-DSWIGWORDSIZE64")
find_package(SWIG REQUIRED)
include(UseSWIG)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if (Python_VERSION VERSION_GREATER_EQUAL 3)
    list(APPEND CMAKE_SWIG_FLAGS "-py3;-DPY3")
endif ()


# Swig wrap all libraries
set_property(SOURCE ${SWIG_SRC} PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${SWIG_SRC} PROPERTY SWIG_MODULE_NAME pytensor)
swig_add_library(pytensor
        LANGUAGE python
        OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        SOURCES ${SWIG_SRC})
set_property(TARGET pytensor PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
target_include_directories(pytensor
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include_swig
        ${Python_INCLUDE_DIRS}
        )
target_link_libraries(pytensor PRIVATE tensor ${PYTHON_LIBRARIES})

file(COPY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/_pytensor.so DESTINATION ${PYTHON_PROJECT})
file(COPY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pytensor.py DESTINATION ${PYTHON_PROJECT})

