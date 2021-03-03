set(COMPILE_OPTIONS -fno-exceptions -fno-rtti)

set(SOURCES
        src/tensor/ts.hpp
        src/tensor/tensor.hpp
        src/tensor/tensor.cpp
        src/tensor/ops.hpp
        src/tensor/ops_common.hpp
        src/tensor/ops_common.cpp
        )

if (TENSOR_USE_BLAS)
    set(SOURCES ${SOURCES}  src/tensor/ops_dot_blas.cpp)
else ()
    set(SOURCES ${SOURCES}  src/tensor/ops_dot_naive.cpp)
endif()

add_library(tensor ${SOURCES})
add_library(tensor::tensor ALIAS tensor)
set_target_properties(tensor PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(tensor PRIVATE ${COMPILE_OPTIONS})
target_include_directories(tensor
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

if (TENSOR_USE_BLAS)
    message("Using BLAS for matrix operations.")
    set(BLA_STATIC ON)
    find_package(BLAS REQUIRED)
    target_compile_definitions(tensor PUBLIC USE_BLAS)
    target_link_libraries(tensor ${BLAS_LIBRARIES})
endif ()


add_library(nn
        src/tensor/nn/feed_forward.cpp
        src/tensor/nn/cross_entropy_loss.cpp
        src/tensor/nn/regularization.hpp
        src/tensor/nn/softmax.hpp
        src/tensor/nn/planar_dataset.cpp
        src/tensor/nn/dataset_iterator.cpp
        src/tensor/nn/conv2d.cpp
        src/tensor/nn/conv2d.hpp
        src/tensor/nn/functional.hpp
        src/tensor/nn/functional.cpp
        src/tensor/nn/functional_helpers.hpp
        src/tensor/nn/functional_helpers.cpp
        src/tensor/nn/softmax.cpp
        src/tensor/nn/activations.cpp
        src/tensor/nn/activations.hpp
        src/tensor/nn/optimizers.cpp
        src/tensor/nn/optimizers.hpp
        )
add_library(tensor::nn ALIAS nn)
target_link_libraries(nn tensor)
set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(nn PRIVATE ${COMPILE_OPTIONS})
target_include_directories(nn
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

# Enable tests only if project is not imported via add_subdirectory()
if (TENSOR_BUILD_TESTS)
    add_executable(tests
            tests/main_catch2.cpp
            tests/tensor/test_tensor.cpp
            tests/tensor/test_ops_common.cpp
            tests/tensor/test_ops_dot.cpp
            tests/tensor/nn/test_feed_forward.cpp
            tests/tensor/nn/test_cross_entropy_loss.cpp
            tests/tensor/nn/test_softmax.cpp
            tests/tensor/nn/test_conv2d.cpp
            tests/tensor/nn/test_planar_dataset.cpp
            tests/tensor/nn/test_functional.cpp
            tests/tensor/nn/test_functional_helpers.cpp
            tests/tensor/nn/test_activation.cpp
            tests/tensor/nn/test_optimizers.cpp
            )

    target_link_libraries(tests tensor nn Catch2::Catch2)

    include(${CMAKE_SOURCE_DIR}/third-party/Catch2/contrib/Catch.cmake)
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


if (TENSOR_BUILD_PYTHON_WRAPPER)
    set(PYTHON_LIB libtensor)
    add_library(${PYTHON_LIB} MODULE
            src/wrapper/tensor.cpp)

    pybind11_extension(${PYTHON_LIB})
    pybind11_strip(${PYTHON_LIB})

    set_target_properties(${PYTHON_LIB} PROPERTIES
            CXX_VISIBILITY_PRESET "hidden"
            CUDA_VISIBILITY_PRESET "hidden")
    target_link_libraries(${PYTHON_LIB} PRIVATE
            pybind11::module
            pybind11::lto
            tensor
            nn
            )
endif()
