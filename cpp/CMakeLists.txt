set(COMPILE_OPTIONS -fno-exceptions -fno-rtti)

set(SOURCES
        src/tensor/ts.hpp
        src/tensor/tensor.hpp
        src/tensor/tensor.cpp
        src/tensor/ops.hpp
        src/tensor/ops_common.hpp
        src/tensor/ops_common.cpp
        src/tensor/statistics.hpp
        )

if (TENSOR_USE_BLAS)
    set(SOURCES ${SOURCES} src/tensor/ops_dot_blas.cpp)
    if (TENSOR_BUILD_BENCHMARK)
        set(SOURCES ${SOURCES} src/tensor/ops_dot_naive.cpp)
    endif ()
else ()
    set(SOURCES ${SOURCES} src/tensor/ops_dot_naive.cpp)
endif ()

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
    set(BLA_STATIC ON)
    find_package(BLAS REQUIRED)
    target_compile_definitions(tensor PUBLIC USE_BLAS)
    target_link_libraries(tensor ${BLAS_LIBRARIES})
endif ()

if (TENSOR_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tensor OpenMP::OpenMP_CXX)
endif ()


set(NN_SOURCES
        src/tensor/nn/cross_entropy_loss.cpp
        src/tensor/nn/regularization.hpp
        src/tensor/nn/softmax.hpp
        src/tensor/nn/planar_dataset.cpp
        src/tensor/nn/dataset_iterator.cpp
        src/tensor/nn/image_utils.hpp
        src/tensor/nn/image_utils.cpp
        src/tensor/nn/conv_2d_helpers.hpp
        src/tensor/nn/conv_2d_helpers.cpp
        src/tensor/nn/softmax.cpp
        src/tensor/nn/activations.cpp
        src/tensor/nn/activations.hpp
        src/tensor/nn/optimizers.cpp
        src/tensor/nn/optimizers.hpp
        src/tensor/nn/variable.cpp
        src/tensor/nn/variable.hpp
        src/tensor/nn/initialization.hpp
        src/tensor/nn/im2col.cpp
        src/tensor/nn/im2col.hpp
        src/tensor/nn/conv_2d.cpp
        src/tensor/nn/max_pool_2d.cpp

        src/tensor/nn/layer/feed_forward.cpp
        src/tensor/nn/layer/max_pool_2d.cpp
        src/tensor/nn/layer/layer_base.cpp
        src/tensor/nn/layer/conv_2d_naive.cpp
        src/tensor/nn/layer/conv_2d_im2col.cpp
        )

if (TENSOR_USE_PROTOBUF)
    set(NN_PROTO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/nn/proto)
    add_custom_command(
            OUTPUT
            ${NN_PROTO_PATH}/array.pb.h
            ${NN_PROTO_PATH}/array.pb.cc
            COMMAND
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/protoc array.proto
            --proto_path ${NN_PROTO_PATH}
            --cpp_out ${NN_PROTO_PATH}
            DEPENDS
            protobuf::protoc
    )
    add_custom_target(proto
            DEPENDS ${NN_PROTO_PATH}/array.pb.h ${NN_PROTO_PATH}/array.pb.cc
            )

    list(APPEND NN_SOURCES
            src/tensor/nn/saver.cpp
            src/tensor/nn/saver.hpp
            ${NN_PROTO_PATH}/array.pb.cc
            ${NN_PROTO_PATH}/array.pb.h
            )
    set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
endif ()


add_library(nn ${NN_SOURCES})
add_library(tensor::nn ALIAS nn)
target_link_libraries(nn tensor)
set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(nn PRIVATE ${COMPILE_OPTIONS})
target_include_directories(nn
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

if (TENSOR_USE_PROTOBUF)
    add_dependencies(nn proto)
    target_link_libraries(nn protobuf::libprotobuf)
endif ()

# Enable tests only if project is not imported via add_subdirectory()
if (TENSOR_BUILD_TESTS)
    set(NN_TEST_SOURCES
            tests/main_catch2.cpp
            tests/tensor/test_tensor.cpp
            tests/tensor/test_ops_common.cpp
            tests/tensor/test_ops_dot.cpp
            tests/tensor/nn/test_cross_entropy_loss.cpp
            tests/tensor/nn/test_softmax.cpp
            tests/tensor/nn/test_planar_dataset.cpp
            tests/tensor/nn/test_image_utils.cpp
            tests/tensor/nn/test_conv_2d_helpers.cpp
            tests/tensor/nn/test_activation.cpp
            tests/tensor/nn/test_optimizers.cpp
            tests/tensor/nn/test_variable.cpp
            tests/tensor/nn/test_im2col.cpp
            tests/tensor/nn/test_max_pool_2d.cpp
            tests/tensor/nn/test_conv_2d.cpp

            tests/tensor/nn/layer/test_feed_forward.cpp
            tests/tensor/nn/layer/test_conv_2d.cpp
            )

    if (TENSOR_USE_PROTOBUF)
        set(NN_TEST_SOURCES ${NN_TEST_SOURCES} tests/tensor/nn/test_saver.cpp tests/tensor/nn/test_max_pool_2d.cpp tests/tensor/nn/test_conv_2d.cpp)
    endif ()

    add_executable(tests ${NN_TEST_SOURCES})

    target_link_libraries(tests tensor nn Catch2::Catch2)

    include(${CMAKE_SOURCE_DIR}/third-party/Catch2/contrib/Catch.cmake)
    enable_testing()
    catch_discover_tests(tests)

    if (TENSOR_ENABLE_COVERAGE)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third-party/Catch2/CMake")
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
endif ()
