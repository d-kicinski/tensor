add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark tensor nn)
set_target_properties(benchmark PROPERTIES LINKER_LANGUAGE CXX)
