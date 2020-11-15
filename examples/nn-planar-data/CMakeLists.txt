
add_executable(nn-planar-data src/main.cpp)
target_link_libraries(nn-planar-data tensor nn)
set_target_properties(nn-planar-data PROPERTIES LINKER_LANGUAGE CXX)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
