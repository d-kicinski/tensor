
add_executable(char-rnn src/main.cpp)
target_link_libraries(char-rnn tensor nn)
set_target_properties(char-rnn PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET char-rnn PROPERTY CXX_STANDARD 20)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
