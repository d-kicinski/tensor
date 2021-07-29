
add_executable(char-rnn
        src/train_rnn.cpp
        src/vocabulary.hpp
        )
target_link_libraries(char-rnn tensor nn)
set_target_properties(char-rnn PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET char-rnn PROPERTY CXX_STANDARD 20)

add_executable(char-lstm
        src/train_lstm.cpp
        src/vocabulary.hpp
        )
target_link_libraries(char-lstm tensor nn)
set_target_properties(char-lstm PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET char-lstm PROPERTY CXX_STANDARD 20)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
