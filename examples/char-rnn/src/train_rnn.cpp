#include <cmath>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <set>
#include <vector>

#include <tensor/nn/layer/rnn.hpp>
#include <tensor/nn/optimizer/adagrad.hpp>

#include "vocabulary.hpp"

int main()
{
    auto path = "resources/pan-tadeusz-clean.txt";
    std::u32string buffer;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;

    if (std::ifstream input = std::ifstream(path)) {
        auto temp_buffer = std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        buffer = converter.from_bytes(temp_buffer);
    } else {
        std::cerr << "Couldn't open file: " << path << std::endl;
        std::exit(1);
    }

    auto vocabulary = Vocabulary(buffer);
    vocabulary.save("vocab.txt");

    int epoch_num = 10;
    int hidden_size = 100;
    int sequence_length = 25;
    float learning_rate = 1e-1;
    ulong vocab_size = vocabulary.size();

    ts::RNN rnn(hidden_size, sequence_length, vocab_size);
    ts::Adagrad<float> optimizer(rnn.parameters(), learning_rate);
    ts::MatrixF current_state(1, hidden_size);

    int sample_length = sequence_length + 1;
    int chunk_num = std::floor(buffer.length() / sample_length);

    float smooth_loss = -std::log(1.0 / vocab_size) * sequence_length;
    for (int i_epoch = 0; i_epoch < epoch_num; ++i_epoch) {
        for (int i = 0; i < chunk_num; ++i) {
            auto current_chunk = std::u32string(std::next(buffer.begin(), i * sample_length),
                                                std::next(buffer.begin(), (i + 1) * sample_length));
            auto all_indices = vocabulary.to_indices(current_chunk);
            auto indices = std::vector<int>(all_indices.begin(), all_indices.end() - 1);
            auto targets = std::vector<int>(all_indices.begin() + 1, all_indices.end());

            if (i % 2000 == 0) {
                auto sample_idx = rnn.sample(indices[0], current_state, 200);
                auto text = converter.to_bytes(vocabulary.to_text(sample_idx));

                std::cout << std::endl << "Generated text:" << std::endl;
                std::cout << text << std::endl << std::endl;
            }

            optimizer.zero_gradients();
            float loss = rnn.forward(indices, targets, current_state);
            rnn.backward();
            optimizer.step();
            current_state = rnn.state();

            smooth_loss = smooth_loss * 0.999 + loss * 0.001;

            if (i % 1000 == 0) {
                std::cout << "epoch [" << i_epoch << "/" << epoch_num << "]   steps: [" << i << "/" << chunk_num
                          << "]    loss: " << smooth_loss << std::endl;
            }
        }
    }
    return 0;
}
