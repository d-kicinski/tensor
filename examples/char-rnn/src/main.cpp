#include <cmath>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <set>
#include <unordered_map>
#include <vector>

#include <tensor/nn/layer/rnn.hpp>
#include <tensor/nn/optimizer/adagrad.hpp>

class Vocabulary {
  public:
    Vocabulary() = default;

    explicit Vocabulary(std::u32string const &text)
    {
        auto vocab = _get_unique_characters(text);
        for (int i = 0; i < vocab.size(); ++i) {
            auto str = *std::next(vocab.begin(), i);
            _id_to_char[i] = str;
            _char_to_id[str] = i;
        }
    }

    auto to_text(std::vector<int> const &indices) -> std::u32string
    {
        std::u32string chars;
        for (int const &idx : indices) {
            chars += _id_to_char[idx];
        }
        return chars;
    }

    auto to_indices(std::u32string text) -> std::vector<int>
    {
        std::vector<int> indices(text.length());
        for (int i = 0; i < text.length(); ++i) {
            indices[i] = _char_to_id[text[i]];
        }
        return indices;
    }

    auto save(std::string const &path) -> void
    {
        if (std::ofstream output = std::ofstream(path)) {
            for (auto &[token, idx] : _char_to_id) {
                output << _converter.to_bytes(token) << "\n";
            }
        } else {
            std::cerr << "Couldn't open file: " << path << std::endl;
            std::exit(1);
        }
    }

    auto load(std::string const &path) -> void
    {
        if (std::ifstream input = std::ifstream(path)) {
            int line_num = 0;
            for (std::string line; std::getline(input, line); ++line_num) {
                char32_t c = _converter.from_bytes(line).at(0);
                _char_to_id[c] = line_num;
                _id_to_char[line_num] = c;
            }
        } else {
            std::cerr << "Couldn't open file: " << path << std::endl;
            std::exit(1);
        }
    }

    auto size() -> ulong { return _char_to_id.size(); }

  private:
    std::unordered_map<int, char32_t> _id_to_char;
    std::unordered_map<char32_t, int> _char_to_id;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> _converter{};

    static auto _get_unique_characters(std::u32string const &text) -> std::set<char32_t>
    {
        return std::set<char32_t>(text.begin(), text.end());
    }
};

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

    int hidden_size = 100;
    int sequence_length = 25;
    float learning_rate = 1e-1;
    ulong vocab_size = vocabulary.size();
    ts::RNN rnn(hidden_size, sequence_length, vocab_size);
    ts::Adagrad<float> optimizer(rnn.parameters(), learning_rate);

    int sample_length = sequence_length + 1;
    int chunk_num = std::floor(buffer.length() / sample_length);

    ts::MatrixF current_state(1, 128);

    for (int i = 0; i < chunk_num; ++i) {
        auto current_chunk = std::u32string(std::next(buffer.begin(), i * sample_length), std::next(buffer.begin(), (i + 1) * sample_length));
        auto all_indices = vocabulary.to_indices(current_chunk);
        auto indices = std::vector<int>(all_indices.begin(), all_indices.end() - 1);
        auto targets = std::vector<int>(all_indices.begin() + 1, all_indices.end());

        float loss = rnn.forward(indices, targets, current_state);
        rnn.backward();
        optimizer.step();
        current_state = rnn.state();
        optimizer.zero_gradients();

        if (i % 100 == 0) {
            std::cout << i << "/" << chunk_num << " loss: " << loss << std::endl;
        }
    }
    return 0;
}