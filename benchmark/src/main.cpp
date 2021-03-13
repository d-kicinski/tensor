#include <chrono>
#include <tensor/nn/conv2d.hpp>
#include <tensor/nn/feed_forward.hpp>
#include <tensor/tensor.hpp>

using namespace std::chrono;

auto create_matrices(int in, int middle, int out) -> std::pair<ts::MatrixF, ts::MatrixF>
{
    auto a = ts::MatrixF(in, middle);
    auto b = ts::MatrixF(middle, out);
    return std::make_pair(std::move(a), std::move(b));
}

auto benchmark(std::function<void(void)> const &fn, size_t const runs = 1000) -> uint
{
    std::vector<uint> results(runs);

    for (auto & r: results) {
        auto time_start = high_resolution_clock::now();
        fn();
        auto time_end = high_resolution_clock::now();
        r = duration_cast<microseconds>(time_end - time_start).count();
    }
    return std::reduce(results.begin(), results.end(), 0, std::plus()) / runs;
}

auto benchmark_matrix_multiply() -> void
{
    std::cout << "ts::dot with blas: " << std::endl;
    {
        auto [a, b] = create_matrices(64, 512, 10);
        auto time = benchmark([a=std::ref(a), b=std::ref(b)](){ ts::blas::dot(a, b);} );
        std::cout << "\t[64, 512]x[512, 10]: " << time << " us" << std::endl;
    }

    {
        auto [a, b] = create_matrices(128, 1024, 10);
        auto time = benchmark([a=std::ref(a), b=std::ref(b)](){ ts::blas::dot(a, b);} );
        std::cout << "\t[128, 1024]x[1024, 10]: " << time << " us" << std::endl;
    }

    std::cout << "ts::dot with naive: " << std::endl;
    {
        auto [a, b] = create_matrices(64, 512, 10);
        auto time = benchmark([a=std::ref(a), b=std::ref(b)](){ ts::naive::dot(a, b);} );
        std::cout << "\t[64, 512]x[512, 10]: " << time << " us" << std::endl;
    }

    {
        auto [a, b] = create_matrices(128, 1024, 10);
        auto time = benchmark([a=std::ref(a), b=std::ref(b)](){ ts::naive::dot(a, b);} );
        std::cout << "\t[128, 1024]x[1024, 10]: " << time << " us" << std::endl;
    }
}

auto benchmark_linear() -> void
{
    auto input = ts::MatrixF(64, 512);
    auto output = ts::MatrixF();
    {
        auto layer = ts::FeedForward::create(512, 10, ts::Activation::NONE);
        auto time_fw = benchmark([&](){ output = layer(input); }, 100);
        auto time_bw = benchmark([&](){ layer.backward(output); }, 100);
        std::cout << "Linear::forward " << time_fw << " us" << std::endl;
        std::cout << "Linear::backward " << time_bw << " us" << std::endl;
    }
    {
        auto layer = ts::FeedForward::create(512, 10, ts::Activation::RELU);
        auto time_fw = benchmark([&](){ output = layer(input); }, 100);
        auto time_bw = benchmark([&](){ layer.backward(output); }, 100);
        std::cout << "Linear[ReLU]::forward " << time_fw << " us" << std::endl;
        std::cout << "Linear[ReLU]::backward " << time_bw << " us" << std::endl;
    }
}

auto benchmark_conv2d() -> void
{
    auto input = ts::Tensor<float, 4>(16, 64, 64, 3);
    auto output = ts::Tensor<float, 4>();
    {
        auto layer = ts::Conv2D::create(3, 24, 3, 1,
                                        ts::Activation::NONE, false);
        auto time_fw = benchmark([&](){ output = layer(input); }, 100);
        auto time_bw = benchmark([&](){ layer.backward(output); }, 100);
        std::cout << "Conv2D::forward " << time_fw << " us" << std::endl;
        std::cout << "Conv2D::backward " << time_bw << " us" << std::endl;
    }
    {
        auto layer = ts::Conv2D::create(3, 24, 3, 1,
                                        ts::Activation::RELU, false);
        auto time_fw = benchmark([&](){ output = layer(input); }, 100);
        auto time_bw = benchmark([&](){ layer.backward(output); }, 100);
        std::cout << "Conv2D[ReLU]::forward " << time_fw << " us" << std::endl;
        std::cout << "Conv2D[ReLU]::backward " << time_bw << " us" << std::endl;
    }
}

auto main() -> int
{
    benchmark_matrix_multiply();
    benchmark_linear();
    benchmark_conv2d();
    return 0;
}