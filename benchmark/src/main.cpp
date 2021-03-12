#include <chrono>
#include <tensor/tensor.hpp>

using namespace std::chrono;

auto create_matrices(int in, int middle, int out) -> std::pair<ts::MatrixF, ts::MatrixF>
{
    auto a = ts::MatrixF(in, middle);
    auto b = ts::MatrixF(middle, out);
    return std::make_pair(std::move(a), std::move(b));
}

auto benchmark(std::function<void(void)> const &fn) -> uint
{
    constexpr size_t runs = 10000;
    std::array<uint, runs> results{};

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
    std::cout << "with blas: " << std::endl;
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

    std::cout << "with naive: " << std::endl;
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

auto main() -> int
{
    benchmark_matrix_multiply();
    return 0;
}