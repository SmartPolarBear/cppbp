//
// Created by cleve on 5/25/2022.
//
#include <layer/xavier_initializer.h>

#include <random>

using namespace std;

using namespace cppbp::layer;
using namespace cppbp::base;

cppbp::base::MatrixType cppbp::layer::XavierInitializer::initialize_weights(size_t w, size_t h, size_t ni, size_t no)
{
    const double limit = std::sqrt(6.0) / std::sqrt(ni + no);

    std::random_device r;
    std::mt19937 gen{r()};
    std::uniform_real_distribution<double> dist{-limit, limit};
    auto uniform_sample = [&](int)
    {
        return dist(gen);
    };
    return cppbp::base::MatrixType::NullaryExpr(w, h, uniform_sample);
}
