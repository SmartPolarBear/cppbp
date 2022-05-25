//
// Created by cleve on 5/25/2022.
//

#include <layer/he_initializer.h>

using namespace cppbp::layer;
using namespace cppbp::base;

MatrixType HeInitializer::initialize_weights(size_t w, size_t h, size_t ni, size_t no)
{
    std::random_device r;
    std::mt19937 gen{r()};
    std::normal_distribution<double> dist{0.0, std::sqrt(2.0 / no)};
    auto normal_sample = [&](int)
    {
        return dist(gen);
    };
    return cppbp::base::MatrixType::NullaryExpr(w, h, normal_sample);
}