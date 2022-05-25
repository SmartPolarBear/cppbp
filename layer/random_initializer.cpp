//
// Created by cleve on 5/25/2022.
//
#include <layer/random_initializer.h>

using namespace cppbp::base;

cppbp::base::MatrixType cppbp::layer::RandomInitializer::initialize_weights(size_t w, size_t h, size_t ni, size_t no)
{
    return MatrixType::Random(w, h);
}
