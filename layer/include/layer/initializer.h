//
// Created by cleve on 5/25/2022.
//

#pragma once

#include <base/base.h>

#include <concepts>
#include <memory>

namespace cppbp::layer
{
class IWeightInitializer
{
public:
    virtual base::MatrixType initialize_weights(size_t rows, size_t cols, size_t ni, size_t no) = 0;

    template<std::derived_from<IWeightInitializer> T, typename...TArgs>
    static inline std::shared_ptr<IWeightInitializer> make(TArgs &&... args)
    {
        return std::make_shared<T>(std::forward<TArgs>(args)...);
    }
};
}// namespace cppbp::layer
