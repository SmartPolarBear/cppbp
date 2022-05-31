//
// Created by cleve on 5/31/2022.
//

#pragma once

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <fstream>

namespace cppbp::base
{
class not_implemented
        : public std::exception
{
public:
    not_implemented()
            : std::exception("Not implemented")
    {
    }
};
}