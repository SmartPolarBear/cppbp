//
// Created by cleve on 5/23/2022.
//

#pragma once

#include <memory>
#include <tuple>

namespace cppbp::base
{
class ISerializable
{
public:
    virtual std::ostream & serialize(std::ostream &out) = 0;

    virtual std::istream &deserialize(std::istream &input) = 0;
};
}