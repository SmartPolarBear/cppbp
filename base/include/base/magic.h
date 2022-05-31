//
// Created by cleve on 5/31/2022.
//
#pragma once

#include <concepts>
#include <cstdint>
#include <fstream>

namespace cppbp::base
{
template<typename T>
concept UIntegerType=std::same_as<T, uint8_t> || std::same_as<T, uint16_t> || std::same_as<T, uint32_t> ||
                     std::same_as<T, uint64_t>;


template<UIntegerType T>
class IMagic
{
public:
    virtual T magic() const = 0;
};

template<base::UIntegerType T>
[[maybe_unused]] bool check_magic(const IMagic<T> &mgc, std::istream &stm)
{
    T magic{0};
    stm >> magic;
    return magic == mgc.magic();
}

}
