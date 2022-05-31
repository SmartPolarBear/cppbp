//
// Created by cleve on 5/30/2022.
//

#pragma once

#include <base/magic.h>

#include <vector>
#include <algorithm>
#include <span>
#include <numeric>

namespace cppbp::utils
{
template<typename T>
int argmax(const T &vals)
{
    int ret = 0;
    for (int i = 1; i < vals.size(); i++)
    {
        if (vals[i] > vals[ret])
        {
            ret = i;
        }
    }
    return ret;
}

template<typename T>
std::vector<int> argsort(const T &vals)
{
    std::vector<int> ret(vals.size());
    std::iota(ret.begin(), ret.end(), 0);
    std::sort(ret.begin(), ret.end(), [&vals](int a, int b)
    {
        return vals[a] > vals[b];
    });
    return ret;
}

template<base::UIntegerType T>
[[maybe_unused]] constexpr T magic_from_string(std::string_view s)
{
    T ret = *reinterpret_cast<const T *>(s.data());
    return ret;
}

template<typename T>
void endian_swap(T &obj)
{
    uint8_t *memp = reinterpret_cast<uint8_t *>(&obj);
    std::reverse(memp, memp + sizeof(T));
}

}