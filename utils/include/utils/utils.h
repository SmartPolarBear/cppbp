//
// Created by cleve on 5/30/2022.
//

#pragma once

#include <vector>
#include <algorithm>

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
}