//
// Created by cleve on 5/26/2022.
//

#pragma once

#include <base/serializable.h>

#include <layer/layer.h>

#include <model/persist.h>

#include <utils/counter.h>

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class BatchNorm
        : public ILayer,
          public utils::Counter<BatchNorm>
{

};
}// namespace cppbp::layer