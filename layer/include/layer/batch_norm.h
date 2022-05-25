//
// Created by cleve on 5/24/2022.
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
 public:
	BatchNorm() = default;

	explicit BatchNorm(size_t len, double drop_prob = 0.1);
};
}// namespace cppbp::layer