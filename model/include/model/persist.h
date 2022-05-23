//
// Created by cleve on 5/23/2022.
//
#pragma once

#include <cstdint>

namespace cppbp::model::persist
{

static inline constexpr auto HEADER_MAGIC = "CPPBP000";

template<typename>
struct LossFunctionTypeId;

template<typename>
struct ActivationFunctionTypeId;

template<typename>
struct LayerTypeId;


#pragma(pack(push, 1))
struct ModelHeader
{
	char magic[8]; //CPPBP000
	uint32_t layer_nums;
	uint32_t loss_func;
	uint64_t checksum;
};
#pragma(pack(pop))

#pragma(pack(push, 1))
struct LayerDescriptor
{
	uint32_t type;
	uint32_t act_func;
	union
	{
		uint64_t neurons;
		uint64_t inputs;
	};
};
#pragma(pack(pop))

}