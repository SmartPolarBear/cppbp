//
// Created by cleve on 5/30/2022.
//

#include <gtest/gtest.h>

#include <layer/sigmoid.h>

using namespace cppbp::layer;

TEST(SigmoidTest, EvalTest)
{
    Sigmoid sigmoid{};
    ASSERT_EQ(sigmoid.eval(100000000), 1);
    ASSERT_EQ(sigmoid.eval(0), 0.5);
    ASSERT_EQ(sigmoid.eval(-100000000), 0);
}

TEST(SigmoidTest, DerivativeTest)
{
    Sigmoid sigmoid{};
    ASSERT_EQ(sigmoid.derive(sigmoid.eval(100000000)), 0);
    ASSERT_FLOAT_EQ(sigmoid.derive(sigmoid.eval(0)), 0.25);
    ASSERT_EQ(sigmoid.derive(sigmoid.eval(-100000000)), 0);
}