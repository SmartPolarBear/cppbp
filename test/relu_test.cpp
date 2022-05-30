//
// Created by cleve on 5/30/2022.
//

#include <gtest/gtest.h>

#include <layer/relu.h>

using namespace cppbp::layer;

TEST(ReLUTest, EvalTest)
{
    Relu relu{};
    ASSERT_EQ(relu.eval(-1), 0);
    ASSERT_EQ(relu.eval(0), 0);
    ASSERT_EQ(relu.eval(1), 1);
}

TEST(ReLUTest, DerivativeTest)
{
    Relu relu{};
    ASSERT_EQ(relu.derive(-1), 0);
    ASSERT_EQ(relu.derive(0), 0);
    ASSERT_EQ(relu.derive(1), 1);
    ASSERT_EQ(relu.derive(2), 1);
}