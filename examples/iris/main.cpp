//
// Created by cleve on 5/11/2022.
//

#include <layer/dropout.h>
#include <layer/fully_connected.h>
#include <layer/input.h>
#include <layer/relu.h>
#include <layer/softmax.h>

#include <model/model.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>
#include <optimizer/sgd_optimizer.h>

#include <dataloader/dataloader.h>
#include <dataloader/iris_dataset.h>

#include <fmt/format.h>

#include "optimizer/cross_entropy.h"
#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::dataloader;

using namespace std;


int argmax(const Eigen::VectorXd &vals)
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

int acc = 0;

void print_result(Eigen::VectorXd &label, Eigen::VectorXd &ret)
{
    auto a = argmax(label);
    auto b = argmax(ret);
    if (a == b)
    {
        acc++;
    }
    cout << fmt::format("Ground Truth:{}, Predict:{}", a, b) << endl;
}

int main()
{
    Sigmoid sigmoid{};
    Relu relu{};
    softmax softmax{};

    Input in{4};
    FullyConnected fc1{5, relu};
    FullyConnected fc2{8, sigmoid};
    DropOut drop1{0.05};
    FullyConnected fc3{12, sigmoid};
    FullyConnected out{3, softmax};

    CrossEntropyLoss loss{};
    Model model{in | fc1 | fc2 | drop1 | fc3 | out, loss};

    std::cout << model.summary() << endl;

    IrisDataset iris{"data/iris.data", true};
    DataLoader dl{iris, 16, true};

    SGDOptimizer optimizer{0.1};
    model.fit(dl, 2000, optimizer, true, 100);

    for (int i = 0; i < iris.size(); i++)
    {
        auto [data, label] = iris.get(i);
        auto ret = model(data);
        print_result(label, ret);

    }
    cout << acc << endl;
    return 0;
}