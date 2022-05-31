//
// Created by cleve on 5/11/2022.
//

#include <layer/dropout.h>
#include <layer/fully_connected.h>
#include <layer/layer_norm.h>
#include <layer/input.h>
#include <layer/relu.h>
#include <layer/Softmax.h>

#include <model/model.h>
#include <model/loss_output_callback.h>
#include <model/accuracy_callback.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>
#include <optimizer/sgd_optimizer.h>

#include <dataloader/dataloader.h>
#include <dataloader/mnist_dataset.h>

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
//    cout << fmt::format("Ground Truth:{}, Predict:{}", a, b) << endl;
}

int main()
{
    Sigmoid sigmoid{};
    Relu relu{};
    Softmax softmax{};

    Input in{784};
    FullyConnected fc1{800, relu};
    FullyConnected fc2{700, relu};
    FullyConnected fc3{400, relu};
    FullyConnected fc4{80, sigmoid};
    FullyConnected out{10, softmax};

    CrossEntropyLoss loss{};
    Model model{in | fc1 | fc2 | fc3 | fc4 | out, loss};

    MNISTDataset mnist{"data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte", true};
    MNISTDataset mnist_test{"data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte", true};

    DataLoader dl{mnist, 16, true, 0.0005};

    SGDOptimizer optimizer{0.1};

    auto loss_output_callback = IModelCallback::make<LossOutputCallback>();
    auto accuracy_callback = IModelCallback::make<AccuracyCallback>();
    auto top_3_accuracy_callback = IModelCallback::make<AccuracyCallback>(vector<int>{1, 3});

    vector<shared_ptr<IModelCallback>> callbacks{loss_output_callback, accuracy_callback};
    model.fit(dl, 1000, optimizer, true, 100, callbacks);

    DataLoader eval_dl{mnist_test, 16, true};
    vector<shared_ptr<IModelCallback>> eval_callbacks{loss_output_callback, top_3_accuracy_callback};
    model.evaluate(eval_dl, true, eval_callbacks);

    auto result = model.predict(mnist_test);

    for (int i = 0; i < mnist_test.size(); i++)
    {
        auto [data, label] = mnist_test.get(i);
        print_result(label, result[i]);
    }

    cout << acc << endl;
    return 0;
}