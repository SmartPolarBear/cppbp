//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/sigmoid.h>

#include <model/model.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>

#include <dataloader/iris_dataset.h>
#include <dataloader/dataloader.h>

#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::dataloader;

using namespace std;

int main()
{
	Sigmoid sigmoid{};

	FullyConnected fc1{ 4, sigmoid };
	FullyConnected fc2{ 6, sigmoid };
	FullyConnected fc3{ 3, sigmoid };

	MSELoss loss{};
	Model model{ fc1.connect(fc2).connect(fc3), loss };

	std::cout << model.summary() << endl;

	IrisDataset iris{ "data/iris.data" };
	DataLoader dl{ iris, 16, true };

	FixedStepOptimizer optimizer{ 0.5 };
	model.fit(dl, 5, optimizer);

	return 0;
}