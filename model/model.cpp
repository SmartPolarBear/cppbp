//
// Created by cleve on 5/11/2022.
//

#include <model/model.h>

#include <utility>
#include <iostream>

#include <fstream>

using namespace std;

using namespace Eigen;

cppbp::model::Model::Model(layer::ILayer& layer, optimizer::ILossFunction& loss)
	: input_(&layer), output_(&layer), loss_(&loss)
{
	while (input_->prev())
	{
		input_ = input_->prev();
	}

	while (output_->next())
	{
		output_ = output_->next();
	}
}

cppbp::model::Model::Model(std::vector<layer::ILayer>& layers, optimizer::ILossFunction& loss)
	: input_(&layers.front()), output_(&layers.back()), loss_(&loss)
{
	for (int i = 1; i < layers.size(); ++i)
	{
		layers[i - 1].connect(layers[i]);
	}
}

void cppbp::model::Model::set(std::vector<double> values)
{
	VectorXd vec(values.size());
	for (int i = 0; i < values.size(); i++)
	{
		vec[i] = values[i];
	}
	input_->set(vec);
}

Eigen::VectorXd cppbp::model::Model::operator()(std::vector<double> input)
{
	set(std::move(input));
	input_->forward();
	return output_->get();
}

Eigen::VectorXd cppbp::model::Model::operator()(Eigen::VectorXd input)
{
	input_->set(std::move(input));
	input_->forward();
	return output_->get();
}

void cppbp::model::Model::forward()
{
	input_->forward();
}

void cppbp::model::Model::backprop()
{
	output_->backprop();
}

std::string cppbp::model::Model::name() const
{
	return name_;
}

std::string cppbp::model::Model::summary() const
{
	return input_->summary();
}

void cppbp::model::Model::optimize(cppbp::optimizer::IOptimizer& opt)
{
	opt.step();
	input_->optimize(opt);
}

void cppbp::model::Model::fit(cppbp::dataloader::DataLoader& dl,
	size_t epoch,
	cppbp::optimizer::IOptimizer& opt,
	bool verbose)
{
	for (size_t e = 0; e < epoch; e++)
	{
		if (verbose)
		{
			std::cout << "Epoch: " << e << std::endl << "Train..." << std::endl;

		}
		auto batch = dl.train_batch();
		for (size_t step = 0; auto& [data, label] : batch)
		{
			auto predicts = (*this)(data);
			auto loss = (*loss_)(predicts, label);

			if (verbose)
			{
				std::cout << "Step:" << step << " Loss:" << loss << " ";
				//TODO
				std::cout << std::endl;
			}

			auto errors = loss_->derive(predicts, label);

			output_->set_errors(errors);
			this->backprop();
			this->optimize(opt);

			step++;
		}

		if (verbose)
		{
			std::cout << "Eval..." << std::endl;
		}
		batch = dl.eval_batch();
		for (size_t step = 0; auto& [data, label] : batch)
		{
			auto predicts = (*this)(data);
			auto loss = (*loss_)(predicts, label);
			if (verbose)
			{
				std::cout << "Step:" << step << " Loss:" << loss << " ";
				//TODO
				std::cout << std::endl;
			}
			step++;
		}

	}
}

void cppbp::model::Model::save(const string& filename)
{
	ofstream ofs{ filename, ios::binary };
	auto [mem, sz] = serialize();
	ofs.write(mem.get(), sz);
}

std::optional<cppbp::model::Model> cppbp::model::Model::from_file(const string& filename)
{
	std::ifstream infile{ filename, ios::binary };

	infile.seekg(0, std::ios::end);
	size_t length = infile.tellg();
	infile.seekg(0, std::ios::beg);

	auto buffer = make_unique<char[]>(length);

	infile.read(buffer.get(), length);

	Model mdl{};
	mdl.deserialize(buffer.get());

	return mdl;
}

std::tuple<std::shared_ptr<char[]>, size_t> cppbp::model::Model::serialize()
{
	auto l = input_->next();
	auto [mem, size] = input_->serialize();
	while (l)
	{
		auto [m, s] = l->serialize();
		auto old = mem;

		mem = make_shared<char[]>(size + s);

		memmove(mem.get(), old.get(), size);
		memmove(mem.get() + size, m.get(), s);
		size += s;

		l = l->next();
	}

	return { mem, size };
}

char* cppbp::model::Model::deserialize(char* data)
{
	return nullptr;
}
