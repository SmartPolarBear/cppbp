//
// Created by cleve on 5/11/2022.
//

#include <model/model.h>
#include <model/loss_output_callback.h>

#include <utils/utils.h>

#include <optimizer/mse.h>

#include <layer/fully_connected.h>
#include <layer/input.h>

#include <utility>
#include <iostream>

#include <fstream>

using namespace std;

using namespace Eigen;

using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::utils;

cppbp::model::Model::Model(layer::ILayer &layer, optimizer::ILossFunction &loss)
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

cppbp::model::Model::Model(std::vector<layer::ILayer> &layers, optimizer::ILossFunction &loss)
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

void cppbp::model::Model::optimize(cppbp::optimizer::IOptimizer &opt)
{
    input_->optimize(opt);
    opt.step();
}

void cppbp::model::Model::fit(cppbp::dataloader::DataLoader &dl,
                              size_t epoch,
                              cppbp::optimizer::IOptimizer &opt,
                              bool verbose,
                              size_t callback_skip_epoch,
                              std::optional<std::vector<std::shared_ptr<IModelCallback>>> cbks)
{
    std::vector<std::shared_ptr<IModelCallback>> callbacks{};
    if (cbks.has_value())
    {
        callbacks.assign(cbks.value().begin(), cbks.value().end());
    }

    if (verbose && callbacks.empty())
    {
        callbacks.push_back(IModelCallback::make<LossOutputCallback>());
    }

    const auto should_callback = [verbose, callback_skip_epoch](int64_t epoch)
    {
        return verbose && !(epoch % callback_skip_epoch);
    };

    if (verbose)
    {
        for (auto &c: callbacks)
        {
            std::cout << c->before_world() << " ";
        }
        std::cout << std::endl;
    }

    for (size_t e = 0; e < epoch; e++)
    {
        if (should_callback(e))
        {
            std::cout << "Epoch: " << e << std::endl << "Train..." << " ";
            for (auto &c: callbacks)
            {
                std::cout << c->before_train(e);
            }
            std::cout << std::endl;
        }

        auto batch = dl.train_batch();
        for (size_t step = 0; auto &[data, label]: batch)
        {
            auto predicts = (*this)(data);
            auto loss = (*loss_)(predicts, label);

            if (should_callback(e))
            {
                std::cout << "Step:" << step << " ";
                for (auto &c: callbacks)
                {
                    std::cout << c->train_step(step, make_pair(data, label), predicts, loss) << " ";
                }
                std::cout << std::endl;
            }

            auto errors = loss_->derive(predicts, label);

            output_->set_errors(errors);
            this->backprop();
            this->optimize(opt);

            step++;
        }

        if (should_callback(e))
        {
            for (auto &c: callbacks)
            {
                std::cout << c->after_train(e) << " ";
            }
            std::cout << endl;
        }

        if (should_callback(e))
        {
            std::cout << "Eval...";
            for (auto &c: callbacks)
            {
                std::cout << c->before_eval(e) << " ";
            }
            std::cout << endl;
        }

        batch = dl.eval_batch();
        for (size_t step = 0; auto &[data, label]: batch)
        {
            auto predicts = (*this)(data);
            auto loss = (*loss_)(predicts, label);
            if (should_callback(e))
            {
                std::cout << "Step:" << step << " ";
                for (auto &c: callbacks)
                {
                    std::cout << c->eval_step(step, make_pair(data, label), predicts, loss) << " ";
                }
                std::cout << std::endl;
            }

            step++;
        }

        if (should_callback(e))
        {
            for (auto &c: callbacks)
            {
                std::cout << c->after_eval(e) << " ";
            }
            std::cout << std::endl;
        }

    }

    if (verbose)
    {
        for (auto &c: callbacks)
        {
            std::cout << c->after_world() << " ";
        }
        std::cout << std::endl;
    }
}

void cppbp::model::Model::save_state(const string &filename)
{
    ofstream ofs{filename, ios::binary};
    serialize(ofs);
}

void cppbp::model::Model::load_state(const string &filename)
{
    std::ifstream infile{filename, ios::binary};
    deserialize(infile);
}

ostream &cppbp::model::Model::serialize(std::ostream &out)
{
    out << magic();
    for (auto iter = input_; iter; iter = iter->next())
    {
        iter->serialize(out);
    }
    return out;
}

istream &cppbp::model::Model::deserialize(istream &input)
{
    if (!check_magic<uint64_t>(*this, input))
    {
        throw base::magic_checking_failure{};
    }

    for (auto iter = input_; iter; iter = iter->next())
    {
        iter->deserialize(input);
    }
    return input;
}

uint64_t Model::magic() const
{
    return magic_from_string<uint64_t>("MODEL000");
}
