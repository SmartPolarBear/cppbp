//
// Created by cleve on 5/18/2022.
//

#include <layer/input.h>
#include <utils/utils.h>

#include <base/exceptions.h>


#include <fmt/format.h>

using namespace std;
using namespace Eigen;


cppbp::layer::Input::Input(size_t size)
        : len_(size)
{
}

void cppbp::layer::Input::backprop()
{
    // Do nothing
}

void cppbp::layer::Input::forward()
{
    if (next_)
    {
        next_->set(values_);
        next_->forward();
    }
}

cppbp::layer::ILayer *cppbp::layer::Input::next()
{
    return next_;
}

cppbp::layer::ILayer *cppbp::layer::Input::prev()
{
    return nullptr; // It must be the leftmost layer
}

std::string cppbp::layer::Input::name() const
{
    return "Input";
}

std::string cppbp::layer::Input::summary() const
{
    stringstream ss{};
    ss << fmt::format("Input [{}]", len_);

    if (next_)
    {
        ss << "\n";
        ss << next_->summary();
    }

    return ss.str();
}

cppbp::layer::ILayer &cppbp::layer::Input::connect(cppbp::layer::ILayer &next)
{
    next.reshape(len_);

    this->set_next(&next);
    next.set_prev(this);

    return next;
}

void cppbp::layer::Input::set(Eigen::VectorXd vec)
{
    values_ = vec;
}

void cppbp::layer::Input::set_deltas(Eigen::VectorXd deltas)
{
    // Do nothing
}

void cppbp::layer::Input::set_errors(Eigen::VectorXd errors)
{
    // Do nothing
}

Eigen::VectorXd cppbp::layer::Input::get() const
{
    return values_;
}

cppbp::layer::IActivationFunction &cppbp::layer::Input::activation_function()
{
    throw base::not_implemented{};

}

cppbp::layer::ILayer &cppbp::layer::Input::operator|(cppbp::layer::ILayer &next)
{
    return connect(next);
}

void cppbp::layer::Input::reshape(size_t input)
{
    len_ = input;
}

void cppbp::layer::Input::optimize(cppbp::optimizer::IOptimizer &opt)
{
    if (next())
    {
        next()->optimize(opt);
    }
}

void cppbp::layer::Input::set_prev(cppbp::layer::ILayer *prev)
{
    // Do nothing
}

void cppbp::layer::Input::set_next(cppbp::layer::ILayer *next)
{
    next_ = next;
}

ostream &cppbp::layer::Input::serialize(std::ostream &out)
{
    out << magic();
    out << len_;
    return out;
}

istream &cppbp::layer::Input::deserialize(istream &input)
{
    if (!check_magic<uint16_t>(*this, input))
    {
        throw base::magic_checking_failure{};
    }

    input >> len_;

    return input;
}

uint16_t cppbp::layer::Input::magic() const
{
    return utils::magic_from_string<uint16_t>("IP");
}
