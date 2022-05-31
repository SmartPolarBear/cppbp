//
// Created by cleve on 5/26/2022.
//

#include <layer/layer_norm.h>
#include <utils/utils.h>

#include <base/exceptions.h>

#include <sstream>

#include <fmt/format.h>
#include <gsl/assert>

using namespace std;

using namespace cppbp::utils;

cppbp::layer::LayerNorm::LayerNorm()
        : id_(cppbp::layer::LayerNorm::objects_alive)
{
}

void cppbp::layer::LayerNorm::backprop()
{
    Eigen::VectorXd errors = errors_.cwiseProduct(gammas_);

    const auto a = errors_.dot(gammas_);
    const auto b = errors_.dot(gammas_.cwiseProduct(normalized_input_));

    const auto H = errors_.size();
    for (int i = 0; i < errors.size(); i++)
    {
        errors[i] -= (1.0 / H) * (a + b * normalized_input_[i]);
    }

    errors /= sigma_;

    if (prev())
    {
        prev()->set_errors(errors);
        prev()->backprop();
    }
}

void cppbp::layer::LayerNorm::forward()
{
    if (next())
    {
        next()->set(activations_);
        next()->forward();
    }
}

void cppbp::layer::LayerNorm::set(Eigen::VectorXd vec)
{
    Expects(!vec.hasNaN());

    input_ = vec;

    auto average = input_.sum() / input_.size();
    auto sq_average = input_.cwiseProduct(input_).sum() / input_.size();

    sigma_ = std::sqrt(sq_average - average * average + base::EPS);

    scaled_input_ = input_ - Eigen::VectorXd::Ones(input_.size()) * average;
    normalized_input_ = scaled_input_ / sigma_;

    activations_ = normalized_input_.cwiseProduct(gammas_) + betas_;

    Ensures(!activations_.hasNaN());
}

void cppbp::layer::LayerNorm::optimize(cppbp::optimizer::IOptimizer &opt)
{
    Expects(!betas_.hasNaN());
    Expects(!gammas_.hasNaN());


    betas_ = opt.optimize(betas_, errors_);
    gammas_ = opt.optimize(gammas_, errors_.cwiseProduct(normalized_input_));

    if (next_)
    {
        next_->optimize(opt);
    }

    Ensures(!betas_.hasNaN());
    Expects(!gammas_.hasNaN());
}

void cppbp::layer::LayerNorm::set_deltas(Eigen::VectorXd dlts)
{
}

void cppbp::layer::LayerNorm::set_errors(Eigen::VectorXd errors)
{
    errors_ = errors;
}

Eigen::VectorXd cppbp::layer::LayerNorm::get() const
{
    return activations_;
}

cppbp::layer::ILayer *cppbp::layer::LayerNorm::next()
{
    return next_;
}

cppbp::layer::ILayer *cppbp::layer::LayerNorm::prev()
{
    return prev_;
}

void cppbp::layer::LayerNorm::set_prev(cppbp::layer::ILayer *prev)
{
    prev_ = prev;
}

void cppbp::layer::LayerNorm::set_next(cppbp::layer::ILayer *next)
{
    next_ = next;
}

ostream &cppbp::layer::LayerNorm::serialize(std::ostream &out)
{
    out << magic();
    out << gammas_.size();
    for (int i = 0; i < gammas_.size(); i++)
    {
        out << gammas_.coeffRef(i);
    }
    out << betas_.size();
    for (int i = 0; i < betas_.size(); i++)
    {
        out << betas_.coeffRef(i);
    }
    return out;
}

istream &cppbp::layer::LayerNorm::deserialize(istream &input)
{
    if (!check_magic<uint16_t>(*this, input))
    {
        throw base::magic_checking_failure{};
    }

    int gamma_size{0};
    input >> gamma_size;
    for (int i = 0; i < gamma_size; i++)
    {
        input >> gammas_.coeffRef(i);
    }

    int beta_size{0};
    input >> beta_size;
    for (int i = 0; i < beta_size; i++)
    {
        input >> betas_.coeffRef(i);
    }
    return input;
}

std::string cppbp::layer::LayerNorm::name() const
{
    return fmt::format("Layer Norm {}", id_);
}

std::string cppbp::layer::LayerNorm::summary() const
{
    stringstream ss{};
    ss << name();

    if (next_)
    {
        ss << "\n";
        ss << next_->summary();
    }

    return ss.str();
}

cppbp::layer::ILayer &cppbp::layer::LayerNorm::connect(cppbp::layer::ILayer &next)
{
    next.reshape(input_len_);

    this->set_next(&next);
    next.set_prev(this);

    return next;
}

cppbp::layer::IActivationFunction &cppbp::layer::LayerNorm::activation_function()
{
    throw base::not_implemented{};
}

cppbp::layer::ILayer &cppbp::layer::LayerNorm::operator|(cppbp::layer::ILayer &next)
{
    return connect(next);
}

void cppbp::layer::LayerNorm::reshape(size_t input)
{
    input_len_ = input;
    gammas_ = Eigen::VectorXd::Random(input_len_);
    betas_ = Eigen::VectorXd::Random(input_len_);
}

uint16_t cppbp::layer::LayerNorm::magic() const
{
    return magic_from_string<uint16_t>("LN");
}
