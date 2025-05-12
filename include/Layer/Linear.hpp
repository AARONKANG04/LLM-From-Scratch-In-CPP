#pragma once

#include <Datatype/Tensor.hpp>
#include <Network/Module.hpp>

#include <memory>

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool use_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> biases;
    bool use_bias;    
};

std::shared_ptr<Linear> linear(int in_features, int out_features, bool use_bias = true);

