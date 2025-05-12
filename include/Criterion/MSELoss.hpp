#pragma once

#include <Datatype/Tensor.hpp>
#include <memory>

class MSELoss {
public:
    MSELoss() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> preds, std::shared_ptr<Tensor> targets);
};