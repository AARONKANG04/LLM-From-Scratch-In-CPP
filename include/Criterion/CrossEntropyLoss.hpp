#pragma once

#include <Datatype/Tensor.hpp>

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets);
};