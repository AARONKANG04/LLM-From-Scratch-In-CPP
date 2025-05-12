#pragma once

#include <Datatype/Tensor.hpp>

#include <memory>
#include <vector>

class Module {
public:
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() { return {}; }
    virtual ~Module() = default;
};