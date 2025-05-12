#pragma once

#include <Network/Module.hpp>
#include <Datatype/Tensor.hpp>

#include <memory>

class Sigmoid : public Module {
public:
    Sigmoid() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {};
    }
};

std::shared_ptr<Sigmoid> sigmoid();