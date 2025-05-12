#pragma once

#include <Datatype/Tensor.hpp>
#include <Network/Module.hpp>

class Softmax : public Module {
public:
    Softmax() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {}; 
    }
};

std::shared_ptr<Softmax> softmax();

