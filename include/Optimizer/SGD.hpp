#pragma once

#include <Datatype/Tensor.hpp>

#include <vector>
#include <memory>

class SGD {
public:
    SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr);
    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;
};