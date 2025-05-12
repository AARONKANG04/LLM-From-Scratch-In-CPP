#pragma once

#include <Datatype/Tensor.hpp>
#include <Network/Module.hpp>

#include <memory>
#include <vector>

class LayerNorm : public Module {
public:
    LayerNorm(int dim);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    std::vector<std::shared_ptr<Tensor>> parameters() override;

    int dim;
    static constexpr float eps = 1e-5f;

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
};

std::shared_ptr<LayerNorm> layer_norm(int dim);