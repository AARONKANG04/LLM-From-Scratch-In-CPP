#pragma once

#include <Network/Module.hpp>

class Embedding : public Module {
public:
    Embedding(int vocab_size, int embedding_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Tensor> weights;
};

std::shared_ptr<Embedding> embedding(int vocab_size, int embedding_dim);