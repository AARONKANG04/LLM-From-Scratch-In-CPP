#pragma once

#include <Layer/Transformer/SingleHeadAttentionBlock.hpp>
#include <Network/Module.hpp>
#include <Datatype/Tensor.hpp>
#include <Layer/ReLU.hpp>

#include <memory>

class MultiHeadAttentionBlock : public Module {
public:
    MultiHeadAttentionBlock(int d_model, int n_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::vector<std::shared_ptr<SingleHeadAttentionBlock>> heads;
    std::shared_ptr<Linear> mlp;
    int d_model;
    int n_heads;
};

std::shared_ptr<MultiHeadAttentionBlock> multi_head_attention_block(int d_model, int n_heads);