#pragma once

#include <Network/Module.hpp>
#include <Layer/Transformer/MultiHeadAttentionBlock.hpp>
#include <Layer/Linear.hpp>
#include <Layer/ReLU.hpp>
#include <Layer/LayerNorm.hpp>

class TransformerBlock : public Module {
public:
    TransformerBlock(int d_model, int n_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<MultiHeadAttentionBlock> attn_layer;
    std::shared_ptr<LayerNorm> norm1;
    std::shared_ptr<LayerNorm> norm2;
    std::shared_ptr<Linear> up_proj;
    std::shared_ptr<ReLU> relu_layer;
    std::shared_ptr<Linear> down_proj;
    int d_model;
    int n_heads;
};

std::shared_ptr<TransformerBlock> transformer_block(int d_model, int n_heads);