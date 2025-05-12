#pragma once

#include <Layer/Transformer/TransformerBlock.hpp>
#include <Datatype/Tensor.hpp>
#include <Layer/Embedding.hpp>

#include <memory>
#include <vector>

class Transformer : public Module {
public:
    Transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Embedding> token_embedding;
    std::shared_ptr<Embedding> position_embedding;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    std::shared_ptr<Linear> projection;
    int vocab_size;
    int d_model; 
    int n_heads; 
    int n_layers;
};

std::shared_ptr<Transformer> transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers);