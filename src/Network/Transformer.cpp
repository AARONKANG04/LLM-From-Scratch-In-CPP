#include <Network/Transformer.hpp>
#include <Layer/Transformer/TransformerBlock.hpp>
#include <Layer/Linear.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <iostream>

Transformer::Transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers) : vocab_size(vocab_size), d_model(d_model), n_heads(n_heads), n_layers(n_layers) {
    token_embedding = embedding(vocab_size, d_model);
    position_embedding = embedding(max_seq_len, d_model);

    for (int i = 0; i < n_layers; ++i) {
        blocks.push_back(transformer_block(d_model, n_heads));
    }

    projection = linear(d_model, vocab_size, true);
}

std::shared_ptr<Tensor> Transformer::forward(const std::shared_ptr<Tensor> input) {
    auto tok_embd = token_embedding->forward(input);

    int seq_len = input->shape.back(); 
    std::vector<float> pos_indices(seq_len);
    for (int i = 0; i < seq_len; ++i) pos_indices[i] = i;
    auto pos_tensor = tensor(pos_indices, {seq_len}, false);

    auto pos_embd = position_embedding->forward(pos_tensor);
    
    auto out = tok_embd + pos_embd;

    for (std::shared_ptr<TransformerBlock> block : blocks) {
        out = block->forward(out);
    }
    auto logits = projection->forward(out);
    return logits;
}

std::vector<std::shared_ptr<Tensor>> Transformer::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;

    auto tok_params = token_embedding->parameters();
    params.insert(params.end(), tok_params.begin(), tok_params.end());

    auto pos_params = position_embedding->parameters();
    params.insert(params.end(), pos_params.begin(), pos_params.end());

    for (auto& block : blocks) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }

    auto proj_params = projection->parameters();
    params.insert(params.end(), proj_params.begin(), proj_params.end());

    return params;
}

std::shared_ptr<Transformer> transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers) {
    return std::make_shared<Transformer>(vocab_size, max_seq_len, d_model, n_heads, n_layers);
}