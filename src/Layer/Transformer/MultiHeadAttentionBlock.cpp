#include <Layer/Transformer/MultiHeadAttentionBlock.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <cassert>

MultiHeadAttentionBlock::MultiHeadAttentionBlock(int d_model, int n_heads) : d_model(d_model), n_heads(n_heads) {
    assert(d_model % n_heads == 0);
    int d_head = d_model / n_heads;

    for (int i = 0; i < n_heads; ++i) {
        heads.push_back(std::make_shared<SingleHeadAttentionBlock>(d_model, d_head));
    }
    mlp = linear(d_model, d_model, true);
}

std::shared_ptr<Tensor> MultiHeadAttentionBlock::forward(std::shared_ptr<Tensor> input) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int i = 0; i < n_heads; ++i) {
        outputs.push_back(heads[i]->forward(input));
    }
    auto out = concat_last_dim(outputs);
    return mlp->forward(out);
}

std::vector<std::shared_ptr<Tensor>> MultiHeadAttentionBlock::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;

    for (const auto& head : heads) {
        auto head_params = head->parameters();
        params.insert(params.end(), head_params.begin(), head_params.end());
    }

    auto mlp_params = mlp->parameters();

    params.insert(params.end(), mlp_params.begin(), mlp_params.end());

    return params;
}

std::shared_ptr<MultiHeadAttentionBlock> multi_head_attention_block(int d_model, int n_heads) {
    return std::make_shared<MultiHeadAttentionBlock>(d_model, n_heads);
}