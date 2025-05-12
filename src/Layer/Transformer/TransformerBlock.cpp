#include <Layer/Transformer/TransformerBlock.hpp>
#include <Datatype/Tensor_ops.hpp>

TransformerBlock::TransformerBlock(int d_model, int n_heads) 
    : d_model(d_model), n_heads(n_heads) {
    attn_layer = multi_head_attention_block(d_model, n_heads);
    norm1 = layer_norm(d_model);
    norm2 = layer_norm(d_model);
    up_proj = linear(d_model, 3 * d_model, true);
    relu_layer = relu();
    down_proj = linear(3 * d_model, d_model, true);
}

std::shared_ptr<Tensor> TransformerBlock::forward(std::shared_ptr<Tensor> input) {
    auto attn_out = attn_layer->forward(norm1->forward(input));
    auto x = input + attn_out;

    auto ff_out = up_proj->forward(norm2->forward(x));
    ff_out = relu_layer->forward(ff_out);
    ff_out = down_proj->forward(ff_out);

    auto out = x + ff_out;
    return out;
}

std::vector<std::shared_ptr<Tensor>> TransformerBlock::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;

    auto attn_params = attn_layer->parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());

    auto up_params = up_proj->parameters();
    params.insert(params.end(), up_params.begin(), up_params.end());

    auto down_params = down_proj->parameters();
    params.insert(params.end(), down_params.begin(), down_params.end());

    return params;
}

std::shared_ptr<TransformerBlock> transformer_block(int d_model, int n_heads) {
    return std::make_shared<TransformerBlock>(d_model, n_heads);
}
