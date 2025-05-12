#include <Layer/Transformer/SingleHeadAttentionBlock.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <limits>

SingleHeadAttentionBlock::SingleHeadAttentionBlock(int d_model, int d_head) 
    : d_model(d_model), d_head(d_head) {
    float sigma_head = std::sqrt(2.0 / float(d_head)); 
    float sigma_model = std::sqrt(2.0 / float(d_model));
    W_k = linear(d_model, d_head, false);
    W_q = linear(d_model, d_head, false);
    W_v = linear(d_model, d_head, false);

    softmax_layer = softmax();
}

std::shared_ptr<Tensor> SingleHeadAttentionBlock::forward(std::shared_ptr<Tensor> input) {
    auto Q = W_q->forward(input);
    auto K = W_k->forward(input);
    auto V = W_v->forward(input);

    auto K_T = transpose_last2(K);
    auto scores = matmul(Q, K_T);

    scores = scale(scores, 1.0f / std::sqrt(d_head));

    int seq_len = scores->shape[scores->shape.size() - 1];
    if (!causal_mask || causal_mask->shape[0] != seq_len) {
        make_causal_mask(seq_len);
    }

    scores = scores + causal_mask; 

    auto attn_weights = softmax_layer->forward(scores);
    auto output = matmul(attn_weights, V);

    return output;
}

std::vector<std::shared_ptr<Tensor>> SingleHeadAttentionBlock::parameters() {
    return std::vector{W_q->weights, W_k->weights, W_v->weights};
}

void SingleHeadAttentionBlock::make_causal_mask(int seq_len) {
    std::vector<float> mask_data(seq_len * seq_len, 0.0f);
    float neg_inf = std::numeric_limits<float>::lowest();

    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = neg_inf; 
        }
    }

    causal_mask = std::make_shared<Tensor>(mask_data, std::vector<int>{seq_len, seq_len}, false);
}

std::shared_ptr<SingleHeadAttentionBlock> single_head_attention_block(int d_model, int d_head) {
    return std::make_shared<SingleHeadAttentionBlock>(d_model, d_head);
}