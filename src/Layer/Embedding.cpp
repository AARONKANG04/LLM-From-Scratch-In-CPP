#include <Layer/Embedding.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <cmath>

Embedding::Embedding(int vocab_size, int embedding_dim) {
    float sigma = std::sqrt(2.0f / float(vocab_size + embedding_dim));
    weights = tensor_normal({vocab_size, embedding_dim}, true, 0.0f, sigma);
}

std::shared_ptr<Tensor> Embedding::forward(std::shared_ptr<Tensor> input) {
    return embedding(input, weights);
}

std::vector<std::shared_ptr<Tensor>> Embedding::parameters() {
    return std::vector<std::shared_ptr<Tensor>>{weights};
}

std::shared_ptr<Embedding> embedding(int vocab_size, int embedding_dim) {
    return std::make_shared<Embedding>(vocab_size, embedding_dim);
}