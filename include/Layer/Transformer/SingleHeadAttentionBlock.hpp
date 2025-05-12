#pragma once 

#include <Datatype/Tensor.hpp>
#include <Network/Module.hpp>
#include <Layer/Linear.hpp>
#include <Layer/Softmax.hpp>

#include <memory>
#include <vector>

class SingleHeadAttentionBlock : public Module {
public:
    SingleHeadAttentionBlock(int d_model, int d_head);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    std::shared_ptr<Linear> W_k;
    std::shared_ptr<Linear> W_q;
    std::shared_ptr<Linear> W_v;
    std::shared_ptr<Softmax> softmax_layer;
    std::shared_ptr<Tensor> causal_mask;
    int d_model;
    int d_head;

private:
    void make_causal_mask(int seq_len);
};

std::shared_ptr<SingleHeadAttentionBlock> single_head_attention_block(int d_model, int d_head);