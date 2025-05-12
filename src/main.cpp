#include <Datatype/Tensor.hpp>
#include <Data/DataLoader.hpp>
#include <Layer/Transformer/TransformerBlock.hpp>
#include <Network/Transformer.hpp>
#include <Optimizer/SGD.hpp>
#include <Criterion/CrossEntropyLoss.hpp>
#include <Layer/Softmax.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    DataLoader dataloader("../input.txt", /*batch=*/32, /*seq_len=*/64);
    CrossEntropyLoss criterion;
    Softmax softmax;
    auto model = transformer(/*vocab_size=*/dataloader.get_vocab_size(), /*max_seq_len=*/64, /*d_model=*/64, /*n_heads=*/4, /*n_layers=*/3);
    SGD optimizer(model->parameters(), 8e-4);

    const int EPOCHS = 100;
    const int STEPS_PER_EPOCH = 50;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;
        
        for (int step = 0; step < STEPS_PER_EPOCH; ++step) {
            auto start = std::chrono::high_resolution_clock::now();

            auto [input_tensor, label_tensor] = dataloader.get_batch();
            auto logits = model->forward(input_tensor);
            auto probs = softmax.forward(logits);
            auto loss = criterion.forward(probs, label_tensor);

            loss->backward();
            optimizer.step();
            optimizer.zero_grad();

            auto end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(end - start).count();

            total_loss += loss->data[0];
            std::cout << "Step: " << step
                      << " | Loss: " << loss->data[0] 
                      << " | Time (ms): " << step_ms << "\n";
        }

        std::vector<int> gen = {4, 5};
        int seq_len = 64;
        std::mt19937 rng(std::random_device{}());

        for (int i = 0; i < 64; ++i) {
            std::vector<int> ctx_tokens = gen.size() > seq_len
                ? std::vector<int>(gen.end() - seq_len, gen.end())
                : gen;

            std::vector<float> vf(ctx_tokens.begin(), ctx_tokens.end());
            auto ctx = tensor(vf, {1, (int)ctx_tokens.size()}, false);

            auto out = model->forward(ctx);
            const auto& sh = out->shape;
            int V = sh[2], T = sh[1];
            int base = (T - 1) * V;

            
            std::vector<float> probs(V);
            float sum = 0.0f;
            for (int j = 0; j < V; ++j) {
                float val = std::exp(out->data[base + j]);
                probs[j] = val;
                sum += val;
            }
            for (float& p : probs) p /= sum;

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next = dist(rng);

            gen.push_back(next);
        }

        std::string sample = dataloader.decode(gen);
        std::cout << "\n>>> sample: "
                    << sample
                    << "\n\n";

        std::cout << "Epoch " << (epoch+1) << "/" << EPOCHS
                << " | avg loss: " << (total_loss / STEPS_PER_EPOCH) << "\n";
    }

    return 0;
}
