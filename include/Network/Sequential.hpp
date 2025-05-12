#include <Network/Module.hpp>

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;
    Sequential() = default;
    Sequential(const std::vector<std::shared_ptr<Module>>& modules) : layers(modules) {}
    void add(const std::shared_ptr<Module>& layer);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
};