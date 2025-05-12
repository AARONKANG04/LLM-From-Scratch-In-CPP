#pragma once

#include <Datatype/Tensor.hpp>

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& file_path, int batch_size, int sequence_length);
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_batch();
    std::string decode(const std::vector<int>& indices);
    void tokenize();
    int get_vocab_size() const;

    std::string file_path;
    int batch_size;
    int sequence_length;
    
    std::string raw_text;
    std::vector<int> encoded_data;
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    int vocab_size;
    int current_index = 0;
};