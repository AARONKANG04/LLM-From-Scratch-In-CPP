#include <Data/DataLoader.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <random>
#include <sstream>

DataLoader::DataLoader(const std::string& file_path, int batch_size, int sequence_length)
    : file_path(file_path), batch_size(batch_size), sequence_length(sequence_length) {
    tokenize();
}

void DataLoader::tokenize() {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    raw_text = std::string((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

    std::unordered_map<char, bool> seen;
    for (char ch : raw_text) {
        seen[ch] = true;
    }

    int idx = 0;
    for (const auto& [ch, _] : seen) {
        stoi[ch] = idx;
        itos[idx] = ch;
        ++idx;
    }
    vocab_size = stoi.size();

    for (char ch : raw_text) {
        encoded_data.push_back(stoi[ch]);
    }

    file.close();
}

std::string DataLoader::decode(const std::vector<int>& indices) {
    std::ostringstream oss;

    for (int val : indices) {
        auto it = itos.find(val);
        if (it != itos.end()) {
            oss << it->second;
        } else {
            oss << '?';
        }
    }

    return oss.str();
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::get_batch() {
    int total_tokens = batch_size * sequence_length;
    if (current_index + total_tokens + 1 >= (int)encoded_data.size()) {
        current_index = 0;
    }

    std::vector<float> input_data;
    std::vector<float> label_data;

    for (int i = 0; i < total_tokens; ++i) {
        input_data.push_back((float)encoded_data[current_index + i]);
        label_data.push_back((float)encoded_data[current_index + i + 1]);  
    }

    current_index += total_tokens;

    auto input_tensor = std::make_shared<Tensor>(input_data, std::vector<int>{batch_size, sequence_length}, false);
    auto label_tensor = std::make_shared<Tensor>(label_data, std::vector<int>{batch_size, sequence_length}, false);

    return {input_tensor, label_tensor};
}

int DataLoader::get_vocab_size() const {
    return stoi.size();
}