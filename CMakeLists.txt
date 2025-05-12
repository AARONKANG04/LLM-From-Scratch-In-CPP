cmake_minimum_required(VERSION 3.28)
project(SFMLProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(main 
    src/main.cpp
    src/Datatype/Value.cpp
    src/Datatype/Value_ops.cpp
    src/Datatype/Tensor.cpp
    src/Datatype/Tensor_ops.cpp
    src/Layer/Linear.cpp
    src/Layer/ReLU.cpp
    src/Layer/Sigmoid.cpp
    src/Layer/Embedding.cpp
    src/Layer/Softmax.cpp
    src/Layer/LayerNorm.cpp
    src/Layer/Transformer/SingleHeadAttentionBlock.cpp
    src/Layer/Transformer/MultiHeadAttentionBlock.cpp
    src/Layer/Transformer/TransformerBlock.cpp
    src/Network/Sequential.cpp
    src/Network/Transformer.cpp
    src/Optimizer/SGD.cpp
    src/Criterion/MSELoss.cpp
    src/Criterion/CrossEntropyLoss.cpp
    src/Data/DataLoader.cpp
)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Configuring for AppleClang with OpenMP and Apple M2 optimization")

    # Specifiy the include and library directives for libomp
    target_include_directories(main PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(main PRIVATE /opt/homebrew/opt/libomp/lib)

    # Link against the OpenMP library
    target_link_libraries(main PRIVATE omp
        -fsanitize=address
    )

    # Add compiler options for OpenMP and Apple M2 optimization
    target_compile_options(main PRIVATE
        -Xpreprocessor -fopenmp
        -mcpu=apple-m2
        -fsanitize=address -g
    )

endif()