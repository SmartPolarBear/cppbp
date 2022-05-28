cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp INTERFACE include)

target_sources(cppbp
        INTERFACE fully_connected.cpp
        INTERFACE input.cpp
        INTERFACE layer.cpp
        INTERFACE activation_function.cpp
        INTERFACE relu.cpp
        INTERFACE tanh.cpp
        INTERFACE dropout.cpp
        INTERFACE batch_norm.cpp
        INTERFACE layer_norm.cpp
        INTERFACE sigmoid.cpp
        INTERFACE softmax.cpp
        INTERFACE xavier_intializer.cpp
        INTERFACE he_initializer.cpp
        INTERFACE random_initializer.cpp)


