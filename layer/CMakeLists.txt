cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp PRIVATE include)

target_sources(cppbp
        PRIVATE fully_connected.cpp
        PRIVATE neuron.cpp
        PRIVATE sigmoid.cpp)


