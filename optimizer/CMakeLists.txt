cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp INTERFACE include)

target_sources(cppbp
        INTERFACE fixed_step_optimizer.cpp
        INTERFACE sgd_optimizer.cpp
        INTERFACE mse.cpp)


