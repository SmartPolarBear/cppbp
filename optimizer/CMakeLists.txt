cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp PRIVATE include)

target_sources(cppbp
        PRIVATE fixed_step_optimizer.cpp
        PRIVATE sgd_optimizer.cpp
        PRIVATE mse.cpp)


