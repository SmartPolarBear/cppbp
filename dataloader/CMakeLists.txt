cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp PRIVATE include)

target_sources(cppbp
        PRIVATE dataloader.cpp
        PRIVATE iris_dataloader.cpp
        PRIVATE minst_dataloader.cpp)


