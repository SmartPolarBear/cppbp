cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp INTERFACE include)

target_sources(cppbp
        INTERFACE dataloader.cpp
        INTERFACE iris_dataloader.cpp
        INTERFACE minst_dataloader.cpp)


