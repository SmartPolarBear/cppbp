cmake_minimum_required(VERSION 3.19)

include(GoogleTest)

add_subdirectory(gtest)

add_executable(cppbp_test
        relu_test.cpp
        sigmoid_test.cpp
        )

target_include_directories(cppbp_test
        PRIVATE include
        PRIVATE ${gtest_SOURCE_DIR}
        PRIVATE ${gtest_SOURCE_DIR}/include)

target_link_libraries(cppbp_test
        PUBLIC gtest
        PUBLIC gtest_main
        PRIVATE cppbp
        PRIVATE Eigen3::Eigen
        PRIVATE csv2::csv2
        PRIVATE fmt::fmt
        PRIVATE GSL
        PRIVATE xtl
        PRIVATE xtensor
        )

gtest_discover_tests(cppbp_test)
