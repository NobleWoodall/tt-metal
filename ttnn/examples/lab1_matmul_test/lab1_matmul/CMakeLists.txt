cmake_minimum_required(VERSION 3.22...4.2)
project(ttnn_example_lab1_matmul)

add_executable(ttnn_example_lab1_matmul)
target_sources(ttnn_example_lab1_matmul PRIVATE lab1_matmul.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab1_matmul
    PRIVATE
        TT::Metalium
        TTNN::TTNN
)
