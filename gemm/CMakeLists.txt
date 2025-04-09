cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "/workspace/CodeForLLm/libtorch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

# 添加 CUDA 文件
set(CUDA_SOURCES src/gemm_cuda.cu)
# 添加 CPP 文件
set(CPP_SOURCES src/gemm_cpu.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_executable(example-app example-app.cpp "${CUDA_SOURCES}" "${CPP_SOURCES}")
add_executable(example-app src/sgemm_v1.cu)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_include_directories(example-app PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

