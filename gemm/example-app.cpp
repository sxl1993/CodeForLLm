#include <torch/torch.h>
#include <iostream>

// torch::Tensor matmul_cpu_wrapper(const torch::Tensor& input1,
//                                  const torch::Tensor& input2,
//                                  torch::Tensor& output);

// torch::Tensor matmul_cuda_wrapper(const torch::Tensor& input1,
//                                   const torch::Tensor& input2,
//                                   torch::Tensor& output);

// torch::Tensor matmul_block_cpu_wrapper(const torch::Tensor& input1,
//                                        const torch::Tensor& input2,
//                                        torch::Tensor& output,
//                                        const int block_size);

#define TILE_SIZE 16

gemm_kernel_sm(const float *a_ptr,
               const float *b_ptr,
               float *c_ptr,
               const int m,
               const int k,
               const int n);

int main()
{
  torch::manual_seed(42);
  const int m = 256;
  const int n = 256;
  const int k = 256;

  torch::Tensor tensor1 = torch::randn({m, k}, torch::dtype(torch::kFloat32)).to("cuda").contiguous();
  torch::Tensor tensor2 = torch::randn({k, n}, torch::dtype(torch::kFloat32)).to("cuda").contiguous();
  // torch::Tensor tensor1 = torch::arange(0, 16).reshape({m, k}).to("cuda").to(torch::kFloat32);
  // torch::Tensor tensor2 = torch::arange(0, 16).reshape({k, n}).to("cuda").to(torch::kFloat32);
  // torch::Tensor tensor1 = torch::ones(16).reshape({m, k}).to("cuda").to(torch::kFloat32);
  // torch::Tensor tensor2 = torch::ones(16).reshape({m, k}).to("cuda").to(torch::kFloat32) * 2;

  // std::cout << tensor1.sizes() << std::endl;
  // std::cout << tensor2.sizes() << std::endl;

  auto output1 = torch::empty({m, n}, torch::dtype(torch::kFloat32)).to("cuda");
  // output1 = matmul_cuda_wrapper(tensor1, tensor2, output1);
  // std::cout << output1.sizes() << std::endl;

  // auto output2 = torch::matmul(tensor1, tensor2);
  // std::cout << output2.sizes() << std::endl;

  // if (torch::allclose(output1, output2, 1e-03, 1e-03)) {
  //   printf("result is correct\n");
  // }

  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, m * k * sizeof(float));
  cudaMalloc((void **)&d_b, k * n * sizeof(float));
  cudaMalloc((void **)&d_c, m * n * sizeof(float));

  cudaMemcpy(d_a, tensor_a.data_ptr<float>(), m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, tensor_b.data_ptr<float>(), k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, m * n * sizeof(float));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

  gemm_kernel_sm<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
  cudaDeviceSynchronize();
}