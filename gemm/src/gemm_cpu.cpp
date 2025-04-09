#include <torch/torch.h>

void matmul_cpu_kernel(const float *mat1,
                       const float *mat2, 
                       float *output,
                       const int m,
                       const int n,
                       const int k) {
    // printf("matmul_cpu_kernel\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float value = 0.0f;
            for (int l = 0; l < k; ++l) {
                value += mat1[i * k + l] * mat2[l * n + j];
            }
            output[i*n + j] = value;
            // printf("output[%d][%d] = %f\n", i, j, value);
        }
    }
}


void matmul_block_kernel(const float *mat1,  // m*k
                         const float *mat2,  // k*n
                         float *output,
                         const int m,
                         const int n,
                         const int k,
                         const int block_size) {
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int l = 0; l < k; l += block_size) {
                int i_end = std::min(i + block_size, m);
                int j_end = std::min(j + block_size, n);
                int l_end = std::min(l + block_size, k);
                // printf("A_block, i=%d, i_end=%d, l=%d, l_end=%d\n", i, i_end, l, l_end);
                // printf("B_block, l=%d, l_end=%d, j=%d, j_end=%d\n", l, l_end, j, j_end);

                for (int a_i = i; a_i < i_end; a_i++) {
                    for (int b_j = j; b_j < j_end; b_j++) {
                        float sum = 0.0f;
                        for (int a_l = l; a_l < l_end; a_l++) {
                            sum += mat1[a_i * k + a_l] * mat2[a_l * n + b_j];
                        }
                        output[a_i * n + b_j] += sum;
                    }
                }
            }
        }
    }
}


torch::Tensor matmul_cpu_wrapper(const torch::Tensor& input1, 
                                 const torch::Tensor& input2, 
                        torch::Tensor& output) {
    const float* tensor1_ptr = input1.data_ptr<float>();
    const float* tensor2_ptr = input2.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const int m = input1.size(0);
    const int k = input1.size(1);
    const int n = input2.size(1);

    matmul_cpu_kernel(tensor1_ptr, tensor2_ptr, output_ptr, m, n, k);
    return output;
}

torch::Tensor matmul_block_cpu_wrapper(const torch::Tensor& input1, 
                                       const torch::Tensor& input2, 
                                       torch::Tensor& output,
                                       const int block_size) {
    const float* tensor1_ptr = input1.data_ptr<float>();
    const float* tensor2_ptr = input2.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const int m = input1.size(0);
    const int k = input1.size(1);
    const int n = input2.size(1);

    matmul_block_kernel(tensor1_ptr, tensor2_ptr, output_ptr, m, n, k, block_size);
    return output;
}