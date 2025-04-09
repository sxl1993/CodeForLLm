#include <torch/torch.h>

#define TILE_SIZE 16 // 定义共享内存的块大小
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void __global__ matmul_cuda_kernel(const float *mat1,
                                   const float *mat2, 
                                   float *output,
                                   const int m,
                                   const int n,
                                   const int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / n;
    int col = tid % n;

    float value = 0.0f;
    for (int i = 0; i < k; ++i) {
        value += mat1[row * k + i] * mat2[i * n + col];
    }
    output[row * n + col] = value;
}


void __global__ matmul_cuda_kernel_sm(const float *A,
                                      const float *B, 
                                      float *output,
                                      const int m,
                                      const int k,
                                      const int n) {
    __shared__ float sharedA[TILE_SIZE * TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE * TILE_SIZE];
    
    int numTileCols = (n + TILE_SIZE - 1) / TILE_SIZE; // 按列分块
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("numTileCols: %d\n", numTileCols);
    int tile_idx = blockIdx.x;  // 一维 block 对应一个 tile
    int tile_row = tile_idx / numTileCols;  // tile 在矩阵中的行号
    int tile_col = tile_idx % numTileCols;  // tile 在矩阵中的列号

    // 每个 block 内的线程总数为 TILE_SIZE * TILE_SIZE（1D 线程索引）
    // 通过 threadIdx.x 计算线程在 tile 内的二维坐标
    int t_row = threadIdx.x / TILE_SIZE;  // tile 内的行索引
    int t_col = threadIdx.x % TILE_SIZE;  // tile 内的列索引

    // 全局矩阵 C 的行、列索引
    int row = tile_row * TILE_SIZE + t_row;
    int col = tile_col * TILE_SIZE + t_col;

    printf("bid=%d, tid=%d, row=%d, col=%d, t_row=%d, t_col=%d\n", tile_idx, tid, row, col, t_row, t_col);
    float sum = 0.0f;
    for (int i = 0; i < k; i += TILE_SIZE) {
        // 从 A 载入当前 tile 的一部分：
        // A按行加载tile，行是全局行索引*k行， 列为tile的列索引(i + t_col)
        if (row < m && (i + t_col) < k)
            sharedA[t_row * TILE_SIZE + t_col] = A[row * k + (i + t_col)];
        else
            sharedA[t_row * TILE_SIZE + t_col] = 0.0f;

        // 从 B 载入当前 tile 的一部分：
        // B按列加载，行是tile的行索引(i+t_row)*n行, 列为全局列索引col
        if ((i + t_row) < k && col < n)
            sharedB[t_row * TILE_SIZE + t_col] = B[(i + t_row) * n + col];
        else
            sharedB[t_row * TILE_SIZE + t_col] = 0.0f;

        __syncthreads();

        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += sharedA[t_row * TILE_SIZE + t] * sharedB[t * TILE_SIZE + t_col];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        output[row * n + col] = sum;
    }
}


__global__ void gemm_kernel(const float *A,
                            const float *B, 
                            float *C,
                            const int M,
                            const int K,
                            const int N) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    float dot_prod = 0.0f;
    for (int k = 0; k < K; k++) {
        dot_prod += A[row * K + k] * B[k * N + col];
    } 

    C[row * N + col] = dot_prod;
}


__global__ void gemm_kernel_sm(const float *a_ptr,
                               const float *b_ptr, 
                               float *c_ptr,
                               const int m,
                               const int k,
                               const int n) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE + 1];

    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    const int tile_row = threadIdx.y;
    const int tile_col = threadIdx.x;

    float dot_prod = 0.0f;
    for (int tile_offset = 0; tile_offset < k; tile_offset += TILE_SIZE) {
        const int a_row = row;
        const int a_col = tile_col + tile_offset;
        a_tile[tile_row][tile_col] = a_ptr[OFFSET(a_row, a_col, k)];
        const int b_row = tile_row + tile_offset;
        const int b_col = col;
        b_tile[tile_row][tile_col] = b_ptr[OFFSET(b_row, b_col, n)];
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            dot_prod += a_tile[tile_row][i] * b_tile[i][tile_col];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c_ptr[OFFSET(row, col, n)] = dot_prod;
    }
    
}


torch::Tensor matmul_cuda_wrapper(const torch::Tensor& input1, 
                                  const torch::Tensor& input2, 
                                  torch::Tensor& output) {
    const float* tensor1_ptr = input1.data_ptr<float>();
    const float* tensor2_ptr = input2.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const int m = input1.size(0);
    const int k = input1.size(1);
    const int n = input2.size(1);


    // int num_threads = TILE_SIZE * TILE_SIZE; // 线程数
    // int num_blocks = ((m * n) + num_threads - 1) / num_threads; // 计算 block 数
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel_sm<<<grid, block>>>(tensor1_ptr, tensor2_ptr, output_ptr, m, k, n);
    return output;
}