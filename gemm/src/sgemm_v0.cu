#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
// #include <torch/torch.h>
#include <cublas_v2.h>

#define TILE_SIZE 16                                // Block 处理的 tile 大小
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 8
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_X 8
#define OFFSET(row, col, ld) ((row) * (ld) + (col)) // 计算 1D 数组索引
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

__global__ void gemm_kernel(const float *a_ptr,
                            const float *b_ptr,
                            float *c_ptr,
                            const int m,
                            const int k,
                            const int n)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < k; i++)
    {
        sum += a_ptr[row * k + i] * b_ptr[i * n + col];
    }
    c_ptr[row * n + col] = sum;
}

__global__ void gemm_kernel_sm_v1(const float *a_ptr,
                                  const float *b_ptr,
                                  float *c_ptr,
                                  const int m,
                                  const int k,
                                  const int n)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_row = threadIdx.y;
    const int tile_col = threadIdx.x;

    float dot_prod = 0.0f;
    for (int tile_offset = 0; tile_offset < k; tile_offset += TILE_SIZE)
    {
        const int a_row = row;
        const int a_col = tile_col + tile_offset;
        if (a_row < m && a_col < k)
            a_tile[tile_row][tile_col] = a_ptr[OFFSET(a_row, a_col, k)];
        else
            a_tile[tile_row][tile_col] = 0.0f;

        const int b_row = tile_row + tile_offset;
        const int b_col = col;
        if (b_row < k && b_col < n)
            b_tile[tile_row][tile_col] = b_ptr[OFFSET(b_row, b_col, n)];
        else
            b_tile[tile_row][tile_col] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            dot_prod += a_tile[tile_row][i] * b_tile[i][tile_col];

        __syncthreads();
    }

    if (row < m && col < n)
        c_ptr[OFFSET(row, col, n)] = dot_prod;
}

__global__ void gemm_kernel_sm_float4(float *a_ptr,
                                      float *b_ptr, 
                                      float *c_ptr,
                                      const int m,
                                      const int k,
                                      const int n)
{
    __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

    float r_c[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0};

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BLOCK_SIZE_M + load_a_smem_m;
    int load_b_gmem_n = bx * BLOCK_SIZE_N + load_b_smem_n;

    for (int bk = 0; bk < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; bk++) {
        int load_a_gmem_k = bk * BLOCK_SIZE_K + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, k);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a_ptr[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BLOCK_SIZE_K + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, k);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b_ptr[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_Y; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_X; n++) {
                    int comp_a_smem_m = ty * THREAD_SIZE_Y + m;
                    int comp_b_smem_n = tx * THREAD_SIZE_X + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++) {
        int store_c_gmem_m = by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + i;
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; j += 4) {
            int store_c_gmem_n = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, n);
            FLOAT4(c_ptr[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t m = atoi(argv[1]);
    size_t k = atoi(argv[2]);
    size_t n = atoi(argv[3]);

    assert( m%8 == 0); 
    assert( n%8 == 0); 
    assert( k%8 == 0); 

    size_t bytes_A = sizeof(float) * m * k;
    size_t bytes_B = sizeof(float) * k * n;
    size_t bytes_C = sizeof(float) * m * n;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * m * n * k;

    // 1. 在 CPU 生成输入矩阵 (Torch Tensor)
    for( int i = 0; i < m * k; i++ ){
        h_A[i] = i / 13;
        // h_A[i] = 1.0f;
    }

    // generate B
    for( int i = 0; i < k * n; i++ ) {
        h_B[i] = i % 13;
        // h_B[i] = 1.0f;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));

    // 5. 启动 CUDA 核函数
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
        gemm_kernel_sm_float4<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
    }
    // cudaDeviceSynchronize();

    
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    
    // 6. 复制结果回 CPU
    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            m, n, k, &alpha, 
            d_A, k, d_B, n, &beta, d_C, n
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        int row = i / n;
        int col = i % n;
        double abs_err = fabs(h_C[i] - h_C1[col * m + row]);
        double dot_length = m;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * m + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    // 9. 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);

    return 0;
}
