import numpy as np



def block_matrix_multiply(A, B, block_size):
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))  # 初始化结果矩阵 C
    for i in range(0, m, block_size):
        for j in range(0, p, block_size):
            for k in range(0, n, block_size):
                A_block = A[i:i+block_size, k:k+block_size]
                B_block = B[k:k+block_size, j:j+block_size]
                # print(f"A_block: \n{A_block}\n, B_block: \n{B_block}\n")
                C[i:i+block_size, j:j+block_size] += np.dot(A_block, B_block)
    return C

if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

    B = np.array([[17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32]])
    block_size = 2
    C = block_matrix_multiply(A, B, block_size)
    print(C)
    print(np.dot(A, B))


