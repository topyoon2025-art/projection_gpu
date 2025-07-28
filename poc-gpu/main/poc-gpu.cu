#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void FillYValKernel(int32_t* yval, int label_mod, int64_t rows) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows) {
        yval[i] = static_cast<int>((i % label_mod) + 1);  // 1-based
        if (i < 10) {
            printf("d_yval[%ld] = %d\n", i, yval[i]);
        }
    }
}

void yval_mod(int label_mod, int64_t rows) {
    std::cout << "Beginning in CPU.\n" << std::endl;

    // Allocate host memory
    std::vector<int32_t> h_yval(rows);

    // Allocate device memory
    int32_t* d_yval;
    cudaMalloc((void**)&d_yval, rows * sizeof(int32_t));

    // Launch CUDA kernel
    dim3 grid_size(16); //number of blocks in a grid
    dim3 block_size(256); //number of threads in a block
    printf("Beginning in GPU\n");
    FillYValKernel<<<grid_size, block_size>>>(d_yval, label_mod, rows);
    cudaDeviceSynchronize();
    printf("Ending in GPU\n");

    // Copy result back to host
    cudaMemcpy(h_yval.data(), d_yval, rows * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_yval);

    // Print some values using CPU
    for (int64_t i = 0; i < 10; ++i) {
        std::cout << "h_yval[" << i << "] = " << h_yval[i] << std::endl;
    }

    std::cout << "Ending in CPU.\n" << std::endl;
}