#include "launch.h"

#include <stdio.h>

__global__ void increment_value (int *a, int N) {
    int i = threadIdx.x;
    if (i < N) {
        a[i] = a[i] + 1;
        printf("Hello from CUDA thread %d! Value: %d\n", i, a[i]);
    }

}


void launch () {
    int N = 10;

    //from host
    int h_a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    //Allocate arrary in Device memory
    int *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(int));

    //Copy memory from Host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Block and Grid dimensions
    dim3 grid_size(1); //number of blocks in a grid
    dim3 block_size(N); //number of threads in a block

    // Launch Kernel
    increment_value<<<grid_size, block_size>>>(d_a, N);
    cudaDeviceSynchronize();

    // //Free device memory
    // cudaFree(d_a);
}