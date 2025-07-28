// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>


__global__ void Flat_Projection_Kernel(
    const float* dataset, 
    const int* row_indices,
    const int* column_indices,
    float* projected,
    int num_features,
    int num_selected_rows,
    int num_selected_columns) 
{   
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_selected_rows) {
        float sum = 0.0f;
        for (int j = 0; j < num_selected_columns; j++) {
            int dataset_index = row_indices[i] * num_features + column_indices[j];
            sum += dataset[dataset_index];
        }
        projected[i] = sum;
        printf("Running GPU[%d]: %.3f\n", i, projected[i]);
    }
}


std::vector<float> cudaFlatProjection(const std::vector<float>& dataset, const std::vector<int>& row_indices, const std::vector<int>& column_indices, int num_features) {
    
    int num_selected_rows = row_indices.size();
    int num_selected_columns = column_indices.size();
    int total_dataset_size = dataset.size();
  
    //Allocate device memory
    float *d_dataset, *d_projected;
    int *d_row_indices, *d_col_indices;
    cudaMalloc(&d_dataset, total_dataset_size * sizeof(float));
    cudaMalloc(&d_row_indices, num_selected_rows * sizeof(int));
    cudaMalloc(&d_col_indices, num_selected_columns * sizeof(int));
    cudaMalloc(&d_projected, num_selected_rows * sizeof(float));

    //Copy data to device
    cudaMemcpy(d_dataset, dataset.data(), total_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices.data(), num_selected_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, column_indices.data(), num_selected_columns * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    dim3 grid_size(16); //number of blocks in a grid
    dim3 block_size(256); //number of threads in a bloc
    std::cout << "Beginning in GPU." << std::endl;
    Flat_Projection_Kernel<<<grid_size, block_size>>>(d_dataset, d_row_indices, d_col_indices, d_projected,
        num_features, num_selected_rows, num_selected_columns);

    cudaDeviceSynchronize();  // Ensure kernel finishes

    std::vector<float> projected_output(num_selected_rows);//Vectorize projected_output

    cudaMemcpy(projected_output.data(), d_projected, num_selected_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dataset);
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_projected);

    return projected_output;
}

