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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

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
        //printf("Running GPU[%ld]: %.3f\n", i, projected[i]);
    }
}

absl::Status CudaStatus(cudaError_t code) {
  if (code != cudaSuccess) {
    const char *error = cudaGetErrorString(code);
    return absl::InvalidArgumentError(absl::StrCat("Cuda error: ", error));
  }
  return absl::OkStatus();
}
#define RETURN_IF_ERROR(expr) do {   \                           
    absl::Status _status = (expr);   \ 
    if (!_status.ok()) return _status; \
  } while (0)

#define RET_CUDA(x) RETURN_IF_ERROR(CudaStatus(x))

// Copies a vector from host to device.
template <typename T>
absl::Status EasyCudaCopyH2D(absl::Span<const T> src, T *dst) {
  return CudaStatus(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
}

// Copies a vector from device to host.
template <typename T>
absl::Status EasyCudaCopyD2H(T *src, size_t n, absl::Span<T> dst) {
  return CudaStatus(
      cudaMemcpy(dst.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// Allocates and copy data from host to device.
template <typename T>
absl::Status EasyCudaAllocAndCopy(absl::Span<const T> src, T **dst) {
  RET_CUDA(cudaMalloc((void **)dst, src.size() * sizeof(T)));
  return EasyCudaCopyH2D<T>(src, *dst);
}

absl::Status CheckHasGPU(bool print_info) {
  static absl::Status status = [&]() -> absl::Status {
    int driver_version = 0;
    RET_CUDA(cudaDriverGetVersion(&driver_version));
    if (driver_version == 0) {
      return absl::InvalidArgumentError("No matching cuda driver found");
    }
    cudaDeviceProp prop;
    RET_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (print_info) {
      LOG(INFO) << "Using CUDA device: " << prop.name
                << " (driver:" << driver_version << ")";
    }
    return absl::OkStatus();
  }();
  return status;
}

absl::Status cudaFlatProjection(absl::Span<const float> dataset,
                                absl::Span<const int> row_indices,
                                absl::Span<const int> column_indices,
                                int num_features,
                                absl::Span<float> projected) {
    
    int num_selected_rows = row_indices.size();
    int num_selected_columns = column_indices.size();
    int total_dataset_size = dataset.size();
  
    //Allocate device memory
    float *d_dataset = nullptr;
    float *d_projected = nullptr;
    int *d_row_indices = nullptr;
    int *d_col_indices = nullptr;

    RET_CUDA(cudaMalloc((void **)&d_dataset, total_dataset_size * sizeof(float)));
    RET_CUDA(cudaMalloc((void **)&d_row_indices, num_selected_rows * sizeof(int)));
    RET_CUDA(cudaMalloc((void **)&d_col_indices, num_selected_columns * sizeof(int)));
    RET_CUDA(cudaMalloc((void **)&d_projected, num_selected_rows * sizeof(float)));

    //Copy data to device
    RET_CUDA(cudaMemcpy(d_dataset, dataset.data(), total_dataset_size * sizeof(float), cudaMemcpyHostToDevice));
    RET_CUDA(cudaMemcpy(d_row_indices, row_indices.data(), num_selected_rows * sizeof(int), cudaMemcpyHostToDevice));
    RET_CUDA(cudaMemcpy(d_col_indices, column_indices.data(), num_selected_columns * sizeof(int), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    dim3 grid_size(16); //number of blocks in a grid
    dim3 block_size(256); //number of threads in a bloc

    auto startB = std::chrono::high_resolution_clock::now();
    Flat_Projection_Kernel<<<grid_size, block_size>>>(d_dataset, d_row_indices, d_col_indices, d_projected,
        num_features, num_selected_rows, num_selected_columns);
    auto endB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationB = endB - startB;
    std::cout << "GPU total elapsed time: " << durationB.count() << " ms\n";
    
    RET_CUDA(cudaPeekAtLastError());
    RET_CUDA(cudaDeviceSynchronize());  // Ensure kernel finishes

    //std::vector<float> projected_output(num_selected_rows);//Vectorize projected_output

    RET_CUDA(cudaMemcpy(projected.data(), d_projected, num_selected_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    RET_CUDA(cudaFree(d_dataset));
    RET_CUDA(cudaFree(d_row_indices));
    RET_CUDA(cudaFree(d_col_indices));
    RET_CUDA(cudaFree(d_projected));

    return absl::OkStatus();
}

