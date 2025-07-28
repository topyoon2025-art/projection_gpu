#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

#include "yggdrasil_decision_forests/learner/decision_tree/gpu.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

__global__ void FillYValKernel(int32_t* yval, int label_mod, int64_t rows) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows) {
        yval[i] = static_cast<int>((i % label_mod) + 1);  // 1-based
        if (i < 10) {
            printf("d_yval[%ld] = %d\n", i, yval[i]);
        }
    }
}
// RETURN_IF_ERROR on Cuda status.
#define RET_CUDA(x) RETURN_IF_ERROR(CudaStatus(x))

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

absl::Status CudaStatus(cudaError_t code) {
  if (code != cudaSuccess) {
    const char *error = cudaGetErrorString(code);
    return absl::InvalidArgumentError(absl::StrCat("Cuda error: ", error));
  }
  return absl::OkStatus();
}

absl::Status yval_mod(int label_mod, int64_t rows) {
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
    cudaError_t err = cudaGetLastError();
    
    return CudaStatus(err);
}