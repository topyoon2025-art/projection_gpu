#include <cstdint>
#include <vector>
std::vector<float> cudaFlatProjection(const std::vector<float>& dataset, const std::vector<int>& row_indices, const std::vector<int>& column_indices, int num_column);