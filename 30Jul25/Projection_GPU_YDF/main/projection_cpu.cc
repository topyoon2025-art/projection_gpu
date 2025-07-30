#include<vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

absl::Status FlatProjection(
    absl::Span<const float> dataset,
    absl::Span<const int> row_indices,
    absl::Span<const int> column_indices,
    int num_features,
    absl::Span<float> projected_cpu){
    for (long unsigned int i = 0; i < row_indices.size(); i++) {
        for (long unsigned int j = 0; j < column_indices.size(); j++) {
            projected_cpu[i] += dataset[row_indices[i] * num_features + column_indices[j]];
        }
    }
    return absl::OkStatus();
}