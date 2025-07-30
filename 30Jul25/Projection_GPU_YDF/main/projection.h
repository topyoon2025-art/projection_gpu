#include <vector>
#include "absl/status/status.h"

absl::Status CheckHasGPU(bool print_info);
absl::Status FlatProjection(absl::Span<const float> dataset,
                            absl::Span<const int> row_indices,
                            absl::Span<const int> column_indices,
                            int num_features,
                            absl::Span<float> projected_cpu);
absl::Status cudaFlatProjection(absl::Span<const float> dataset,
                                absl::Span<const int> row_indices,
                                absl::Span<const int> column_indices,
                                int num_features,
                                absl::Span<float> projected);