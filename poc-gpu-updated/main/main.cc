#include "yval_mod.h"
#include <stdio.h>
#include <cstdint>
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/strings/str_cat.h"

int main() {
    int label_mod = 2;
    int64_t rows = 4096;
    absl::Status status = yval_mod(label_mod, rows);
    if (!status.ok()) {
        std::cerr << "Kernel launch failed: " << status.message() << "\n";
        return 1;
    }
    std::cout << "Kernel launched successfully!\n";
    return 0;
}