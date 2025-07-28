#include "yval_mod.h"
#include <stdio.h>
#include <cstdint>

int main() {
    int label_mod = 2;
    int64_t rows = 4096;
    yval_mod(label_mod, rows);
    return 0;
}