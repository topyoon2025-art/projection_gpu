load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "poc-gpu",
    srcs = ["poc-gpu.cu"],
    hdrs = ["yval_mod.h"],
)

cc_binary(
    name = "poc-gpu-app",
    srcs = ["main.cc"],
    deps = [":poc-gpu"],
    linkopts = ["-lcudart"],
)