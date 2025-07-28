load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "projection_gpu",
    srcs = ["projection_gpu.cu"],
    hdrs = ["projection_gpu.h"],
)

cc_binary(
    name = "projection_gpu_app",
    srcs = ["main.cc"],
    deps = [":projection_gpu"],
    linkopts = ["-lcudart"],
)