load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "poc-gpu",
    srcs = ["poc-gpu.cu"],
    hdrs = ["yval_mod.h"],
    deps = ["@abseil-cpp//absl/base", 
            "@abseil-cpp//absl/log:check",
            "@abseil-cpp//absl/log:log",
            "@abseil-cpp//absl/status:statusor",
            ],
)

cc_binary(
    name = "poc-gpu-app",
    srcs = ["main.cc"],
    deps = [":poc-gpu"],
    linkopts = ["-lcudart"],
)