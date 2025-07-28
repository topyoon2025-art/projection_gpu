
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "projection_gpu",
    srcs = ["projection_gpu.cu"],
    hdrs = ["projection_gpu.h"],
    deps = [],
)

cc_binary(
    name = "projection_gpu_app",
    srcs = ["main.cc"],
    deps = [
        ":projection_gpu",
        # ── YDF helpers
        "//yggdrasil_decision_forests/learner/random_forest:random_forest",
        # "//yggdrasil_decision_forests/learner:learner_library",
        # "//yggdrasil_decision_forests/model/random_forest:random_forest",
        # "//yggdrasil_decision_forests/model:model_library",
        # "//yggdrasil_decision_forests/dataset:data_spec",
        # "//yggdrasil_decision_forests/dataset:csv_example_reader",
        # "//yggdrasil_decision_forests/utils:logging",
        # "//yggdrasil_decision_forests/utils:distribution",
        # ── Abseil
        "@com_google_absl//absl/log",
        # "@com_google_absl//absl/status",
        # "@com_google_absl//absl/strings",
    ],
    linkopts = ["-lcudart"],
    features = ["c++17"],
)
