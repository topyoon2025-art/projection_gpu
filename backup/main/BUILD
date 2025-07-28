load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "poc-gpu",
    srcs = ["poc-gpu.cu"],
    hdrs = ["yval_mod.h"],
    deps = ["@abseil-cpp//absl/base", 
            "@abseil-cpp//absl/log:check",
            "@abseil-cpp//absl/log:log",
            "@abseil-cpp//absl/status:statusor",
            "//yggdrasil_decision_forests/learner/random_forest:random_forest",
            "//yggdrasil_decision_forests/learner:learner_library",
            "//yggdrasil_decision_forests/model/random_forest:random_forest",
            "//yggdrasil_decision_forests/model:model_library",
            "//yggdrasil_decision_forests/dataset:data_spec",
            "//yggdrasil_decision_forests/dataset:csv_example_reader",
            "//yggdrasil_decision_forests/utils:logging",
            "//yggdrasil_decision_forests/utils:distribution",
            ],
)

cc_binary(
    name = "poc-gpu-app",
    srcs = ["main.cc"],
    deps = [":poc-gpu"],
    linkopts = ["-lcudart"],
)