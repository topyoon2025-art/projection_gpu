
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "projection_gpu",
    srcs = ["projection_gpu.cu"],
    hdrs = ["projection.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "projection_cpu",
    srcs = ["projection_cpu.cc"],
    hdrs = ["projection.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status:status",
    ],
)

cc_binary(
    name = "projection_gpu_app",
    srcs = ["main.cc"],
    deps = [
        ":projection_gpu",
        ":projection_cpu",
        # ── YDF helpers
        "//yggdrasil_decision_forests/learner/random_forest:random_forest",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/model/random_forest:random_forest",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:types",
        #"//yggdrasil_decision_forests/dataset:csv_example_reader",
        # ── Abseil
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    linkopts = ["-lcudart"],
    features = ["c++17"],
)

cc_binary(
    name = "generate_csv",
    srcs = ["generate_csv.cc"],
    deps = [],
)