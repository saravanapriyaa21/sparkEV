import json
import os
import math


# ============================================================
# SparkEV Model 1
# XGBoost JSON -> ESP32 C++ converter
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VECTORS_DIR = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors")
FIRMWARE_DIR = os.path.join(
    PROJECT_ROOT, "esp32", "firmware", "SparkEV_Integrated"
)

MODEL_PATH = os.path.join(
    MODELS_DIR, "model1_battery_stress_xgb.json"
)

# Generate directly into the Arduino sketch folder
OUTPUT_PATH = os.path.join(
    FIRMWARE_DIR,
    "model1_esp32_generated.h"
)


# ============================================================
# C++ FLOAT FORMATTER
# ============================================================
#
# IMPORTANT:
#
# Python's:
#
#     f"{1.0:.10g}"
#
# produces:
#
#     "1"
#
# Adding "f" would create:
#
#     1f
#
# which is INVALID C++.
#
# This function makes sure we generate:
#
#     1.0f
#     3.0f
#     17.0f
#     0.5f
#     1.2345f
#
# Scientific notation is also handled correctly.
# ============================================================

def cpp_float(value):

    value = float(value)

    # NaN
    if math.isnan(value):
        return "NAN"

    # Infinity
    if math.isinf(value):

        if value > 0:
            return "INFINITY"

        return "-INFINITY"

    # Keep enough precision for the model
    s = format(value, ".10g")

    # Scientific notation
    #
    # Example:
    # 0.00003113634
    #
    # becomes:
    # 3.113634e-05
    #
    # This is already valid C++ when followed by f.
    if "e" in s.lower():
        return s + "f"

    # Integer-valued floats
    #
    # 1 -> 1.0f
    # 3 -> 3.0f
    # 17 -> 17.0f
    if "." not in s:
        s += ".0"

    return s + "f"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SparkEV Model 1 -> ESP32 Converter")
print("=" * 70)

print()
print("Loading:")
print(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nModel file was not found:\n"
        + MODEL_PATH
    )


with open(
    MODEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    model = json.load(f)


learner = model["learner"]

feature_names = learner["feature_names"]

booster = learner[
    "gradient_booster"
]["model"]

trees = booster["trees"]

learner_params = learner[
    "learner_model_param"
]

base_score_text = learner_params[
    "base_score"
]


# ============================================================
# READ BASE SCORE
# ============================================================
#
# XGBoost 3.x commonly stores:
#
#     "[1.1836274E1]"
#
# Convert that safely to float.
# ============================================================

base_score_clean = (
    base_score_text
    .strip()
    .strip("[")
    .strip("]")
)

# If multiple values are ever present,
# take the first one.
base_score_clean = (
    base_score_clean
    .split(",")[0]
    .strip()
)

base_score = float(
    base_score_clean
)


# ============================================================
# MODEL INFORMATION
# ============================================================

num_features = len(
    feature_names
)

num_trees = len(
    trees
)


print()
print("Model information")
print("-" * 70)

print(
    "Features :",
    num_features
)

print(
    "Trees    :",
    num_trees
)

print(
    "Base     :",
    base_score
)

print(
    "Version  :",
    model.get("version")
)


print()
print("Features:")

for i, name in enumerate(
    feature_names
):

    print(
        f"{i:2d}: {name}"
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

if num_features != 24:

    print()
    print(
        "WARNING: Expected 24 features, "
        f"but found {num_features}"
    )


if num_trees != 300:

    print()
    print(
        "WARNING: Expected 300 trees, "
        f"but found {num_trees}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

output_directory = os.path.dirname(
    OUTPUT_PATH
)

os.makedirs(
    output_directory,
    exist_ok=True
)


# ============================================================
# GENERATE HEADER
# ============================================================

print()
print("Generating header...")
print()


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as out:


    # ========================================================
    # HEADER
    # ========================================================

    out.write(
        "// ============================================================\n"
    )

    out.write(
        "// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY\n"
    )

    out.write(
        "// SparkEV Model 1 XGBoost model for ESP32\n"
    )

    out.write(
        "// Generated from model1_battery_stress_xgb.json\n"
    )

    out.write(
        "// ============================================================\n\n"
    )


    out.write(
        "#pragma once\n\n"
    )

    out.write(
        "#include <stdint.h>\n"
    )

    out.write(
        "#include <math.h>\n\n"
    )


    # ========================================================
    # MODEL CONSTANTS
    # ========================================================

    out.write(
        f"#define MODEL1_NUM_FEATURES "
        f"{num_features}\n"
    )

    out.write(
        f"#define MODEL1_NUM_TREES "
        f"{num_trees}\n"
    )

    out.write(
        f"#define MODEL1_BASE_SCORE "
        f"{cpp_float(base_score)}\n\n"
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    out.write(
        "static const char* "
        "MODEL1_FEATURE_NAMES[] = {\n"
    )


    for name in feature_names:

        escaped = (
            name
            .replace(
                "\\",
                "\\\\"
            )
            .replace(
                '"',
                '\\"'
            )
        )

        out.write(
            f'    "{escaped}",\n'
        )


    out.write(
        "};\n\n"
    )


    # ========================================================
    # TREE STRUCTURE
    # ========================================================

    out.write(
        "typedef struct {\n"
    )

    out.write(
        "    const int16_t* left;\n"
    )

    out.write(
        "    const int16_t* right;\n"
    )

    out.write(
        "    const int8_t* feature;\n"
    )

    out.write(
        "    const float* threshold;\n"
    )

    out.write(
        "    const float* leaf;\n"
    )

    out.write(
        "    uint16_t nodes;\n"
    )

    out.write(
        "} Model1Tree;\n\n"
    )


    # ========================================================
    # GENERATE EVERY TREE
    # ========================================================

    for tree_index, tree in enumerate(
        trees
    ):

        base_weights = tree[
            "base_weights"
        ]

        left_children = tree[
            "left_children"
        ]

        right_children = tree[
            "right_children"
        ]

        split_conditions = tree[
            "split_conditions"
        ]

        split_indices = tree[
            "split_indices"
        ]


        node_count = len(
            base_weights
        )


        # ----------------------------------------------------
        # Tree comment
        # ----------------------------------------------------

        out.write(
            f"// Tree {tree_index} "
            f"- {node_count} nodes\n"
        )


        # ====================================================
        # LEFT CHILDREN
        # ====================================================

        out.write(
            f"static const int16_t "
            f"tree_{tree_index}_left[] = {{"
        )


        out.write(
            ",".join(
                str(int(x))
                for x in left_children
            )
        )


        out.write(
            "};\n"
        )


        # ====================================================
        # RIGHT CHILDREN
        # ====================================================

        out.write(
            f"static const int16_t "
            f"tree_{tree_index}_right[] = {{"
        )


        out.write(
            ",".join(
                str(int(x))
                for x in right_children
            )
        )


        out.write(
            "};\n"
        )


        # ====================================================
        # FEATURE INDICES
        # ====================================================
        #
        # Internal nodes:
        #     actual feature index
        #
        # Leaf nodes:
        #     -1
        #
        # Feature value is never accessed for leaves.
        # ====================================================

        out.write(
            f"static const int8_t "
            f"tree_{tree_index}_feature[] = {{"
        )


        feature_values = []


        for i in range(
            node_count
        ):

            # Leaf node
            if left_children[i] == -1:

                feature_values.append(
                    -1
                )

            # Internal node
            else:

                feature_values.append(
                    int(
                        split_indices[i]
                    )
                )


        out.write(
            ",".join(
                str(x)
                for x in feature_values
            )
        )


        out.write(
            "};\n"
        )


        # ====================================================
        # THRESHOLDS
        # ====================================================
        #
        # IMPORTANT:
        #
        # Use cpp_float()
        #
        # NOT:
        #
        #     f"{float(x):.10g}f"
        #
        # because that can generate:
        #
        #     1f
        #
        # instead of:
        #
        #     1.0f
        # ====================================================

        out.write(
            f"static const float "
            f"tree_{tree_index}_threshold[] = {{"
        )


        out.write(
            ",".join(
                cpp_float(x)
                for x in split_conditions
            )
        )


        out.write(
            "};\n"
        )


        # ====================================================
        # LEAF VALUES
        # ====================================================
        #
        # XGBoost stores base_weights for both:
        #
        #     internal nodes
        #     leaf nodes
        #
        # Only leaf-node values are used during prediction.
        #
        # Internal nodes therefore get:
        #
        #     0.0f
        # ====================================================

        out.write(
            f"static const float "
            f"tree_{tree_index}_leaf[] = {{"
        )


        leaf_values = []


        for i in range(
            node_count
        ):

            # Leaf
            if left_children[i] == -1:

                leaf_values.append(
                    cpp_float(
                        base_weights[i]
                    )
                )

            # Internal node
            else:

                leaf_values.append(
                    "0.0f"
                )


        out.write(
            ",".join(
                leaf_values
            )
        )


        out.write(
            "};\n\n"
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            tree_index == 0
            or
            (tree_index + 1) % 25 == 0
            or
            tree_index == num_trees - 1
        ):

            print(
                f"Generated tree "
                f"{tree_index + 1}/{num_trees}"
            )


    # ========================================================
    # TREE TABLE
    # ========================================================

    out.write(
        "// ============================================================\n"
    )

    out.write(
        "// Tree table\n"
    )

    out.write(
        "// ============================================================\n"
    )


    out.write(
        "static const Model1Tree "
        "MODEL1_TREES[] = {\n"
    )


    for tree_index, tree in enumerate(
        trees
    ):

        node_count = len(
            tree["base_weights"]
        )


        out.write(
            "    {\n"
        )

        out.write(
            f"        tree_{tree_index}_left,\n"
        )

        out.write(
            f"        tree_{tree_index}_right,\n"
        )

        out.write(
            f"        tree_{tree_index}_feature,\n"
        )

        out.write(
            f"        tree_{tree_index}_threshold,\n"
        )

        out.write(
            f"        tree_{tree_index}_leaf,\n"
        )

        out.write(
            f"        {node_count}\n"
        )

        out.write(
            "    },\n"
        )


    out.write(
        "};\n\n"
    )


    # ========================================================
    # TREE EVALUATION FUNCTION
    # ========================================================

    out.write(
        """
// ============================================================
// Evaluate one XGBoost tree
// ============================================================

static inline float model1_eval_tree(
    const Model1Tree& tree,
    const float* features
) {

    int node = 0;

    while (true) {

        int left =
            tree.left[node];

        // ----------------------------------------------------
        // Leaf node
        // ----------------------------------------------------

        if (left == -1) {

            return tree.leaf[node];
        }


        // ----------------------------------------------------
        // Internal node
        // ----------------------------------------------------

        int feature_index =
            tree.feature[node];


        float value =
            features[feature_index];


        float threshold =
            tree.threshold[node];


        // ----------------------------------------------------
        // XGBoost split:
        //
        // if feature < threshold
        //     go left
        // else
        //     go right
        // ----------------------------------------------------

        if (value < threshold) {

            node =
                tree.left[node];

        }
        else {

            node =
                tree.right[node];
        }
    }
}


// ============================================================
// Model 1 prediction
// ============================================================

static inline float model1_predict(
    const float* features
) {

    float prediction =
        MODEL1_BASE_SCORE;


    for (
        int tree_index = 0;
        tree_index < MODEL1_NUM_TREES;
        tree_index++
    ) {

        prediction +=
            model1_eval_tree(
                MODEL1_TREES[tree_index],
                features
            );
    }


    return prediction;
}

"""
    )


# ============================================================
# VERIFY GENERATED FILE
# ============================================================

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_PATH)

print()
print("Model features :", num_features)
print("Model trees    :", num_trees)
print("Base score     :", base_score)

print()
print("Float formatting:")
print("  1.0  ->", cpp_float(1.0))
print("  3.0  ->", cpp_float(3.0))
print(" 17.0  ->", cpp_float(17.0))
print("  0.5  ->", cpp_float(0.5))

print()
print("The generated header contains")
print("the complete trained XGBoost tree ensemble.")

print()
print("Arduino header location:")
print(OUTPUT_PATH)

print()
print("=" * 70)