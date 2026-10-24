import json
import math
import joblib
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "model2_charging_fault_xgb.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "esp32"
    / "firmware"
    / "SparkEV_Integrated"
    / "model2_esp32_generated.h"
)


# ============================================================
# FLOAT FORMATTER
# ============================================================

def cpp_float(value):
    value = float(value)

    if math.isnan(value):
        return "NAN"

    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"

    s = format(value, ".10g")

    if "e" not in s.lower() and "." not in s:
        s += ".0"

    return s + "f"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SparkEV Model 2 -> ESP32 Converter")
print("=" * 70)

print()
print("Loading:")
print(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print()
print("Model information")
print("-" * 70)

print("Model type :", type(model).__name__)
print("Features   :", model.n_features_in_)
print("Trees      :", model.n_estimators)
print("Classes    :", model.classes_)

if hasattr(model, "base_score"):
    print("Base score :", model.base_score)


# ============================================================
# VERIFY MODEL TYPE
# ============================================================

if model.n_features_in_ != 57:
    raise ValueError(
        f"Expected 57 features, found {model.n_features_in_}"
    )

if model.n_estimators != 300:
    raise ValueError(
        f"Expected 300 trees, found {model.n_estimators}"
    )


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = list(model.get_booster().feature_names)

print()
print("Features:")
for i, name in enumerate(feature_names):
    print(f"{i:2d}: {name}")


# ============================================================
# GET XGBOOST JSON MODEL
# ============================================================

booster = model.get_booster()

json_config = json.loads(
    booster.save_config()
)

raw_model = json.loads(
    booster.get_dump(
        dump_format="json",
        with_stats=False
    )[0]
)


# ============================================================
# SAVE COMPLETE JSON MODEL TEMPORARILY
# ============================================================

temp_json = (
    PROJECT_ROOT
    / "esp32"
    / "tools"
    / "_model2_temp_dump.json"
)

booster.save_model(str(temp_json))

with open(temp_json, "r") as f:
    model_json = json.load(f)


# ============================================================
# MODEL PARAMETERS
# ============================================================

learner = model_json["learner"]

objective_name = (
    learner["objective"]["name"]
)

base_score_text = (
    learner["learner_model_param"]["base_score"]
)

base_score = float(
    base_score_text.strip("[]")
)


# ============================================================
# TREE EXTRACTION
# ============================================================

gradient_booster = learner["gradient_booster"]

model_trees = gradient_booster[
    "model"
]["trees"]


if len(model_trees) != 300:
    raise ValueError(
        f"Expected 300 trees in JSON, "
        f"found {len(model_trees)}"
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

header = []

header.append("#pragma once")
header.append("")
header.append("#include <stdint.h>")
header.append("#include <math.h>")
header.append("")
header.append("// ============================================================")
header.append("// SparkEV Model 2")
header.append("// Automatically generated from trained XGBoost classifier")
header.append("// ============================================================")
header.append("")

header.append(
    f"#define MODEL2_NUM_FEATURES {len(feature_names)}"
)

header.append(
    f"#define MODEL2_NUM_TREES {len(model_trees)}"
)

header.append(
    f"#define MODEL2_BASE_SCORE {cpp_float(base_score)}"
)

header.append("")

header.append(
    "typedef struct {"
)

header.append(
    "    const int16_t* left;"
)

header.append(
    "    const int16_t* right;"
)

header.append(
    "    const int8_t* feature;"
)

header.append(
    "    const float* threshold;"
)

header.append(
    "    const float* leaf;"
)

header.append(
    "    uint16_t nodes;"
)

header.append(
    "} Model2Tree;"
)

header.append("")


# ============================================================
# TREE ARRAYS
# ============================================================

print()
print("Generating header...")
print()

for tree_index, tree in enumerate(model_trees):

    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]
    base_weights = tree["base_weights"]

    node_count = len(left)

    features = []

    for i in range(node_count):

        if left[i] == -1 and right[i] == -1:
            features.append(-1)
        else:
            features.append(
                int(split_indices[i])
            )

    header.append(
        f"static const int16_t model2_tree_{tree_index}_left[] = {{"
        + ",".join(str(int(x)) for x in left)
        + "};"
    )

    header.append(
        f"static const int16_t model2_tree_{tree_index}_right[] = {{"
        + ",".join(str(int(x)) for x in right)
        + "};"
    )

    header.append(
        f"static const int8_t model2_tree_{tree_index}_feature[] = {{"
        + ",".join(str(int(x)) for x in features)
        + "};"
    )

    header.append(
        f"static const float model2_tree_{tree_index}_threshold[] = {{"
        + ",".join(
            cpp_float(x)
            for x in split_conditions
        )
        + "};"
    )

    header.append(
        f"static const float model2_tree_{tree_index}_leaf[] = {{"
        + ",".join(
            cpp_float(x)
            for x in base_weights
        )
        + "};"
    )

    header.append("")

    if (
        tree_index == 0
        or (tree_index + 1) % 25 == 0
        or tree_index == len(model_trees) - 1
    ):
        print(
            f"Generated tree "
            f"{tree_index + 1}/{len(model_trees)}"
        )


# ============================================================
# TREE TABLE
# ============================================================

header.append(
    "static const Model2Tree model2_trees[MODEL2_NUM_TREES] = {"
)

for tree_index, tree in enumerate(model_trees):

    node_count = len(
        tree["left_children"]
    )

    header.append(
        "    {"
        f"model2_tree_{tree_index}_left,"
        f"model2_tree_{tree_index}_right,"
        f"model2_tree_{tree_index}_feature,"
        f"model2_tree_{tree_index}_threshold,"
        f"model2_tree_{tree_index}_leaf,"
        f"{node_count}"
        "},"
    )

header.append("};")
header.append("")


# ============================================================
# SINGLE TREE EVALUATION
# ============================================================

header.append(
    "static inline float model2_eval_tree("
)
header.append(
    "    const Model2Tree& tree,"
)
header.append(
    "    const float* features"
)
header.append(
    ") {"
)

header.append(
    "    int node = 0;"
)

header.append("")

header.append(
    "    while (true) {"
)

header.append(
    "        if (tree.left[node] == -1 &&"
)

header.append(
    "            tree.right[node] == -1) {"
)

header.append(
    "            return tree.leaf[node];"
)

header.append(
    "        }"
)

header.append("")

header.append(
    "        int feature = tree.feature[node];"
)

header.append(
    "        float value = features[feature];"
)

header.append("")

header.append(
    "        if (value < tree.threshold[node]) {"
)

header.append(
    "            node = tree.left[node];"
)

header.append(
    "        } else {"
)

header.append(
    "            node = tree.right[node];"
)

header.append(
    "        }"
)

header.append(
    "    }"
)

header.append(
    "}"
)

header.append("")


# ============================================================
# MODEL 2 PREDICTION
# ============================================================

header.append(
    "// Returns raw XGBoost probability/logit output"
)

header.append(
    "static inline float model2_predict_raw("
)

header.append(
    "    const float* features"
)

header.append(
    ") {"
)

header.append(
    "    float sum = MODEL2_BASE_SCORE;"
)

header.append("")

header.append(
    "    for (int i = 0;"
)

header.append(
    "         i < MODEL2_NUM_TREES;"
)

header.append(
    "         i++) {"
)

header.append(
    "        sum += model2_eval_tree("
)

header.append(
    "            model2_trees[i],"
)

header.append(
    "            features"
)

header.append(
    "        );"
)

header.append(
    "    }"
)

header.append("")

header.append(
    "    return sum;"
)

header.append(
    "}"
)

header.append("")


# ============================================================
# SIGMOID
# ============================================================

header.append(
    "static inline float model2_sigmoid(float x) {"
)

header.append(
    "    if (x >= 0.0f) {"
)

header.append(
    "        float z = expf(-x);"
)

header.append(
    "        return 1.0f / (1.0f + z);"
)

header.append(
    "    } else {"
)

header.append(
    "        float z = expf(x);"
)

header.append(
    "        return z / (1.0f + z);"
)

header.append(
    "    }"
)

header.append(
    "}"
)

header.append("")


# ============================================================
# FINAL PREDICTION
# ============================================================

header.append(
    "// Returns probability of class 1"
)

header.append(
    "static inline float model2_predict("
)

header.append(
    "    const float* features"
)

header.append(
    ") {"
)

header.append(
    "    float raw = model2_predict_raw(features);"
)

header.append(
    "    return model2_sigmoid(raw);"
)

header.append(
    "}"
)

header.append("")


# ============================================================
# CLASSIFICATION
# ============================================================

header.append(
    "static inline int model2_class("
)

header.append(
    "    const float* features"
)

header.append(
    ") {"
)

header.append(
    "    float probability = model2_predict(features);"
)

header.append(
    "    return probability >= 0.5f ? 1 : 0;"
)

header.append(
    "}"
)

header.append("")


# ============================================================
# WRITE HEADER
# ============================================================

with open(
    OUTPUT_PATH,
    "w"
) as f:

    f.write(
        "\n".join(header)
    )


# ============================================================
# CLEAN TEMP FILE
# ============================================================

try:
    temp_json.unlink()
except Exception:
    pass


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_PATH)

print()
print("Model features :", len(feature_names))
print("Model trees    :", len(model_trees))
print("Base score     :", base_score)

print()
print("Float formatting:")
print("  1.0  -> 1.0f")
print("  3.0  -> 3.0f")
print(" 17.0  -> 17.0f")
print("  0.5  -> 0.5f")

print()
print("The generated header contains")
print("the complete trained XGBoost")
print("Model 2 tree ensemble.")

print()
print("Arduino header location:")
print(OUTPUT_PATH)

print("=" * 70)