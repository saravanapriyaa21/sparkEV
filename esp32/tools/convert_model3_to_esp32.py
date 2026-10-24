import pickle
import json
import os
import math


# ============================================================
# CONFIGURATION
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

MODEL_PATH = os.path.join(MODELS_DIR, "model3_battery_degradation_xgb.pkl")

OUTPUT_DIR = FIRMWARE_DIR

OUTPUT_HEADER = os.path.join(
    OUTPUT_DIR,
    "model3_esp32_generated.h"
)


# ============================================================
# MODEL CONSTANTS
# ============================================================

NUMERICAL_FEATURE_COUNT = 122
EXPECTED_FEATURE_COUNT = 199

EXPECTED_TREES = 300
NUM_OUTPUTS = 3
TREES_PER_OUTPUT = EXPECTED_TREES // NUM_OUTPUTS


# ============================================================
# C++ FLOAT FORMATTER
# ============================================================

def cpp_float(value):

    value = float(value)

    if math.isnan(value):
        return "NAN"

    if math.isinf(value):

        if value > 0:
            return "INFINITY"

        return "-INFINITY"

    s = format(
        value,
        ".10g"
    )

    if (
        "e" not in s.lower()
        and "." not in s
    ):

        s += ".0"

    return s + "f"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SparkEV Model 3 ESP32 Converter")
print("=" * 70)

print()
print("Loading Model 3...")

with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)


booster = model.get_booster()

feature_names = (
    booster.feature_names
)

if feature_names is None:

    raise RuntimeError(
        "XGBoost booster does not contain feature names."
    )


feature_to_index = {
    name: i
    for i, name in enumerate(feature_names)
}


# ============================================================
# BASIC VALIDATION
# ============================================================

num_features = len(
    feature_names
)

print(
    "Features:",
    num_features
)

if num_features != EXPECTED_FEATURE_COUNT:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"found {num_features}"
    )


if NUMERICAL_FEATURE_COUNT >= num_features:

    raise RuntimeError(
        "Invalid numerical feature boundary."
    )


print(
    "Numerical features:",
    NUMERICAL_FEATURE_COUNT
)

print(
    "Integer features:",
    num_features - NUMERICAL_FEATURE_COUNT
)

print(
    "Total outputs:",
    NUM_OUTPUTS
)

print(
    "Total trees:",
    EXPECTED_TREES
)

print(
    "Trees per output:",
    TREES_PER_OUTPUT
)


# ============================================================
# VALIDATE FEATURE TYPES
# ============================================================

feature_types = (
    booster.feature_types
)

if feature_types is not None:

    float_count = sum(
        t == "float"
        for t in feature_types
    )

    integer_count = sum(
        t == "i"
        for t in feature_types
    )

    print()
    print(
        "Booster feature types:"
    )

    print(
        "  float  :",
        float_count
    )

    print(
        "  integer:",
        integer_count
    )

    if float_count != NUMERICAL_FEATURE_COUNT:

        raise RuntimeError(
            f"Expected {NUMERICAL_FEATURE_COUNT} "
            f"float features, found {float_count}"
        )

    if integer_count != (
        EXPECTED_FEATURE_COUNT
        - NUMERICAL_FEATURE_COUNT
    ):

        raise RuntimeError(
            "Unexpected integer feature count."
        )


# ============================================================
# LOAD JSON TREE DUMP
# ============================================================

print()
print(
    "Loading XGBoost tree dump..."
)

dump = booster.get_dump(
    dump_format="json",
    with_stats=False
)

num_trees = len(
    dump
)

print(
    "Trees:",
    num_trees
)

if num_trees != EXPECTED_TREES:

    raise RuntimeError(
        f"Expected {EXPECTED_TREES} trees, "
        f"found {num_trees}"
    )


# ============================================================
# STORAGE
# ============================================================

nodes = []

tree_roots = []

leaf_node_count = 0
numerical_node_count = 0
integer_threshold_node_count = 0


# ============================================================
# PARSE ONE XGBOOST NODE
# ============================================================

def parse_node(node):

    global leaf_node_count
    global numerical_node_count
    global integer_threshold_node_count

    # --------------------------------------------------------
    # Reserve global node index
    # --------------------------------------------------------

    index = len(
        nodes
    )

    nodes.append(
        None
    )


    # ========================================================
    # LEAF NODE
    # ========================================================

    if "leaf" in node:

        leaf_node_count += 1

        nodes[index] = {

            "feature": -1,

            "threshold": 0.0,

            "yes": -1,

            "no": -1,

            "missing": -1,

            "leaf": float(
                node["leaf"]
            ),

            # All tree splits are evaluated using
            # normal XGBoost threshold routing.
            "categorical": False,

        }

        return index


    # ========================================================
    # SPLIT NODE
    # ========================================================

    if "split" not in node:

        raise RuntimeError(
            "Unexpected XGBoost node without "
            "'split' or 'leaf':\n"
            + str(node)
        )


    feature_name = (
        node["split"]
    )


    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    if feature_name not in feature_to_index:

        raise RuntimeError(
            f"Unknown feature '{feature_name}'"
        )


    feature_index = (
        feature_to_index[
            feature_name
        ]
    )


    # ========================================================
    # FEATURE TYPE
    # ========================================================

    #
    # Model 3 contains:
    #
    #   0   - 121 -> float features
    #   122 - 198 -> integer car_* features
    #
    # The integer car_* features are represented in the
    # native tree using normal threshold splits.
    #
    # For example:
    #
    #   car_DE09 < 1.0  -> YES
    #   car_DE09 >= 1.0 -> NO
    #
    # Therefore the ESP32 evaluator must use the same
    # numerical threshold routing for ALL features.
    #
    # --------------------------------------------------------

    if feature_index >= NUMERICAL_FEATURE_COUNT:

        integer_threshold_node_count += 1

        #
        # The textual JSON dump does not provide the normal
        # split_condition for these integer car_* nodes.
        #
        # The native representation and verified evaluator
        # show that these binary integer features use 1.0
        # as the threshold.
        #

        threshold = 1.0

    else:

        numerical_node_count += 1

        if "split_condition" not in node:

            raise RuntimeError(
                "Numerical split node is missing "
                "'split_condition':\n"
                + str(node)
            )

        threshold = float(
            node["split_condition"]
        )


    # ========================================================
    # CHILD IDS
    # ========================================================

    if "yes" not in node:

        raise RuntimeError(
            f"Split node missing 'yes': {node}"
        )


    if "no" not in node:

        raise RuntimeError(
            f"Split node missing 'no': {node}"
        )


    yes_id = int(
        node["yes"]
    )

    no_id = int(
        node["no"]
    )


    # ========================================================
    # MISSING CHILD
    # ========================================================

    if "missing" in node:

        missing_id = int(
            node["missing"]
        )

    else:

        missing_id = no_id


    # ========================================================
    # CHILDREN
    # ========================================================

    if "children" not in node:

        raise RuntimeError(
            f"Split node has no children: {node}"
        )


    children = {
        int(child["nodeid"]): child
        for child in node["children"]
    }


    if yes_id not in children:

        raise RuntimeError(
            f"YES child {yes_id} "
            f"not found for feature "
            f"'{feature_name}'"
        )


    if no_id not in children:

        raise RuntimeError(
            f"NO child {no_id} "
            f"not found for feature "
            f"'{feature_name}'"
        )


    if missing_id not in children:

        raise RuntimeError(
            f"MISSING child {missing_id} "
            f"not found for feature "
            f"'{feature_name}'"
        )


    # ========================================================
    # RECURSIVELY PARSE CHILDREN
    # ========================================================

    yes_index = parse_node(
        children[yes_id]
    )

    no_index = parse_node(
        children[no_id]
    )


    # --------------------------------------------------------
    # Missing branch
    # --------------------------------------------------------

    if missing_id == yes_id:

        missing_index = yes_index

    elif missing_id == no_id:

        missing_index = no_index

    else:

        missing_index = parse_node(
            children[missing_id]
        )


    # ========================================================
    # STORE NODE
    # ========================================================

    nodes[index] = {

        "feature": feature_index,

        "threshold": threshold,

        "yes": yes_index,

        "no": no_index,

        "missing": missing_index,

        "leaf": 0.0,

        #
        # IMPORTANT:
        #
        # Do NOT mark car_* features as categorical.
        #
        # Their verified routing is:
        #
        # value < threshold -> YES
        # value >= threshold -> NO
        #
        "categorical": False,

    }


    return index


# ============================================================
# PARSE ALL TREES
# ============================================================

print()
print(
    "Parsing XGBoost trees..."
)

print()

for tree_id, tree_json in enumerate(
    dump
):

    root = json.loads(
        tree_json
    )

    root_index = parse_node(
        root
    )

    tree_roots.append(
        root_index
    )

    if (
        tree_id + 1
    ) % 25 == 0:

        print(
            f"Parsed "
            f"{tree_id + 1}/"
            f"{num_trees} trees..."
        )


# ============================================================
# TREE VALIDATION
# ============================================================

if len(tree_roots) != EXPECTED_TREES:

    raise RuntimeError(
        "Tree root count mismatch."
    )


if len(nodes) == 0:

    raise RuntimeError(
        "No nodes were generated."
    )


# ============================================================
# TREE TARGET ORDER
# ============================================================

#
# Model 3 uses interleaved target order:
#
# Tree 0 -> CAP
# Tree 1 -> CAP_CORR
# Tree 2 -> SOH
# Tree 3 -> CAP
# Tree 4 -> CAP_CORR
# Tree 5 -> SOH
# ...
#
# target = tree_id % 3
#
# ============================================================

tree_targets = [
    tree_id % NUM_OUTPUTS
    for tree_id in range(
        EXPECTED_TREES
    )
]


# ============================================================
# VALIDATE AGAINST XGBOOST TREE DATAFRAME
# ============================================================

print()
print("=" * 70)
print("VALIDATING TREE TARGET ORDER")
print("=" * 70)

tree_dataframe = (
    booster.trees_to_dataframe()
)

tree_target_map = (
    tree_dataframe
    .groupby("Tree")["Target"]
    .first()
)


target_mismatches = []

for tree_id in range(
    EXPECTED_TREES
):

    actual_target = int(
        tree_target_map.loc[
            tree_id
        ]
    )

    expected_target = (
        tree_id % NUM_OUTPUTS
    )

    if actual_target != expected_target:

        target_mismatches.append(
            (
                tree_id,
                actual_target,
                expected_target
            )
        )


print(
    "First 12 tree targets:",
    tree_targets[:12]
)

print(
    "Target mismatches:",
    len(target_mismatches)
)


if target_mismatches:

    print()
    print(
        "First target mismatches:"
    )

    for item in target_mismatches[:10]:

        print(
            f"Tree {item[0]}: "
            f"actual={item[1]}, "
            f"expected={item[2]}"
        )

    raise RuntimeError(
        "XGBoost tree target order does not "
        "match the expected interleaved order."
    )


print(
    "Tree target order confirmed."
)

print(
    "CAP, CAP_CORR, SOH repeated "
    "for every boosting round."
)


# ============================================================
# PRINT NODE STATISTICS
# ============================================================

print()
print("=" * 70)
print("NODE STATISTICS")
print("=" * 70)

print(
    "Total nodes:",
    len(nodes)
)

print(
    "Leaf nodes:",
    leaf_node_count
)

print(
    "Numerical splits:",
    numerical_node_count
)

print(
    "Integer threshold splits:",
    integer_threshold_node_count
)


# ============================================================
# PRINT TREE ROOTS
# ============================================================

print()
print(
    "First tree roots:"
)

for i in range(
    min(10, len(tree_roots))
):

    print(
        f"Tree {i:3d}: "
        f"root node {tree_roots[i]}"
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GENERATE ESP32 HEADER
# ============================================================

print()
print(
    "Generating ESP32 header..."
)


with open(
    OUTPUT_HEADER,
    "w"
) as f:

    # ========================================================
    # HEADER
    # ========================================================

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// SparkEV Model 3 - ESP32 XGBoost Model\n"
    )

    f.write(
        "// Generated from model3_battery_degradation_xgb.pkl\n"
    )

    f.write(
        "// Verified native tree representation\n"
    )

    f.write(
        "// ============================================================\n\n"
    )


    f.write(
        "#pragma once\n"
    )

    f.write(
        "#include <math.h>\n"
    )

    f.write(
        "#include <stdint.h>\n\n"
    )


    # ========================================================
    # MODEL CONSTANTS
    # ========================================================

    f.write(
        "// Model dimensions\n"
    )

    f.write(
        f"#define MODEL3_NUM_FEATURES "
        f"{num_features}\n"
    )

    f.write(
        f"#define MODEL3_NUM_TREES "
        f"{num_trees}\n"
    )

    f.write(
        f"#define MODEL3_TREES_PER_OUTPUT "
        f"{TREES_PER_OUTPUT}\n"
    )

    f.write(
        f"#define MODEL3_NUM_OUTPUTS "
        f"{NUM_OUTPUTS}\n"
    )

    f.write(
        f"#define MODEL3_NUM_NODES "
        f"{len(nodes)}\n"
    )

    f.write(
        f"#define MODEL3_NUMERICAL_FEATURES "
        f"{NUMERICAL_FEATURE_COUNT}\n"
    )

    f.write("\n")


    # ========================================================
    # BASE SCORES
    # ========================================================

    #
    # These are the exact native model base scores.
    #
    # CAP      = 72.21674
    # CAP_CORR = 72.515686
    # SOH      = 0.926701
    #
    # ========================================================

    f.write(
        "// Exact XGBoost base scores\n"
    )

    f.write(
        "static const float "
        "MODEL3_BASE_SCORES"
        "[MODEL3_NUM_OUTPUTS] = {\n"
    )

    f.write(
        "    72.21674f,\n"
    )

    f.write(
        "    72.515686f,\n"
    )

    f.write(
        "    0.926701f,\n"
    )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    f.write(
        "// Feature names\n"
    )

    f.write(
        "static const char* "
        "MODEL3_FEATURE_NAMES"
        "[MODEL3_NUM_FEATURES] = {\n"
    )

    for name in feature_names:

        f.write(
            f'    "{name}",\n'
        )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # NODE STRUCTURE
    # ========================================================

    f.write(
        "// Compact XGBoost node\n"
    )

    f.write(
        "struct Model3Node {\n"
    )

    f.write(
        "    int16_t feature;\n"
    )

    f.write(
        "    float threshold;\n"
    )

    f.write(
        "    int32_t yes_child;\n"
    )

    f.write(
        "    int32_t no_child;\n"
    )

    f.write(
        "    int32_t missing_child;\n"
    )

    f.write(
        "    float leaf;\n"
    )

    f.write(
        "    uint8_t categorical;\n"
    )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # NODE ARRAY
    # ========================================================

    f.write(
        "// All tree nodes\n"
    )

    f.write(
        "static const Model3Node "
        "MODEL3_NODES"
        "[MODEL3_NUM_NODES] = {\n"
    )

    for node in nodes:

        f.write(
            "    {"
            f"{node['feature']}, "
            f"{cpp_float(node['threshold'])}, "
            f"{node['yes']}, "
            f"{node['no']}, "
            f"{node['missing']}, "
            f"{cpp_float(node['leaf'])}, "
            f"{1 if node['categorical'] else 0}"
            "},\n"
        )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # TREE ROOTS
    # ========================================================

    f.write(
        "// Root node for every tree\n"
    )

    f.write(
        "static const int32_t "
        "MODEL3_TREE_ROOTS"
        "[MODEL3_NUM_TREES] = {\n"
    )

    for root in tree_roots:

        f.write(
            f"    {root},\n"
        )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # TREE TARGETS
    # ========================================================

    #
    # IMPORTANT:
    #
    # Trees are interleaved by target.
    #
    # 0 -> CAP
    # 1 -> CAP_CORR
    # 2 -> SOH
    # 3 -> CAP
    # ...
    #
    # ========================================================

    f.write(
        "// Target index for every tree\n"
    )

    f.write(
        "// 0 = CAP, 1 = CAP_CORR, 2 = SOH\n"
    )

    f.write(
        "static const uint8_t "
        "MODEL3_TREE_TARGET"
        "[MODEL3_NUM_TREES] = {\n"
    )

    for target in tree_targets:

        f.write(
            f"    {target},\n"
        )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # OUTPUT TREE START
    # ========================================================

    #
    # Retained for compatibility with existing ESP32 code.
    #
    # IMPORTANT:
    # These are NOT contiguous tree ranges.
    #
    # Use MODEL3_TREE_TARGET[tree_id] to determine
    # the destination output.
    #
    # ========================================================

    f.write(
        "// Legacy compatibility values.\n"
    )

    f.write(
        "// Do not use these as contiguous tree ranges.\n"
    )

    f.write(
        "static const int "
        "MODEL3_OUTPUT_TREE_START"
        "[MODEL3_NUM_OUTPUTS] = {\n"
    )

    f.write(
        "    0,\n"
    )

    f.write(
        "    1,\n"
    )

    f.write(
        "    2,\n"
    )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # OUTPUT NAMES
    # ========================================================

    f.write(
        "static const char* "
        "MODEL3_OUTPUT_NAMES"
        "[MODEL3_NUM_OUTPUTS] = {\n"
    )

    f.write(
        '    "cap",\n'
    )

    f.write(
        '    "cap_corr",\n'
    )

    f.write(
        '    "soh",\n'
    )

    f.write(
        "};\n\n"
    )


    # ========================================================
    # FEATURE TYPE HELPER
    # ========================================================

    #
    # Although the final 77 features are integer features,
    # their tree routing is threshold-based.
    #
    # This helper therefore returns false for tree routing.
    #
    # ========================================================

    f.write(
        "// All Model 3 tree splits use threshold routing.\n"
    )

    f.write(
        "static inline bool "
        "model3_is_categorical_feature"
        "(int feature) {\n"
    )

    f.write(
        "    (void)feature;\n"
    )

    f.write(
        "    return false;\n"
    )

    f.write(
        "}\n\n"
    )


    # ========================================================
    # TREE TARGET HELPER
    # ========================================================

    f.write(
        "// Return output index for a tree.\n"
    )

    f.write(
        "static inline int "
        "model3_tree_target"
        "(int tree_id) {\n"
    )

    f.write(
        "    return MODEL3_TREE_TARGET[tree_id];\n"
    )

    f.write(
        "}\n\n"
    )


    # ========================================================
    # END
    # ========================================================

    f.write(
        "// ============================================================\n"
    )

    f.write(
        "// End of generated Model 3 header\n"
    )

    f.write(
        "// ============================================================\n"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

file_size = os.path.getsize(
    OUTPUT_HEADER
)


print()
print("=" * 70)
print("MODEL 3 ESP32 CONVERSION COMPLETE")
print("=" * 70)

print(
    "Features             :",
    num_features
)

print(
    "Trees                :",
    num_trees
)

print(
    "Trees/output         :",
    TREES_PER_OUTPUT
)

print(
    "Outputs              :",
    NUM_OUTPUTS
)

print(
    "Total nodes          :",
    len(nodes)
)

print(
    "Leaf nodes           :",
    leaf_node_count
)

print(
    "Numerical splits     :",
    numerical_node_count
)

print(
    "Integer threshold    :",
    integer_threshold_node_count
)

print(
    "Target order         :",
    "0,1,2 repeated"
)

print(
    "Header size          :",
    f"{file_size / 1024:.1f} KB"
)

print(
    "Header               :",
    OUTPUT_HEADER
)

print()
print(
    "IMPORTANT:"
)

print(
    "Tree output mapping uses:"
)

print(
    "tree_id % 3"
)

print(
    "0 -> CAP"
)

print(
    "1 -> CAP_CORR"
)

print(
    "2 -> SOH"
)

print("=" * 70)