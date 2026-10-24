import os
import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb


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
SCALER_PATH = os.path.join(MODELS_DIR, "model3_scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "model3_numerical_features.pkl")
DATA_PATH = os.path.join(DATA_DIR, "NMC01_Car02_fast_feature.xlsx")

NUMERICAL_FEATURE_COUNT = 122
EXPECTED_FEATURE_COUNT = 199

EXPECTED_TREES = 300
NUM_OUTPUTS = 3
TREES_PER_OUTPUT = EXPECTED_TREES // NUM_OUTPUTS

# Small floating-point tolerance for manual tree aggregation.
PREDICTION_TOLERANCE = 1e-4

LEAF_TOLERANCE = 1e-6

BASE_SCORES = np.array(
    [
        72.21674,     # CAP
        72.515686,    # CAP_CORR
        0.926701      # SOH
    ],
    dtype=np.float64
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SparkEV Model 3 Verification")
print("=" * 70)

print()
print("Loading Model 3...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    numerical_features = pickle.load(f)

booster = model.get_booster()

model_features = booster.feature_names

feature_to_index = {
    name: i
    for i, name in enumerate(model_features)
}


# ============================================================
# MODEL VALIDATION
# ============================================================

print()
print("=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

print(
    "Feature count:",
    len(model_features)
)

print(
    "Numerical/scaled feature count:",
    len(numerical_features)
)

print(
    "Base scores:",
    BASE_SCORES
)

print(
    "Booster feature types:"
)

print(
    "  float:",
    sum(
        t == "float"
        for t in booster.feature_types
    )
)

print(
    "  integer:",
    sum(
        t == "i"
        for t in booster.feature_types
    )
)

if len(model_features) != EXPECTED_FEATURE_COUNT:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURE_COUNT} "
        f"model features, found "
        f"{len(model_features)}"
    )

if len(numerical_features) != NUMERICAL_FEATURE_COUNT:

    raise RuntimeError(
        f"Expected {NUMERICAL_FEATURE_COUNT} "
        f"numerical features, found "
        f"{len(numerical_features)}"
    )

if EXPECTED_TREES % NUM_OUTPUTS != 0:

    raise RuntimeError(
        "Expected tree count must be divisible "
        "by number of outputs."
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    if "t_start" in df.columns:

        df["t_start"] = pd.to_datetime(
            df["t_start"]
        )

    if "t_end" in df.columns:

        df["t_end"] = pd.to_datetime(
            df["t_end"]
        )


    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    if (
        "t_start" in df.columns
        and "t_end" in df.columns
    ):

        df["cycle_duration_hours"] = (
            df["t_end"] - df["t_start"]
        ).dt.total_seconds() / 3600

        df["start_month"] = (
            df["t_start"].dt.month
        )

        df["start_day_of_week"] = (
            df["t_start"].dt.dayofweek
        )

        df["start_hour"] = (
            df["t_start"].dt.hour
        )

        df.drop(
            columns=[
                "t_start",
                "t_end"
            ],
            inplace=True
        )


    # --------------------------------------------------------
    # Remove target columns
    # --------------------------------------------------------

    for col in [
        "cap",
        "cap_corr",
        "soh"
    ]:

        if col in df.columns:

            df.drop(
                columns=[col],
                inplace=True
            )


    # --------------------------------------------------------
    # Car identifier
    # --------------------------------------------------------

    if "car_identifier" in df.columns:

        df["car_identifier"] = (
            df["car_identifier"]
            .fillna("Unknown")
            .astype(str)
        )

        df = pd.get_dummies(
            df,
            columns=["car_identifier"],
            prefix="car"
        )


    # --------------------------------------------------------
    # Add missing numerical features
    # --------------------------------------------------------

    missing_numerical = [
        col
        for col in numerical_features
        if col not in df.columns
    ]

    if missing_numerical:

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=missing_numerical
                )
            ],
            axis=1
        )


    # --------------------------------------------------------
    # Apply exact saved scaler
    # --------------------------------------------------------

    df[numerical_features] = (
        scaler.transform(
            df[numerical_features]
        )
    )


    # --------------------------------------------------------
    # Add missing model features
    # --------------------------------------------------------

    missing_model_features = [
        col
        for col in model_features
        if col not in df.columns
    ]

    if missing_model_features:

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=missing_model_features
                )
            ],
            axis=1
        )


    # --------------------------------------------------------
    # Exact model feature order
    # --------------------------------------------------------

    df = df[
        model_features
    ]

    return df


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

raw_data = pd.read_excel(
    DATA_PATH
)

print(
    "Raw shape:",
    raw_data.shape
)


# ============================================================
# PREPROCESS
# ============================================================

X_df = preprocess_data(
    raw_data
)

print(
    "Prepared shape:",
    X_df.shape
)

if X_df.shape[1] != EXPECTED_FEATURE_COUNT:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURE_COUNT} "
        f"prepared features, got "
        f"{X_df.shape[1]}"
    )


# Preserve NaN values exactly.
# XGBoost handles missing values natively.

X = X_df.to_numpy(
    dtype=np.float32
)

final_features = X[-1]


# ============================================================
# NATIVE XGBOOST REFERENCE
# ============================================================

print()
print("=" * 70)
print("NATIVE XGBOOST REFERENCE")
print("=" * 70)

dmatrix = xgb.DMatrix(
    X_df
)

reference = booster.predict(
    dmatrix
)

reference_final = reference[-1]

print(
    f"CAP      : {reference_final[0]:.8f}"
)

print(
    f"CAP_CORR : {reference_final[1]:.8f}"
)

print(
    f"SOH      : {reference_final[2]:.8f}"
)


# ============================================================
# NATIVE LEAF INDICES
# ============================================================

print()
print("=" * 70)
print("GETTING XGBOOST NATIVE LEAF INDICES")
print("=" * 70)

native_leaf_matrix = booster.predict(
    dmatrix,
    pred_leaf=True
)

native_leaf_final = (
    native_leaf_matrix[-1]
)

print(
    "Leaf matrix shape:",
    native_leaf_matrix.shape
)

print(
    "Expected:",
    (493, EXPECTED_TREES)
)

if native_leaf_matrix.shape[1] != EXPECTED_TREES:

    raise RuntimeError(
        "Unexpected number of leaf columns."
    )


# ============================================================
# LOAD RAW NATIVE TREE STRUCTURE
# ============================================================

print()
print("=" * 70)
print("LOADING NATIVE TREE ARRAYS")
print("=" * 70)

native_json = json.loads(
    booster.save_raw(
        raw_format="json"
    )
)

native_trees = (
    native_json["learner"]
    ["gradient_booster"]
    ["model"]
    ["trees"]
)

print(
    "Native trees:",
    len(native_trees)
)

if len(native_trees) != EXPECTED_TREES:

    raise RuntimeError(
        f"Expected {EXPECTED_TREES} trees, "
        f"found {len(native_trees)}"
    )


# ============================================================
# TREE EVALUATOR
# ============================================================

def evaluate_tree(
    tree,
    features
):

    split_indices = tree[
        "split_indices"
    ]

    split_conditions = tree[
        "split_conditions"
    ]

    left_children = tree[
        "left_children"
    ]

    right_children = tree[
        "right_children"
    ]

    default_left = tree[
        "default_left"
    ]

    base_weights = tree[
        "base_weights"
    ]

    node = 0

    while True:

        left = int(
            left_children[node]
        )

        right = int(
            right_children[node]
        )


        # ----------------------------------------------------
        # LEAF
        # ----------------------------------------------------

        if (
            left == -1
            and right == -1
        ):

            return (
                int(node),
                float(
                    base_weights[node]
                )
            )


        # ----------------------------------------------------
        # FEATURE
        # ----------------------------------------------------

        feature = int(
            split_indices[node]
        )

        value = float(
            features[feature]
        )


        # ----------------------------------------------------
        # MISSING VALUE
        # ----------------------------------------------------

        if np.isnan(value):

            if int(
                default_left[node]
            ) == 1:

                node = left

            else:

                node = right

            continue


        # ----------------------------------------------------
        # NUMERICAL / INTEGER ROUTING
        # ----------------------------------------------------
        #
        # XGBoost routing:
        #
        # value < threshold -> YES / left
        # value >= threshold -> NO / right
        #
        # This also applies to the 77 integer car_* features.
        #
        # ----------------------------------------------------

        threshold = float(
            split_conditions[node]
        )

        if value < threshold:

            node = left

        else:

            node = right


# ============================================================
# VERIFY EVERY TREE PATH
# ============================================================

print()
print("=" * 70)
print("VERIFYING ALL 300 TREE PATHS")
print("=" * 70)

tree_mismatches = []

manual_leaf_values = []

native_leaf_values = []


for tree_id in range(
    EXPECTED_TREES
):

    manual_node, manual_value = (
        evaluate_tree(
            native_trees[tree_id],
            final_features
        )
    )

    native_node = int(
        native_leaf_final[tree_id]
    )

    native_tree = (
        native_trees[tree_id]
    )

    native_value = float(
        native_tree[
            "base_weights"
        ][native_node]
    )


    manual_leaf_values.append(
        manual_value
    )

    native_leaf_values.append(
        native_value
    )


    if manual_node != native_node:

        tree_mismatches.append(
            {
                "tree": tree_id,
                "manual_node": manual_node,
                "native_node": native_node,
                "manual_value": manual_value,
                "native_value": native_value,
            }
        )


print()
print(
    "Tree path mismatches:",
    len(tree_mismatches)
)


if tree_mismatches:

    print()
    print("FIRST MISMATCHES:")

    for item in tree_mismatches[:20]:

        print(
            f"Tree {item['tree']:3d} | "
            f"manual node={item['manual_node']:3d} | "
            f"native node={item['native_node']:3d} | "
            f"manual={item['manual_value']:.8f} | "
            f"native={item['native_value']:.8f}"
        )

else:

    print(
        "ALL 300 TREE PATHS MATCH XGBOOST."
    )


# ============================================================
# LEAF VALUE CHECK
# ============================================================

manual_leaf_values = np.array(
    manual_leaf_values,
    dtype=np.float64
)

native_leaf_values = np.array(
    native_leaf_values,
    dtype=np.float64
)

leaf_value_error = np.max(
    np.abs(
        manual_leaf_values
        - native_leaf_values
    )
)


print()
print("=" * 70)
print("LEAF VALUE CHECK")
print("=" * 70)

print(
    "Maximum leaf-value difference:",
    leaf_value_error
)


# ============================================================
# VERIFY TREE TARGET ORDER
# ============================================================

print()
print("=" * 70)
print("VERIFYING TREE TARGET ORDER")
print("=" * 70)

tree_dataframe = (
    booster.trees_to_dataframe()
)

tree_target_map = (
    tree_dataframe
    .groupby("Tree")["Target"]
    .first()
)

target_order = np.array(
    [
        int(
            tree_target_map.loc[
                tree_id
            ]
        )
        for tree_id in range(
            EXPECTED_TREES
        )
    ],
    dtype=np.int32
)

expected_target_order = (
    np.arange(
        EXPECTED_TREES
    )
    % NUM_OUTPUTS
)

target_order_mismatches = np.where(
    target_order
    != expected_target_order
)[0]


print(
    "First 12 tree targets:",
    target_order[:12].tolist()
)

print(
    "Expected first 12:",
    expected_target_order[:12].tolist()
)

print(
    "Target-order mismatches:",
    len(target_order_mismatches)
)


if len(target_order_mismatches) == 0:

    print(
        "Tree target order confirmed:"
    )

    print(
        "CAP, CAP_CORR, SOH repeated "
        "for all boosting rounds."
    )

else:

    print(
        "WARNING: Tree target order differs "
        "from expected interleaving."
    )


# ============================================================
# MANUAL MODEL PREDICTION
# ============================================================

print()
print("=" * 70)
print("MANUAL MODEL PREDICTION")
print("=" * 70)

manual_outputs = []


for target in range(
    NUM_OUTPUTS
):

    # Trees are interleaved:
    #
    # 0 -> CAP
    # 1 -> CAP_CORR
    # 2 -> SOH
    # 3 -> CAP
    # 4 -> CAP_CORR
    # 5 -> SOH
    #
    # Therefore:
    # tree_id % 3 == target

    target_tree_indices = np.arange(
        target,
        EXPECTED_TREES,
        NUM_OUTPUTS
    )

    total = (
        BASE_SCORES[target]
        + np.sum(
            manual_leaf_values[
                target_tree_indices
            ]
        )
    )

    manual_outputs.append(
        total
    )


manual_outputs = np.array(
    manual_outputs,
    dtype=np.float64
)


print(
    f"CAP      : {manual_outputs[0]:.8f}"
)

print(
    f"CAP_CORR : {manual_outputs[1]:.8f}"
)

print(
    f"SOH      : {manual_outputs[2]:.8f}"
)


# ============================================================
# DIFFERENCE FROM XGBOOST
# ============================================================

prediction_error = (
    manual_outputs
    - reference_final
)

absolute_error = np.abs(
    prediction_error
)

maximum_error = np.max(
    absolute_error
)

total_absolute_error = np.sum(
    absolute_error
)


print()
print("=" * 70)
print("PREDICTION ERROR")
print("=" * 70)

print(
    f"CAP error      : "
    f"{prediction_error[0]:.12f}"
)

print(
    f"CAP_CORR error : "
    f"{prediction_error[1]:.12f}"
)

print(
    f"SOH error      : "
    f"{prediction_error[2]:.12f}"
)

print(
    f"Maximum absolute error: "
    f"{maximum_error:.12f}"
)

print(
    f"Total absolute error  : "
    f"{total_absolute_error:.12f}"
)

print(
    f"Allowed tolerance     : "
    f"{PREDICTION_TOLERANCE:.12f}"
)


# ============================================================
# GOLDEN RESULT
# ============================================================

golden = np.array(
    [
        94.946098,
        94.722366,
        0.83178329
    ],
    dtype=np.float64
)


print()
print("=" * 70)
print("GOLDEN RESULT")
print("=" * 70)

print(
    "Expected:"
)

print(
    golden
)

print()
print(
    "XGBoost:"
)

print(
    reference_final
)

print()
print(
    "Manual:"
)

print(
    manual_outputs
)

print()
print(
    "Manual - Expected:"
)

print(
    manual_outputs
    - golden
)


# ============================================================
# SAMPLE TREE COMPARISON
# ============================================================

print()
print("=" * 70)
print("SAMPLE TREE COMPARISON")
print("=" * 70)

for tree_id in [
    0,
    1,
    2,
    41,
    47,
    100,
    147,
    200,
    286,
    297,
    299
]:

    manual_node, manual_value = (
        evaluate_tree(
            native_trees[tree_id],
            final_features
        )
    )

    native_node = int(
        native_leaf_final[tree_id]
    )

    target = int(
        target_order[tree_id]
    )

    target_name = [
        "CAP",
        "CAP_CORR",
        "SOH"
    ][target]

    print(
        f"Tree {tree_id:3d} | "
        f"target={target_name:8s} | "
        f"manual node={manual_node:3d} | "
        f"native node={native_node:3d} | "
        f"value={manual_value:.8f}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("FINAL STATUS")
print("=" * 70)

tree_paths_valid = (
    len(tree_mismatches) == 0
)

leaf_values_valid = (
    leaf_value_error <= LEAF_TOLERANCE
)

predictions_valid = (
    maximum_error <= PREDICTION_TOLERANCE
)

target_order_valid = (
    len(target_order_mismatches) == 0
)


if (
    tree_paths_valid
    and leaf_values_valid
    and predictions_valid
    and target_order_valid
):

    print(
        "SUCCESS: Manual evaluator matches XGBoost "
        "within floating-point tolerance."
    )

    print()
    print(
        "Model 3 tree evaluation is verified."
    )

    print()
    print(
        "Verified outputs:"
    )

    print(
        f"CAP      = {manual_outputs[0]:.8f}"
    )

    print(
        f"CAP_CORR = {manual_outputs[1]:.8f}"
    )

    print(
        f"SOH      = {manual_outputs[2]:.8f}"
    )

    print()
    print(
        "Maximum prediction error = "
        f"{maximum_error:.8f}"
    )

    print()
    print(
        "READY FOR ESP32 CONVERTER UPDATE."
    )

else:

    print(
        "NOT READY: Manual evaluator still "
        "differs from XGBoost."
    )

    print()
    print(
        "Do NOT deploy the ESP32 header yet."
    )

print("=" * 70)