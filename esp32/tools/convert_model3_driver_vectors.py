import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VECTORS_DIR = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors")
FIRMWARE_DIR = os.path.join(
    PROJECT_ROOT, "esp32", "firmware", "SparkEV_Integrated"
)

FILES = {
    17: os.path.join(VECTORS_DIR, "model3_driver17_vector.txt"),
    24: os.path.join(VECTORS_DIR, "model3_driver24_vector.txt"),
}

OUTPUT = os.path.join(FIRMWARE_DIR, "model3_driver_vectors.h")


def cpp_value(value):
    value = value.strip()

    if value == "NAN":
        return "NAN"

    # Remove an existing float suffix if present.
    if value.endswith("f"):
        value = value[:-1]

    # Convert integer-looking values to valid C++ float literals.
    if value in ("0", "1"):
        return value + ".0f"

    return value + "f"


def extract_vector(path):
    values = []

    with open(path, "r") as f:
        for line in f:
            match = re.match(
                r"\s*(\d+):\s+[^=]+=\s*(.+)",
                line
            )

            if match:
                index = int(match.group(1))
                value = match.group(2).strip()

                if index < 199:
                    values.append((index, cpp_value(value)))

    values.sort(key=lambda x: x[0])

    if len(values) != 199:
        raise RuntimeError(
            f"{path}: expected 199 features, "
            f"found {len(values)}"
        )

    for expected, (index, _) in enumerate(values):
        if index != expected:
            raise RuntimeError(
                f"{path}: expected index {expected}, "
                f"found {index}"
            )

    return [value for _, value in values]


with open(OUTPUT, "w") as out:

    out.write("// ==================================================\n")
    out.write("// SparkEV Model 3 Driver Vectors\n")
    out.write("// Generated automatically from verified CPU vectors\n")
    out.write("// ==================================================\n\n")

    for driver_id, filename in FILES.items():

        vector = extract_vector(filename)

        out.write(
            f"// Driver {driver_id}: {os.path.basename(filename)}\n"
        )

        out.write(
            f"static const float MODEL3_DRIVER{driver_id}_VECTOR[199] = {{\n"
        )

        for i, value in enumerate(vector):

            comma = "," if i < 198 else ""

            out.write(
                f"    {value}{comma}  // {i}\n"
            )

        out.write("};\n\n")

        print(
            f"Driver {driver_id}: "
            f"{len(vector)} features extracted"
        )

print()
print(f"Created: {OUTPUT}")
