from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "esp32" / "tools" / "vectors"

FILES = {
    17: BASE / "model2_driver17_vector.txt",
    24: BASE / "model2_driver24_vector.txt",
}

OUT = PROJECT_ROOT / "esp32" / "firmware" / "SparkEV_Integrated" / "model2_driver_vectors.h"


def extract_vector(path):
    text = path.read_text()

    values = {}

    # Actual format:
    # 45.0000000000f,  // 0 soc_start
    pattern = re.compile(
        r"^\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"f,?\s*//\s*(\d+)"
    )

    for line in text.splitlines():
        m = pattern.match(line)

        if m:
            value = float(m.group(1))
            index = int(m.group(2))
            values[index] = value

    if len(values) != 57:
        raise RuntimeError(
            f"{path.name}: expected 57 features, found {len(values)}"
        )

    for i in range(57):
        if i not in values:
            raise RuntimeError(
                f"{path.name}: missing feature index {i}"
            )

    return [values[i] for i in range(57)]


def cpp_float(value):
    s = format(value, ".10g")

    if "e" not in s.lower() and "." not in s:
        s += ".0"

    return s + "f"


vectors = {}

for driver_id, path in FILES.items():
    vector = extract_vector(path)
    vectors[driver_id] = vector

    print(
        f"Driver {driver_id}: "
        f"{len(vector)} features extracted"
    )


# ------------------------------------------------------------
# Verify that Driver 17 and Driver 24 are actually different
# ------------------------------------------------------------

differences = []

for i in range(57):
    if vectors[17][i] != vectors[24][i]:
        differences.append(i)

if not differences:
    raise RuntimeError(
        "Driver 17 and Driver 24 vectors are identical. "
        "Check the source vector files before generating the header."
    )

print(
    f"Driver vectors differ at {len(differences)} feature(s)."
)


# ------------------------------------------------------------
# Generate C++ header
# ------------------------------------------------------------

lines = []

lines.append("#ifndef MODEL2_DRIVER_VECTORS_H")
lines.append("#define MODEL2_DRIVER_VECTORS_H")
lines.append("")
lines.append("// ============================================================")
lines.append("// SparkEV Model 2 Driver-Specific Test Vectors")
lines.append("// ============================================================")
lines.append("//")
lines.append("// Driver 17:")
lines.append("//   Transaction: 44010000100000052203030401089310")
lines.append("//   Expected probability ~= 0.999552")
lines.append("//   Expected class = Faulty")
lines.append("//")
lines.append("// Driver 24:")
lines.append("//   Transaction: 44020100400000012010151800001210")
lines.append("//   Expected probability ~= 0.005638")
lines.append("//   Expected class = Healthy")
lines.append("//")
lines.append("// Genuine 57-feature vectors generated from the")
lines.append("// corresponding charging transactions.")
lines.append("// ============================================================")
lines.append("")

for driver_id in [17, 24]:

    vector = vectors[driver_id]

    lines.append(
        f"static const float MODEL2_DRIVER{driver_id}_VECTOR[57] = {{"
    )

    for i, value in enumerate(vector):
        lines.append(
            f"    {cpp_float(value)},  // {i}"
        )

    lines.append("};")
    lines.append("")


lines.append("#endif")

OUT.write_text("\n".join(lines))

print("==============================================")
print("Model 2 driver vectors generated")
print("==============================================")
print("Driver 17: 57 features")
print("Driver 24: 57 features")
print(f"Created: {OUT}")
print("==============================================")
