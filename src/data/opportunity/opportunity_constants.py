WINDOW = 64    # samples per window (~2.1 s at 30 Hz)
STRIDE = 32    # 50% overlap

# sensor_key -> (npy file suffix, sequence length)
OPP_SENSOR_FILES = {
    "back_acc":    ("BackAcc",    WINDOW),
    "back_gyro":   ("BackGyro",   WINDOW),
    "rua_acc":     ("RuaAcc",     WINDOW),
    "rua_gyro":    ("RuaGyro",    WINDOW),
    "rla_acc":     ("RlaAcc",     WINDOW),
    "rla_gyro":    ("RlaGyro",    WINDOW),
    "lua_acc":     ("LuaAcc",     WINDOW),
    "lua_gyro":    ("LuaGyro",    WINDOW),
    "lla_acc":     ("LlaAcc",     WINDOW),
    "lla_gyro":    ("LlaGyro",    WINDOW),
    "lshoe_acc":   ("LshoeAcc",   WINDOW),
    "lshoe_gyro":  ("LshoeGyro",  WINDOW),
    "rshoe_acc":   ("RshoeAcc",   WINDOW),
    "rshoe_gyro":  ("RshoeGyro",  WINDOW),
}

OPP_SENSOR_NAMES = list(OPP_SENSOR_FILES.keys())

# Drop a whole body location at once during diffusion training.
OPP_DEVICE_GROUPS = {
    "back":      ["back_acc", "back_gyro"],
    "right_arm": ["rua_acc", "rua_gyro", "rla_acc", "rla_gyro"],
    "left_arm":  ["lua_acc", "lua_gyro", "lla_acc", "lla_gyro"],
    "shoes":     ["lshoe_acc", "lshoe_gyro", "rshoe_acc", "rshoe_gyro"],
}

# Raw .dat column indices (0-indexed). Each entry is the [X, Y, Z] triple.
# The raw file is 1-indexed in column_names.txt; subtract 1.
SENSOR_COLUMNS = {
    "back_acc":    [37, 38, 39],     # IMU BACK accX/Y/Z      (cols 38-40)
    "back_gyro":   [40, 41, 42],     # IMU BACK gyroX/Y/Z     (cols 41-43)
    "rua_acc":     [50, 51, 52],     # IMU RUA  accX/Y/Z      (cols 51-53)
    "rua_gyro":    [53, 54, 55],     # IMU RUA  gyroX/Y/Z     (cols 54-56)
    "rla_acc":     [63, 64, 65],     # IMU RLA  accX/Y/Z      (cols 64-66)
    "rla_gyro":    [66, 67, 68],     # IMU RLA  gyroX/Y/Z     (cols 67-69)
    "lua_acc":     [76, 77, 78],     # IMU LUA  accX/Y/Z      (cols 77-79)
    "lua_gyro":    [79, 80, 81],     # IMU LUA  gyroX/Y/Z     (cols 80-82)
    "lla_acc":     [89, 90, 91],     # IMU LLA  accX/Y/Z      (cols 90-92)
    "lla_gyro":    [92, 93, 94],     # IMU LLA  gyroX/Y/Z     (cols 93-95)
    "lshoe_acc":   [108, 109, 110],  # L-SHOE Body_Ax/y/z     (cols 109-111)
    "lshoe_gyro":  [111, 112, 113],  # L-SHOE AngVelBodyX/Y/Z (cols 112-114)
    "rshoe_acc":   [124, 125, 126],  # R-SHOE Body_Ax/y/z     (cols 125-127)
    "rshoe_gyro":  [127, 128, 129],  # R-SHOE AngVelBodyX/Y/Z (cols 128-130)
}

# ------------------------------------------------------------
# Label tracks
# ------------------------------------------------------------
# Opportunity annotates each timestamp with SEVEN parallel label tracks,
# not one. They live in raw .dat columns 244-250 (1-indexed) -> 243-249
# (0-indexed). Each track is an independent classification target; the
# raw class ids are non-contiguous and 0 always means the Null class.
#
#   track name              raw col   0-idx   non-null classes
#   ----------------------  -------   -----   ----------------
#   locomotion              244       243     4   (Stand/Walk/Sit/Lie)
#   hl_activity             245       244     5   (high-level activities)
#   ll_left_arm             246       245     13  (left-arm gestures)
#   ll_left_arm_object      247       246     23  (left-arm + object)
#   ll_right_arm            248       247     13  (right-arm gestures)
#   ll_right_arm_object     249       248     23  (right-arm + object)
#   ml_both_arms            250       249     17  (mid-level gestures)
#
# See dataset/.../label_legend.txt for the full id -> name mapping.
OPP_LABEL_TRACKS = {
    "locomotion":          243,
    "hl_activity":         244,
    "ll_left_arm":         245,
    "ll_left_arm_object":  246,
    "ll_right_arm":        247,
    "ll_right_arm_object": 248,
    "ml_both_arms":        249,
}

OPP_LABEL_TRACK_NAMES = list(OPP_LABEL_TRACKS.keys())

# Default track used when none is specified (backwards compatible).
DEFAULT_LABEL_TRACK = "locomotion"

# Per-track on-disk label file suffix: train{Suffix}.npy / test{Suffix}.npy.
def label_file_suffix(track: str) -> str:
    return f"Labels_{track}"

# Backwards-compatible alias for the original single-track column.
LABEL_COLUMN = OPP_LABEL_TRACKS[DEFAULT_LABEL_TRACK]
