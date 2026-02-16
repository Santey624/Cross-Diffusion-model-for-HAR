# ============================================================
# Download and Preprocess WISDM Dataset
# Converts to CogAge-compatible format (npy arrays)
#
# WISDM: 51 subjects, 18 activities, 20Hz
# Sensors: phone_acc, phone_gyro, watch_acc, watch_gyro
# Format: subject-id, activity-code, timestamp, x, y, z
# ============================================================

from pathlib import Path
import numpy as np
import zipfile
import os
import urllib.request
from scipy.signal import resample

# ============================================================
# CONFIG
# ============================================================
WISDM_URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
RAW_DIR = Path("data/wisdm/raw")
OUT_DIR = Path("data/wisdm/arrays")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# WISDM sampling rate
WISDM_HZ = 20

# CogAge target lengths (per 5-second window at their native sampling rates)
# We resample WISDM windows to match these lengths
TARGET_LENGTHS = {
    "phone_acc": 800,    # CogAge: 200Hz * ~4s
    "phone_gyro": 800,
    "watch_acc": 268,    # CogAge: 67Hz * ~4s
    "watch_gyro": 268,
}

# WISDM directory mapping -> our sensor names
WISDM_SENSORS = {
    "phone/accel": "phone_acc",
    "phone/gyro": "phone_gyro",
    "watch/accel": "watch_acc",
    "watch/gyro": "watch_gyro",
}

# Window size in seconds (to match CogAge ~4s windows)
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(WINDOW_SEC * WISDM_HZ)  # 80 samples at 20Hz

# Train/test split: leave last 10 subjects for test
TEST_SUBJECTS = list(range(1642, 1652))  # Last 10 subject IDs (adjust based on actual data)

# Activity codes in WISDM
ACTIVITY_CODES = {
    'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting',
    'E': 'standing', 'F': 'typing', 'G': 'teeth', 'H': 'soup',
    'I': 'chips', 'J': 'pasta', 'K': 'drinking', 'L': 'sandwich',
    'M': 'kicking', 'O': 'catch', 'P': 'dribbling', 'Q': 'writing',
    'R': 'clapping', 'S': 'folding',
}


# ============================================================
# DOWNLOAD
# ============================================================
def download_wisdm():
    zip_path = RAW_DIR / "wisdm.zip"

    if zip_path.exists():
        print(f"Already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading WISDM dataset (~296MB)...")
    urllib.request.urlretrieve(WISDM_URL, zip_path)
    print(f"Downloaded to {zip_path}")
    return zip_path


def extract_wisdm(zip_path):
    extract_dir = RAW_DIR / "extracted"
    if extract_dir.exists():
        print(f"Already extracted: {extract_dir}")
        return extract_dir

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")
    return extract_dir


# ============================================================
# PARSE WISDM FILES
# ============================================================
def parse_wisdm_file(filepath):
    """
    Parse a WISDM raw data file.
    Format: subject-id, activity-code, timestamp, x, y, z;
    Returns list of (subject_id, activity, timestamp, x, y, z)
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                subject_id = int(parts[0].strip())
                activity = parts[1].strip()
                timestamp = int(parts[2].strip())
                x = float(parts[3].strip())
                y = float(parts[4].strip())
                z = float(parts[5].strip().rstrip(';'))
                records.append((subject_id, activity, timestamp, x, y, z))
            except (ValueError, IndexError):
                continue
    return records


def find_wisdm_files(extract_dir):
    """Find the raw data files in the extracted WISDM directory."""
    sensor_files = {}

    # WISDM structure: wisdm-dataset/raw/phone/accel/*.txt (one per subject)
    # or might be: wisdm-dataset/raw/phone/accel/data_*.txt
    for wisdm_path, sensor_name in WISDM_SENSORS.items():
        # Try different possible paths
        candidates = [
            extract_dir / "wisdm-dataset" / "raw" / wisdm_path,
            extract_dir / "raw" / wisdm_path,
            extract_dir / "WISDM" / "raw" / wisdm_path,
        ]

        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break

        if found is None:
            # Search recursively
            pattern = wisdm_path.replace("/", os.sep)
            for p in extract_dir.rglob("*"):
                if p.is_dir() and str(p).endswith(pattern.replace("/", os.sep)):
                    found = p
                    break

        if found is not None:
            txt_files = sorted(found.glob("*.txt"))
            if txt_files:
                sensor_files[sensor_name] = txt_files
                print(f"  {sensor_name}: {len(txt_files)} files in {found}")
            else:
                print(f"  {sensor_name}: directory found but no .txt files in {found}")
        else:
            print(f"  {sensor_name}: NOT FOUND (searched in {extract_dir})")

    return sensor_files


# ============================================================
# WINDOWING + RESAMPLING
# ============================================================
def create_windows(records, window_samples):
    """
    Create fixed-length windows from continuous records.
    Groups by (subject_id, activity), then slides a window.

    Returns list of (subject_id, activity, np.array of shape (window_samples, 3))
    """
    from collections import defaultdict

    # Group by (subject, activity)
    groups = defaultdict(list)
    for subject_id, activity, timestamp, x, y, z in records:
        groups[(subject_id, activity)].append([x, y, z])

    windows = []
    for (subject_id, activity), data in groups.items():
        data = np.array(data, dtype=np.float32)

        # Slide window with 50% overlap
        stride = window_samples // 2
        for start in range(0, len(data) - window_samples + 1, stride):
            window = data[start:start + window_samples]
            if len(window) == window_samples:
                windows.append((subject_id, activity, window))

    return windows


def resample_window(window, target_length):
    """Resample a (T, 3) window to (target_length, 3) using scipy."""
    if window.shape[0] == target_length:
        return window
    return resample(window, target_length, axis=0).astype(np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("WISDM Dataset Download & Preprocessing")
    print(f"{'='*60}\n")

    # Download and extract
    zip_path = download_wisdm()
    extract_dir = extract_wisdm(zip_path)

    # Find data files
    print("\nLocating sensor files...")
    sensor_files = find_wisdm_files(extract_dir)

    if not sensor_files:
        print("\nERROR: Could not find WISDM data files.")
        print("Please check the extracted directory structure:")
        for p in sorted(extract_dir.rglob("*"))[:30]:
            print(f"  {p}")
        return

    # Parse and create windows for each sensor
    print("\nParsing and windowing...")
    sensor_windows = {}  # sensor_name -> list of (subject, activity, array)

    for sensor_name, files in sensor_files.items():
        print(f"\n  Processing {sensor_name}...")
        all_records = []
        for f in files:
            records = parse_wisdm_file(f)
            all_records.extend(records)
        print(f"    Total records: {len(all_records)}")

        windows = create_windows(all_records, WINDOW_SAMPLES)
        print(f"    Windows ({WINDOW_SEC}s, {WINDOW_SAMPLES} samples): {len(windows)}")
        sensor_windows[sensor_name] = windows

    # Find common (subject, activity, window_index) across all sensors
    # Since windows are created independently, align by subject+activity
    print("\nAligning windows across sensors...")

    # Get all subjects
    all_subjects = set()
    for sensor_name, windows in sensor_windows.items():
        for subject_id, activity, _ in windows:
            all_subjects.add(subject_id)
    all_subjects = sorted(all_subjects)
    print(f"  Total subjects: {len(all_subjects)}")

    # Split subjects into train/test
    n_test = max(1, len(all_subjects) // 5)  # 20% for test
    test_subjects = set(all_subjects[-n_test:])
    train_subjects = set(all_subjects[:-n_test])
    print(f"  Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")

    # Group windows by (subject, activity) and take min count across sensors
    from collections import defaultdict
    grouped = {s: defaultdict(list) for s in sensor_windows}
    for sensor_name, windows in sensor_windows.items():
        for subject_id, activity, data in windows:
            grouped[sensor_name][(subject_id, activity)].append(data)

    # Find common keys across all sensors
    common_keys = None
    for sensor_name in sensor_windows:
        keys = set(grouped[sensor_name].keys())
        common_keys = keys if common_keys is None else common_keys & keys

    print(f"  Common (subject, activity) groups: {len(common_keys)}")

    # Create aligned arrays
    train_arrays = {name: [] for name in sensor_windows}
    test_arrays = {name: [] for name in sensor_windows}

    for key in sorted(common_keys):
        subject_id, activity = key

        # Take min window count across sensors for this key
        min_windows = min(len(grouped[s][key]) for s in sensor_windows)

        for sensor_name in sensor_windows:
            target_len = TARGET_LENGTHS[sensor_name]
            for i in range(min_windows):
                window = grouped[sensor_name][key][i]
                resampled = resample_window(window, target_len)

                if subject_id in test_subjects:
                    test_arrays[sensor_name].append(resampled)
                else:
                    train_arrays[sensor_name].append(resampled)

    # Convert to numpy and save
    print(f"\nSaving arrays...")

    # Map our sensor names to CogAge-compatible file names
    SENSOR_TO_FILE = {
        "phone_acc": "Accelerometer",
        "phone_gyro": "Gyroscope",
        "watch_acc": "MSAccelerometer",
        "watch_gyro": "MSGyroscope",
    }

    train_dir = OUT_DIR / "training"
    test_dir = OUT_DIR / "testing"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name in sensor_windows:
        train_arr = np.stack(train_arrays[sensor_name])  # (N, T, 3)
        test_arr = np.stack(test_arrays[sensor_name])

        file_suffix = SENSOR_TO_FILE[sensor_name]
        np.save(train_dir / f"train{file_suffix}.npy", train_arr)
        np.save(test_dir / f"test{file_suffix}.npy", test_arr)
        print(f"  {sensor_name}: train={train_arr.shape}, test={test_arr.shape}")

    # For sensors WISDM doesn't have (phone_grav, phone_lacc, glasses_acc),
    # create zero-filled placeholders with correct shapes
    MISSING_SENSORS = {
        "phone_grav": ("Gravity", 800),
        "phone_lacc": ("LinearAcceleration", 800),
        "glasses_acc": ("JinsAccelerometer", 80),
    }

    n_train = len(train_arrays[list(sensor_windows.keys())[0]])
    n_test = len(test_arrays[list(sensor_windows.keys())[0]])

    for sensor_name, (file_suffix, seq_len) in MISSING_SENSORS.items():
        train_arr = np.zeros((n_train, seq_len, 3), dtype=np.float32)
        test_arr = np.zeros((n_test, seq_len, 3), dtype=np.float32)
        np.save(train_dir / f"train{file_suffix}.npy", train_arr)
        np.save(test_dir / f"test{file_suffix}.npy", test_arr)
        print(f"  {sensor_name}: ZERO-FILLED train={train_arr.shape}, test={test_arr.shape}")

    # Save dummy labels
    np.save(train_dir / "trainLabels.npy", np.zeros(n_train, dtype=np.int64))
    np.save(test_dir / "testLabels.npy", np.zeros(n_test, dtype=np.int64))

    print(f"\n{'='*60}")
    print(f"WISDM preprocessing complete!")
    print(f"Output: {OUT_DIR}")
    print(f"Train: {n_train} samples, Test: {n_test} samples")
    print(f"Sensors available: {list(sensor_windows.keys())}")
    print(f"Sensors zero-filled: {list(MISSING_SENSORS.keys())}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
