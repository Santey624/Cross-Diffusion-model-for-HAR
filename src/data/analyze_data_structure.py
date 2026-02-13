# ============================================================
# Analyze CogAge Data Structure
# Check if subject/session info is available
# ============================================================

from pathlib import Path
import numpy as np

DATA_ROOTS = {
    "blho": Path("data/cogage/python/arrays/blho"),
    "bbh": Path("data/cogage/python/arrays/bbh"),
    "state": Path("data/cogage/python/arrays/state"),
}

SENSOR_FILES = [
    "Accelerometer",
    "Gyroscope",
    "Gravity",
    "LinearAcceleration",
    "MSAccelerometer",
    "MSGyroscope",
    "JinsAccelerometer",
]


def analyze_subset(name, root):
    print(f"\n{'='*60}")
    print(f"Subset: {name}")
    print(f"{'='*60}")

    for split in ["training", "testing"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  {split}: NOT FOUND")
            continue

        print(f"\n  {split.upper()}:")

        # List all files
        files = list(split_dir.glob("*.npy"))
        print(f"    Files: {[f.name for f in sorted(files)]}")

        # Check for metadata
        meta_files = list(split_dir.glob("*.npz")) + list(split_dir.glob("*.json")) + list(split_dir.glob("*.txt"))
        if meta_files:
            print(f"    Metadata: {[f.name for f in meta_files]}")

        # Load one sensor to check structure
        prefix = "train" if split == "training" else "test"

        for sensor in SENSOR_FILES:
            path = split_dir / f"{prefix}{sensor}.npy"
            if path.exists():
                arr = np.load(path)
                print(f"    {sensor:20s}: shape={arr.shape}, dtype={arr.dtype}")

                # Check for any obvious patterns (e.g., subject boundaries)
                if len(arr.shape) == 3:
                    N, T, C = arr.shape
                    # Check variance along samples - might show subject boundaries
                    sample_means = arr.mean(axis=(1, 2))
                    print(f"      Sample means: min={sample_means.min():.4f}, max={sample_means.max():.4f}, std={sample_means.std():.4f}")

        # Check for label files
        label_path = split_dir / f"{prefix}Labels.npy"
        if label_path.exists():
            labels = np.load(label_path)
            unique_labels = np.unique(labels)
            print(f"    Labels: {len(labels)} samples, {len(unique_labels)} unique classes")
            print(f"      Classes: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"      Classes: {unique_labels}")

            # Check class distribution
            from collections import Counter
            counts = Counter(labels.tolist() if hasattr(labels, 'tolist') else labels)
            print(f"      Distribution: min={min(counts.values())}, max={max(counts.values())}, mean={np.mean(list(counts.values())):.1f}")


def check_parent_directory():
    """Check what's in the parent directory for additional info."""
    print(f"\n{'='*60}")
    print("Checking parent directories for metadata...")
    print(f"{'='*60}")

    base = Path("data/cogage")
    if base.exists():
        for item in sorted(base.rglob("*")):
            if item.is_file() and item.suffix in [".txt", ".json", ".csv", ".md", ".npz"]:
                print(f"  Found: {item}")
                if item.suffix == ".txt" and item.stat().st_size < 10000:
                    print(f"    Content preview:")
                    with open(item) as f:
                        for i, line in enumerate(f):
                            if i < 5:
                                print(f"      {line.rstrip()}")
                            else:
                                print("      ...")
                                break


def estimate_subjects():
    """Try to estimate subject boundaries based on data patterns."""
    print(f"\n{'='*60}")
    print("Estimating subject structure...")
    print(f"{'='*60}")

    # Based on CogAge description:
    # - 4 subjects
    # - 2 sessions each
    # - Each activity performed at least 10 times per session
    # - 61 activities total

    for name, root in DATA_ROOTS.items():
        for split in ["training", "testing"]:
            split_dir = root / split
            prefix = "train" if split == "training" else "test"

            acc_path = split_dir / f"{prefix}Accelerometer.npy"
            label_path = split_dir / f"{prefix}Labels.npy"

            if acc_path.exists() and label_path.exists():
                acc = np.load(acc_path)
                labels = np.load(label_path)

                n_samples = acc.shape[0]
                n_classes = len(np.unique(labels))

                # If session 1 = train, session 2 = test:
                # Each subject did ~10 executions per activity per session
                # So per session: ~10 * n_classes samples per subject
                # With 4 subjects: ~40 * n_classes per session

                samples_per_subject_estimate = n_samples / 4
                samples_per_activity = n_samples / n_classes

                print(f"\n  {name}/{split}:")
                print(f"    Total samples: {n_samples}")
                print(f"    Classes: {n_classes}")
                print(f"    Samples per class (avg): {samples_per_activity:.1f}")
                print(f"    If 4 subjects: ~{samples_per_subject_estimate:.0f} samples each")
                print(f"    If 10 exec/activity/subject: expect {10 * n_classes * 4} samples")


if __name__ == "__main__":
    for name, root in DATA_ROOTS.items():
        analyze_subset(name, root)

    check_parent_directory()
    estimate_subjects()
