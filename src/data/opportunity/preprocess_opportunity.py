# ============================================================
# Opportunity UCI -> per-sensor windowed .npy arrays
#
# Converts the raw whitespace-separated S*-*.dat recordings into
# the same on-disk format the diffusion training expects:
#
#   data/opportunity/arrays/{training|testing}/
#       train{Suffix}.npy        -> (N, WINDOW, 3)  per sensor
#       trainLabels_{track}.npy  -> (N,)            one file per label track
#
# Opportunity has SEVEN parallel label tracks (locomotion, hl_activity,
# the four low-level arm/object tracks, and ml_both_arms). We window and
# save all of them; the downstream classifier picks one at load time.
#
# Steps per recording:
#   1. Read the .dat (NaN literals -> np.nan automatically).
#   2. Linear-interpolate NaNs along time, then ffill/bfill edges,
#      then zero-fill anything still missing.
#   3. Slice out the 14 triaxial body-IMU sensors.
#   4. Sliding window (WINDOW, STRIDE) -> (n_win, WINDOW, 3).
#   5. Per track, window label = majority class in the window.
#
# Train/test split is by run: ADL4 + ADL5 -> testing, rest -> training.
#
# Usage (from repo root):
#   python -m src.data.opportunity.preprocess_opportunity
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.opportunity.opportunity_constants import (
    OPP_SENSOR_FILES, SENSOR_COLUMNS, OPP_LABEL_TRACKS, label_file_suffix,
    WINDOW, STRIDE,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT    = project_root / "dataset" / "OpportunityUCIDataset" / "dataset"
OUT_ROOT     = project_root / "data" / "opportunity" / "arrays"

# Runs whose files go to the test split (matched as substrings of the
# filename stem, e.g. "S1-ADL4").
TEST_RUNS = ("ADL4", "ADL5")


# ------------------------------------------------------------
# NaN handling
# ------------------------------------------------------------
''' In this case we are utlizing linear interpolation. This draws a straight line between the last known value and the next known value incase if the data have missing values (NaN)
    This is a way how we are handling the missing values in the opportunity dataset in case of (NaN) values.'''
def fill_nans(df_sensor: pd.DataFrame) -> np.ndarray:
    """Linear-interpolate along time, then fill remaining edge NaNs.""" 
    filled = (
        df_sensor
        .interpolate(method="linear", limit_direction="both", axis=0)
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    return filled.to_numpy(dtype=np.float32)


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------
def window_signal(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(T, C) -> (n_win, window, C) via sliding window."""
    T = arr.shape[0]
    if T < window:
        return np.empty((0, window, arr.shape[1]), dtype=arr.dtype)
    starts = range(0, T - window + 1, stride)
    return np.stack([arr[s:s + window] for s in starts], axis=0)


def window_labels(labels: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Majority-vote label for each window. (T,) -> (n_win,).

    Uses np.unique rather than np.bincount so it works for tracks with
    large, non-contiguous raw ids (e.g. ml_both_arms ids up to 406520).
    """
    T = labels.shape[0]
    if T < window:
        return np.empty((0,), dtype=np.int64)
    out = []
    for s in range(0, T - window + 1, stride):
        seg = labels[s:s + window].astype(np.int64)
        vals, counts = np.unique(seg, return_counts=True)
        out.append(vals[counts.argmax()])
    return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------
# Per-file processing
# ------------------------------------------------------------
def process_file(dat_path: Path):
    """Returns (per_sensor: dict[str, (N,WINDOW,3)], labels: dict[track, (N,)])."""
    df = pd.read_csv(dat_path, sep=r"\s+", header=None)
    raw = df.to_numpy()

    per_sensor = {}
    for key, cols in SENSOR_COLUMNS.items():
        sensor_clean = fill_nans(df.iloc[:, cols])          # (T, 3)
        per_sensor[key] = window_signal(sensor_clean, WINDOW, STRIDE)

    # One windowed label vector per track. NaN -> 0 (Null class).
    win_labels = {}
    for track, col in OPP_LABEL_TRACKS.items():
        labels_full = np.nan_to_num(raw[:, col], nan=0.0).astype(np.int64)
        win_labels[track] = window_labels(labels_full, WINDOW, STRIDE)
    return per_sensor, win_labels


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    dat_files = sorted(DATA_ROOT.glob("S*-*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files under {DATA_ROOT}")

    print(f"Found {len(dat_files)} recordings under {DATA_ROOT}")

    splits = {
        "training": {k: [] for k in OPP_SENSOR_FILES},
        "testing":  {k: [] for k in OPP_SENSOR_FILES},
    }
    # One list of windowed label vectors per (split, track).
    split_labels = {
        "training": {t: [] for t in OPP_LABEL_TRACKS},
        "testing":  {t: [] for t in OPP_LABEL_TRACKS},
    }

    for dat_path in dat_files:
        stem = dat_path.stem  # e.g. "S1-ADL4"
        split = "testing" if any(r in stem for r in TEST_RUNS) else "training"

        per_sensor, win_labels = process_file(dat_path)
        n = win_labels[next(iter(OPP_LABEL_TRACKS))].shape[0]
        print(f"  {stem:12s} -> {split:8s}  windows={n}")

        for key in OPP_SENSOR_FILES:
            splits[split][key].append(per_sensor[key])
        for track in OPP_LABEL_TRACKS:
            split_labels[split][track].append(win_labels[track])

    for split in ("training", "testing"):
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "train" if split == "training" else "test"

        for key, (suffix, _) in OPP_SENSOR_FILES.items():
            arr = np.concatenate(splits[split][key], axis=0)
            np.save(out_dir / f"{prefix}{suffix}.npy", arr)

        n_windows = None
        for track in OPP_LABEL_TRACKS:
            labels = np.concatenate(split_labels[split][track], axis=0)
            np.save(out_dir / f"{prefix}{label_file_suffix(track)}.npy", labels)
            n_windows = labels.shape[0]

        print(f"\n[{split}] saved {n_windows} windows to {out_dir}")
        for track in OPP_LABEL_TRACKS:
            labels = np.concatenate(split_labels[split][track], axis=0)
            dist = dict(zip(*np.unique(labels, return_counts=True)))
            print(f"  {track:20s} {len(dist):3d} classes  dist={dist}")

    print("\nDone.")


if __name__ == "__main__":
    main()
