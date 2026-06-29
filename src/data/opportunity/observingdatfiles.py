from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
data_root    = project_root / "dataset" / "OpportunityUCIDataset"

SAMPLING_RATE = 30   # Hz

# All 7 label columns (0-indexed) with their names and class mappings
LABEL_CONFIGS = {
    "Locomotion": {
        "col": 243,
        "classes": {0: "Null", 1: "Stand", 2: "Walk", 4: "Sit", 5: "Lie"},
    },
    "HL_Activity": {
        "col": 244,
        "classes": {0: "Null", 101: "Relaxing", 102: "Coffee time", 103: "Early morning",
                    104: "Cleanup", 105: "Sandwich time"},
    },
    "LL_Left_Arm": {
        "col": 245,
        "classes": {0: "Null", 201: "Unlock", 202: "Stir", 203: "Lock", 204: "Close",
                    205: "Reach", 206: "Open", 207: "Sip", 208: "Clean", 209: "Bite",
                    210: "Cut", 211: "Spread", 212: "Release", 213: "Move"},
    },
    "LL_Left_Arm_Object": {
        "col": 246,
        "classes": {0: "Null", 301: "Bottle", 302: "Salami", 303: "Bread", 304: "Sugar",
                    305: "Dishwasher", 306: "Switch", 307: "Milk", 308: "Drawer3",
                    309: "Spoon", 310: "Knife cheese", 311: "Drawer2", 312: "Table",
                    313: "Glass", 314: "Cheese", 315: "Chair", 316: "Door1", 317: "Door2",
                    318: "Plate", 319: "Drawer1", 320: "Fridge", 321: "Cup",
                    322: "Knife salami", 323: "Lazychair"},
    },
    "LL_Right_Arm": {
        "col": 247,
        "classes": {0: "Null", 401: "Unlock", 402: "Stir", 403: "Lock", 404: "Close",
                    405: "Reach", 406: "Open", 407: "Sip", 408: "Clean", 409: "Bite",
                    410: "Cut", 411: "Spread", 412: "Release", 413: "Move"},
    },
    "LL_Right_Arm_Object": {
        "col": 248,
        "classes": {0: "Null", 501: "Bottle", 502: "Salami", 503: "Bread", 504: "Sugar",
                    505: "Dishwasher", 506: "Switch", 507: "Milk", 508: "Drawer3",
                    509: "Spoon", 510: "Knife cheese", 511: "Drawer2", 512: "Table",
                    513: "Glass", 514: "Cheese", 515: "Chair", 516: "Door1", 517: "Door2",
                    518: "Plate", 519: "Drawer1", 520: "Fridge", 521: "Cup",
                    522: "Knife salami", 523: "Lazychair"},
    },
    "ML_Both_Arms": {
        "col": 249,
        "classes": {0: "Null", 406516: "Open Door 1", 406517: "Open Door 2",
                    404516: "Close Door 1", 404517: "Close Door 2", 406520: "Open Fridge",
                    404520: "Close Fridge", 406505: "Open Dishwasher", 404505: "Close Dishwasher",
                    406519: "Open Drawer 1", 404519: "Close Drawer 1", 406511: "Open Drawer 2",
                    404511: "Close Drawer 2", 406508: "Open Drawer 3", 404508: "Close Drawer 3",
                    408512: "Clean Table", 407521: "Drink from Cup", 405506: "Toggle Switch"},
    },
}


def read_data():
    dat_file = data_root / "dataset" / "S1-ADL2.dat"
    df = pd.read_csv(dat_file, sep=r'\s+', header=None)
    print(f"Data shape: {df.shape}")
    print(df)
    return df


def get_segments(labels):
    """Find contiguous segments. Returns list of (label, duration_snapshots)."""
    segments = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        segments.append((labels[i], j - i))
        i = j
    return segments


def discover_label_values():
    """Print the unique values found in each label column across all files."""
    dat_files = sorted((data_root / "dataset").glob("S*-*.dat"))
    print(f"\nDiscovering actual label values in each column...\n")
    for col_name, cfg in LABEL_CONFIGS.items():
        all_vals = set()
        for dat_path in dat_files:
            df   = pd.read_csv(dat_path, sep=r"\s+", header=None)
            vals = np.nan_to_num(df.iloc[:, cfg["col"]].to_numpy(), nan=0).astype(int)
            all_vals.update(np.unique(vals).tolist())
        print(f"  {col_name:<22} (col {cfg['col']}): {sorted(all_vals)}")


def analyze_activity_durations():
    """
    Reads all 24 .dat files and reports how long each activity lasts
    across all 7 label columns. Helps choose the right WINDOW size.
    """
    dat_files = sorted((data_root / "dataset").glob("S*-*.dat"))
    print(f"\nAnalyzing activity durations across {len(dat_files)} recordings...\n")

    # Collect per-label-column durations
    all_durations = {col_name: defaultdict(list) for col_name in LABEL_CONFIGS}

    for dat_path in dat_files:
        df = pd.read_csv(dat_path, sep=r"\s+", header=None)
        for col_name, cfg in LABEL_CONFIGS.items():
            labels = np.nan_to_num(df.iloc[:, cfg["col"]].to_numpy(), nan=0).astype(int)
            for label, dur in get_segments(labels):
                all_durations[col_name][label].append(dur)

    # Report per label column
    for col_name, cfg in LABEL_CONFIGS.items():
        durations = all_durations[col_name]
        classes   = cfg["classes"]
        print(f"\n{'='*70}")
        print(f"  {col_name}")
        print(f"{'='*70}")
        print(f"  {'Class':<22} {'Segs':>6} {'Min(s)':>8} {'Max(s)':>8} {'Mean(s)':>8} {'Median(s)':>10}")
        print(f"  {'-'*64}")

        non_null_durs = []
        for label, name in sorted(classes.items()):
            durs = np.array(durations.get(label, []))
            if len(durs) == 0:
                continue
            if label != 0:
                non_null_durs.extend(durs.tolist())
            print(
                f"  {name:<22} {len(durs):>6} "
                f"{durs.min()/SAMPLING_RATE:>8.2f} "
                f"{durs.max()/SAMPLING_RATE:>8.2f} "
                f"{durs.mean()/SAMPLING_RATE:>8.2f} "
                f"{np.median(durs)/SAMPLING_RATE:>10.2f}"
            )

        if non_null_durs:
            non_null = np.array(non_null_durs)
            print(f"\n  Shortest non-Null : {non_null.min()} snaps = {non_null.min()/SAMPLING_RATE:.2f}s")
            print(f"  5th percentile    : {np.percentile(non_null,5):.0f} snaps = {np.percentile(non_null,5)/SAMPLING_RATE:.2f}s")
            print(f"  Median            : {np.median(non_null):.0f} snaps = {np.median(non_null)/SAMPLING_RATE:.2f}s")

    # Final window recommendation across ALL label columns combined
    print(f"\n{'='*70}")
    print("  OVERALL WINDOW RECOMMENDATION (all label columns combined)")
    print(f"{'='*70}")
    all_non_null = []
    for col_name, cfg in LABEL_CONFIGS.items():
        durations = all_durations[col_name]
        for label, durs in durations.items():
            if label != 0:
                all_non_null.extend(durs)
    all_non_null = np.array(all_non_null)
    print(f"  Shortest non-Null segment : {all_non_null.min()} snaps = {all_non_null.min()/SAMPLING_RATE:.2f}s")
    print(f"  5th percentile            : {np.percentile(all_non_null,5):.0f} snaps = {np.percentile(all_non_null,5)/SAMPLING_RATE:.2f}s")
    print(f"  Median                    : {np.median(all_non_null):.0f} snaps = {np.median(all_non_null)/SAMPLING_RATE:.2f}s")
    print()
    for w in [32, 64, 128, 256]:
        pct = (all_non_null >= w).mean() * 100
        print(f"  WINDOW={w:3d} ({w/SAMPLING_RATE:.1f}s) -> {pct:.1f}% of all activity segments are longer than this")


read_data()
discover_label_values()
analyze_activity_durations()