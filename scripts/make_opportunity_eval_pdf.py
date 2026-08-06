#!/usr/bin/env python3
"""Build a multi-page PDF summarizing Opportunity evaluation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("eval_outputs/opportunity")
OUT_PATH = ROOT / "Opportunity_Evaluation_Results_All_Tracks.pdf"

TRACKS = [
    "locomotion",
    "hl_activity",
    "ll_left_arm",
    "ll_left_arm_object",
    "ll_right_arm",
    "ll_right_arm_object",
    "ml_both_arms",
]

TRACK_TITLES = {
    "locomotion": "Locomotion",
    "hl_activity": "High-level Activity (hl_activity)",
    "ll_left_arm": "Left-arm Gestures (ll_left_arm)",
    "ll_left_arm_object": "Left-arm + Object (ll_left_arm_object)",
    "ll_right_arm": "Right-arm Gestures (ll_right_arm)",
    "ll_right_arm_object": "Right-arm + Object (ll_right_arm_object)",
    "ml_both_arms": "Mid-level Both Arms (ml_both_arms)",
}


def fmt(m):
    return f"{m['acc']:.3f}/{m['af1']:.3f}"


def nanfmt(x):
    return "nan" if x != x else f"{x:.4f}"


def draw_table(ax, col_labels, cell_text, title, fontsize=7, col_widths=None, winner_col=None):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.25)

    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor("#1f4e79")
        cell.set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(cell_text, start=1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f2f2f2")
            else:
                cell.set_facecolor("white")
            if winner_col is not None and j == winner_col:
                txt = str(row[j])
                if "CrossDiff" in txt:
                    cell.set_facecolor("#d4edda")
                elif "Mean" in txt:
                    cell.set_facecolor("#f8d7da")
                elif "Tie" in txt:
                    cell.set_facecolor("#fff3cd")

    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(len(cell_text) + 1):
                table[(i, j)].set_width(w)
    return table


def main():
    data = {}
    for t in TRACKS:
        data[t] = json.loads((ROOT / f"{t}_metrics.json").read_text())

    summary_rows = []
    for t in TRACKS:
        d = data[t]
        base = d["scenarios"]["all_real"]
        strict_cd = strict_mean = ties = 0
        for c in d["comparison"]:
            p = c["pattern"]
            cd = d["scenarios"][f"{p}+crossdiff"]["acc"]
            mn = d["scenarios"][f"{p}+mean"]["acc"]
            if cd > mn + 1e-12:
                strict_cd += 1
            elif mn > cd + 1e-12:
                strict_mean += 1
            else:
                ties += 1
        af1_wins = 0
        for c in d["comparison"]:
            p = c["pattern"]
            if (
                d["scenarios"][f"{p}+crossdiff"]["af1"]
                > d["scenarios"][f"{p}+mean"]["af1"]
            ):
                af1_wins += 1
        summary_rows.append(
            [
                t,
                f"{base['acc']*100:.2f}%",
                f"{base['af1']*100:.2f}%",
                nanfmt(base["map"]),
                nanfmt(base["auc"]),
                str(d["n_classes"]),
                f"{strict_cd}/22",
                f"{strict_mean}/22",
                f"{ties}/22",
                f"{af1_wins}/22",
                f"{d['crossdiff_wins']}/22",
            ]
        )

    with PdfPages(OUT_PATH) as pdf:
        # Cover
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        ax.axis("off")
        ax.text(0.0, 0.95, "Opportunity Cross-Sensor Diffusion Evaluation",
                fontsize=18, fontweight="bold")
        ax.text(0.0, 0.90,
                "Downstream C-LSTM-A recognition after missing-sensor imputation",
                fontsize=11, color="#333333")
        ax.text(0.0, 0.82, "What this PDF reports", fontsize=13, fontweight="bold")
        ax.text(
            0.0, 0.76,
            "• Dataset: Opportunity (14 IMU sensors, 4 device groups)\n"
            "• CrossDiff = diffusion imputation of missing sensors\n"
            "• Mean-Fill = replace missing sensors with training-set mean waveform\n"
            "• Baseline (all_real) = complete sensors, no missingness\n"
            "• Reported as Acc / AF1; MAP and AUC included when defined\n"
            "• Winner decided by Accuracy (same rule as evaluation script)\n"
            "• Δ Acc = CrossDiff Acc − Mean Acc",
            fontsize=10, va="top",
        )
        ax.text(0.0, 0.48, "Sensors (14)", fontsize=12, fontweight="bold")
        ax.text(
            0.0, 0.43,
            "back_acc, back_gyro | rua_acc, rua_gyro, rla_acc, rla_gyro |\n"
            "lua_acc, lua_gyro, lla_acc, lla_gyro | lshoe_acc, lshoe_gyro, rshoe_acc, rshoe_gyro",
            fontsize=9, family="monospace", va="top",
        )
        ax.text(0.0, 0.33, "Device groups (4)", fontsize=12, fontweight="bold")
        ax.text(
            0.0, 0.28,
            "back (2) | right_arm (4) | left_arm (4) | shoes (4)",
            fontsize=9, family="monospace", va="top",
        )
        ax.text(0.0, 0.20, "Scenarios per label track", fontsize=12, fontweight="bold")
        ax.text(
            0.0, 0.15,
            "45 scenarios = 1 all_real + 28 single-sensor + 8 device-all + 8 only-*",
            fontsize=9, family="monospace", va="top",
        )
        ax.text(
            0.0, 0.07,
            "Source: eval_outputs/opportunity/<track>_metrics.json\n"
            "Caution: left-arm tracks are Null-dominated; high Acc with very low AF1 needs careful interpretation.",
            fontsize=9, color="#666666", va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Summary
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        cols = [
            "Track", "Acc", "AF1", "MAP", "AUC", "#cls",
            "CD Acc>", "Mean Acc>", "Ties", "CD AF1>", "JSON CD",
        ]
        draw_table(
            ax, cols, summary_rows,
            title="Summary across all 7 Opportunity label tracks",
            fontsize=7.5,
            col_widths=[0.16, 0.07, 0.07, 0.08, 0.08, 0.05, 0.08, 0.09, 0.06, 0.08, 0.08],
        )
        ax.text(
            0.0, -0.08,
            "CD Acc> = CrossDiff Acc strictly greater than Mean. "
            "JSON CD = win count saved by eval script (ties may count as CrossDiff).",
            transform=ax.transAxes, fontsize=8, color="#444444",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Per-track pages
        for t in TRACKS:
            d = data[t]
            base = d["scenarios"]["all_real"]
            rows = [["all_real", fmt(base), "—", "—", "—", "—"]]
            strict_cd = 0
            for c in d["comparison"]:
                p = c["pattern"]
                cd = d["scenarios"][f"{p}+crossdiff"]
                mn = d["scenarios"][f"{p}+mean"]
                delta = cd["acc"] - mn["acc"]
                if cd["acc"] > mn["acc"] + 1e-12:
                    winner = "CrossDiff"
                    strict_cd += 1
                elif mn["acc"] > cd["acc"] + 1e-12:
                    winner = "Mean"
                else:
                    winner = "Tie"
                rows.append([p, fmt(base), fmt(cd), fmt(mn), winner, f"{delta:+.3f}"])

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            title = (
                f"{TRACK_TITLES[t]}\n"
                f"Baseline all_real: Acc={base['acc']:.4f}  AF1={base['af1']:.4f}  "
                f"MAP={nanfmt(base['map'])}  AUC={nanfmt(base['auc'])}  "
                f"n_classes={d['n_classes']}  |  "
                f"CrossDiff Acc wins={strict_cd}/22 (JSON={d['crossdiff_wins']}/22)"
            )
            draw_table(
                ax,
                ["Pattern", "Baseline Acc/AF1", "CrossDiff Acc/AF1",
                 "Mean-Fill Acc/AF1", "Winner", "Δ Acc"],
                rows,
                title=title,
                fontsize=7.2,
                col_widths=[0.18, 0.16, 0.18, 0.18, 0.14, 0.10],
                winner_col=4,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Device-level matrix
        focus = [
            "back_all", "right_arm_all", "left_arm_all", "shoes_all",
            "only_back", "only_right_arm", "only_left_arm", "only_shoes",
        ]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        cols = ["Pattern"] + [t[:12] for t in TRACKS]
        cell = []
        for p in focus:
            row = [p]
            for t in TRACKS:
                cd = data[t]["scenarios"][f"{p}+crossdiff"]["acc"]
                mn = data[t]["scenarios"][f"{p}+mean"]["acc"]
                w = "C" if cd > mn else ("M" if mn > cd else "=")
                row.append(f"{cd:.2f}/{mn:.2f}({w})")
            cell.append(row)
        draw_table(
            ax, cols, cell,
            title="Device-level & only-* across tracks (CrossDiff / Mean Acc, winner)",
            fontsize=7,
            col_widths=[0.14] + [0.107] * 7,
        )
        ax.text(
            0.0, -0.08,
            "C = CrossDiff better Acc, M = Mean better Acc, = = tie.",
            transform=ax.transAxes, fontsize=8, color="#444444",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Findings
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        ax.axis("off")
        ax.text(0.0, 0.95, "Key verified findings", fontsize=16, fontweight="bold")
        findings = (
            "1) Complete-data baselines (all_real)\n"
            "   • locomotion: 87.32% Acc / 89.16% AF1  (main trustworthy track)\n"
            "   • hl_activity: 69.28% Acc / 69.28% AF1\n"
            "   • ml_both_arms: 88.97% Acc / 56.02% AF1\n\n"
            "2) CrossDiff is not globally better than Mean-Fill\n"
            "   • locomotion 9/22, hl_activity 7/22, ml_both_arms 8/22 CrossDiff Acc wins\n"
            "   • Helps when missing sensors are important and recoverable (often back/arms)\n"
            "   • Often loses for shoes and many gyroscope channels\n\n"
            "3) Example strong CrossDiff wins\n"
            "   • locomotion back_all +0.099; back_acc +0.066; right_arm_all +0.053\n"
            "   • hl_activity left_arm_all +0.095; lla_acc +0.099\n\n"
            "4) Example strong Mean wins\n"
            "   • locomotion shoes_all -0.068 for CrossDiff\n"
            "   • hl_activity right_arm_all -0.092; shoes_all -0.060\n\n"
            "5) Left-arm tracks need caution\n"
            "   • Null-class roughly 80-85%; AF1 roughly 0.04-0.08\n"
            "   • Many tiny Acc deltas; high Acc-win counts are less meaningful\n"
            "   • Right-arm tracks have healthier AF1 and larger drops when right arm is missing\n\n"
            "6) Coverage\n"
            "   • Every label track evaluated all 14 sensors and all 4 device groups\n"
            "   • 45 scenarios per track; metrics recomputed from saved JSON files"
        )
        ax.text(0.0, 0.88, findings, fontsize=9.5, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {OUT_PATH}")
    print(f"size_bytes {OUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
