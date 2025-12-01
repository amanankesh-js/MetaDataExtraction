import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from textwrap import dedent

def load_latest_json(base="output"):
    timestamp_folders = sorted(
        [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
    if not timestamp_folders:
        raise FileNotFoundError("No timestamp folders found inside /output")

    latest = timestamp_folders[0]
    json_path = os.path.join(latest, "shots_gemini_output.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found")

    return latest, json_path


def plot_histogram():
    latest, json_path = load_latest_json()

    with open(json_path, "r") as f:
        shots = json.load(f)

    durations = np.array([shot["end"] - shot["start"] for shot in shots])

    total_shots = len(durations)
    mean_d = durations.mean()
    median_d = np.median(durations)
    min_d = durations.min()
    max_d = durations.max()

    # Percentiles for deeper insight
    p10 = np.percentile(durations, 10)
    p25 = np.percentile(durations, 25)
    p75 = np.percentile(durations, 75)
    p90 = np.percentile(durations, 90)

    # Pacing score
    avg_cuts_per_min = (60 / mean_d) if mean_d > 0 else 0
    if mean_d < 3:
        pacing_label = "⚡ Fast-cut pacing"
    elif mean_d < 6:
        pacing_label = "🎬 Moderate pacing"
    else:
        pacing_label = "📖 Slow / cinematic pacing"

    # Bin calculation (Freedman–Diaconis)
    iqr = np.subtract(*np.percentile(durations, [75, 25]))
    bin_width = 2 * iqr / (total_shots ** (1/3))
    bins = 10 if bin_width <= 0 else int(np.ceil((max_d - min_d) / bin_width))

    plt.figure(figsize=(13, 6))
    counts, bins_edges, patches = plt.hist(
        durations, bins=bins, alpha=0.90, edgecolor="white"
    )

    # 🎨 Gradient color bars
    cmap = cm.get_cmap("viridis")
    for patch, count in zip(patches, counts):
        patch.set_facecolor(cmap(count / counts.max() if counts.max() else 0))

    # X-axis granularity
    interval = 1.0
    xticks = np.arange(0, max_d + interval, interval)
    plt.xticks(xticks, rotation=45, fontsize=8)

    # Statistics markers
    plt.axvline(mean_d, color="red", linestyle="--", linewidth=1.8, label=f"Mean = {mean_d:.2f}s")
    plt.axvline(median_d, color="green", linestyle="--", linewidth=1.8, label=f"Median = {median_d:.2f}s")

    plt.title("Distribution of Shot Durations", fontsize=16, weight="bold")
    plt.xlabel("Shot Duration (seconds)", fontsize=12)
    plt.ylabel("Number of Shots", fontsize=12)
    plt.grid(axis="y", alpha=0.35)
    plt.legend()

    # Count labels above each bar
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(
                patch.get_x() + patch.get_width() / 2,
                count + 0.05,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                color="black"
            )

    # ---- Top 5 bins ----
    sorted_idx = np.argsort(counts)[::-1]
    top_bins = []
    for rank, idx in enumerate(sorted_idx[:5], start=1):
        start = bins_edges[idx]
        end = bins_edges[idx + 1]
        top_bins.append(f"{rank}) {start:.2f}–{end:.2f}s → {int(counts[idx])}")

    # ===== Styled statistics panel =====
    panel_title = "📊 SHOT STATISTICS"
    panel_main = dedent(
        f"""
        • Total shots: {total_shots}
        • Mean / Median: {mean_d:.2f}s / {median_d:.2f}s
        • Min / Max: {min_d:.2f}s / {max_d:.2f}s
        """
    ).strip()

    panel_percentiles = dedent(
        f"""
        P10: {p10:.2f}s   P25: {p25:.2f}s
        P75: {p75:.2f}s   P90: {p90:.2f}s
        """
    ).strip()

    panel_pacing = f"{pacing_label}\n≈ {avg_cuts_per_min:.1f} cuts per minute"
    panel_top5 = "\n".join(top_bins)

    styled_text = (
        f"{panel_title}\n\n{panel_main}"
        f"\n\n──────── Percentiles ────────\n{panel_percentiles}"
        f"\n\n──────── Pacing ────────────\n{panel_pacing}"
        f"\n\n──────── Top 5 Bins ────────\n{panel_top5}"
    )

    plt.text(
        0.985,
        0.985,
        styled_text,
        transform=plt.gca().transAxes,
        fontsize=9.35,
        ha="right",
        va="top",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#ffffff",
            edgecolor="#3b0764",  # deep purple
            linewidth=1.4,
            alpha=0.92,
        ),
    )

    # Save
    plot_path = os.path.join(latest, "shot_duration_histogram.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print(f"\n📁 Histogram saved → {plot_path}")
    print(f"📂 Loaded JSON from → {json_path}")


if __name__ == "__main__":
    plot_histogram()
