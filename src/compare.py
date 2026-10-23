"""Stage 6: compare isolated, federated, federated + DP and centralised training.

Reads the JSON files written by baseline.py and federated.py, then writes
results/comparison.md (the full table) and four charts to results/charts/,
each in a light and a dark version.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

from config import DP_EPSILONS, NUM_BANKS, RESULTS_DIR

CHARTS_DIR = RESULTS_DIR / "charts"
DPI = 200  # 1 CSS px = 2 device px; sizes below are in CSS px
PT = 0.72  # points per CSS px at 100 px per inch

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "accent": "#2a78d6", "series2": "#eb6834", "reference": "#c3c2b7",
        # ordinal ramp: weakest -> strongest (eps 1, 5, 10, no DP); 'before' / 'after' for dumbbells
        "ramp": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "before": "#86b6ef", "after": "#1c5cab",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3987e5", "series2": "#d95926", "reference": "#52514e",
        "ramp": ["#184f95", "#2a78d6", "#5598e7", "#9ec5f4"],
        "before": "#184f95", "after": "#5598e7",
    },
}
UNSEEN = {1: "CASH_OUT", 2: "TRANSFER", 3: "CASH_OUT", 4: "TRANSFER"}  # fraud type each bank never saw


def load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def collect() -> tuple[list[dict], dict]:
    """One row per setup, plus the per-round histories of the federated runs."""
    baseline = load("baseline")
    if baseline is None:
        raise SystemExit("results/baseline.json is missing: run src/baseline.py first")
    rows = [
        {"setup": f"Bank {b} alone", "kind": "isolated", **baseline["isolated"][f"bank_{b}"]}
        for b in range(1, NUM_BANKS + 1)
    ]
    histories = {}
    runs = [("federated", "Aavaran federated", None)] + [
        (f"federated_dp_eps{e:g}", f"Aavaran + DP, ε = {e:g}", e) for e in sorted(DP_EPSILONS, reverse=True)
    ]
    for name, label, eps in runs:
        run = load(name)
        if run is not None:
            rows.append({"setup": label, "kind": "federated", "epsilon": eps, **run["final"]})
            histories[label] = run["history"]
    rows.append({"setup": "Centralised (upper limit)", "kind": "centralised", **baseline["centralised"]})
    return rows, histories


# --- drawing helpers -------------------------------------------------------------


def new_figure(t: dict, width: int, height: int, title: str, subtitle: str, rect: list[float]):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=DPI, facecolor=t["surface"])
    fig.text(24 / width, 1 - 22 / height, title, color=t["ink"], fontsize=15 * PT, fontweight="bold", va="top")
    fig.text(24 / width, 1 - 44 / height, subtitle, color=t["ink2"], fontsize=12 * PT, va="top")
    ax = fig.add_axes(rect, facecolor=t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1 * PT)
    ax.tick_params(colors=t["muted"], labelsize=11 * PT, length=0, pad=6)
    ax.grid(axis="x", color=t["grid"], linewidth=1 * PT)
    ax.set_axisbelow(True)
    return fig, ax


def rounded_hbar(ax, y: float, value: float, thickness_px: float, color: str) -> None:
    """Horizontal bar from 0 to value, square at the baseline and 4px-rounded at the data end."""
    scale = ax.figure.dpi / 100
    x0, yc = ax.transData.transform((0, y))
    x1, _ = ax.transData.transform((value, y))
    h = thickness_px * scale / 2
    r = min(4 * scale, h, max(x1 - x0, 0))
    verts = [(x0, yc - h), (x1 - r, yc - h), (x1, yc - h), (x1, yc - h + r),
             (x1, yc + h - r), (x1, yc + h), (x1 - r, yc + h), (x0, yc + h), (x0, yc - h)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", transform=IdentityTransform()))


def legend(ax, t: dict, handles: list, **placement) -> None:
    """Legend outside the plot area so it never covers data (default: one row above the plot)."""
    placement = placement or {"loc": "lower left", "bbox_to_anchor": (0, 1.02), "ncol": len(handles)}
    ax.legend(handles=handles, frameon=False, fontsize=11 * PT, labelcolor=t["ink2"],
              handlelength=1.2, borderaxespad=0, columnspacing=1.6, **placement)


def save(fig, name: str, theme: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(CHARTS_DIR / f"{name}{suffix}.png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


# --- the four charts -------------------------------------------------------------


def chart_setups(rows: list[dict], t: dict, theme: str) -> None:
    """F1 for every setup: Aavaran runs in the accent colour, references in gray."""
    width, height = 800, 130 + 34 * len(rows)
    fig, ax = new_figure(
        t, width, height, "Federated training closes the gap to pooled data",
        "F1 score on the common test set (1.27M transactions)",
        [200 / width, 40 / height, 540 / width, (height - 130) / height],
    )
    ys = list(range(len(rows)))[::-1]
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(ys, [r["setup"] for r in rows], color=t["ink2"], fontsize=12 * PT)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    for y, r in zip(ys, rows):
        color = t["accent"] if r["kind"] == "federated" else t["reference"]
        rounded_hbar(ax, y, r["f1"], 16, color)
        ax.text(r["f1"] + 0.012, y, f"{r['f1']:.3f}", va="center", color=t["ink2"], fontsize=11 * PT)
    legend(ax, t, [Patch(color=t["accent"], label="Aavaran (federated)"),
                   Patch(color=t["reference"], label="Reference setups")])
    save(fig, "f1_by_setup", theme)


def chart_unseen(rows: list[dict], t: dict, theme: str) -> None:
    """Dumbbell: recall on the fraud type a bank never saw, alone vs federated."""
    by_setup = {r["setup"]: r for r in rows}
    fed = by_setup.get("Aavaran federated")
    if fed is None:
        return
    width, height = 800, 310
    fig, ax = new_figure(
        t, width, height, "Banks catch fraud they have never seen",
        "Recall on the fraud type missing from each bank's own data",
        [200 / width, 40 / height, 540 / width, 180 / height],
    )
    banks = list(UNSEEN)
    ys = list(range(len(banks)))[::-1]
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(banks) - 0.4)
    ax.set_yticks(ys, [f"Bank {b}: {UNSEEN[b]} fraud" for b in banks], color=t["ink2"], fontsize=12 * PT)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    for y, b in zip(ys, banks):
        key = f"recall_{UNSEEN[b]}"
        before, after = by_setup[f"Bank {b} alone"][key], fed[key]
        ax.plot([before, after], [y, y], color=t["axis"], linewidth=2 * PT, solid_capstyle="round", zorder=2)
        for x, c in ((before, t["before"]), (after, t["after"])):
            ax.plot(x, y, "o", ms=10 * PT, color=c, mec=t["surface"], mew=2 * PT, zorder=3, clip_on=False)
        ax.text(after + 0.025, y, f"{after:.0%}", va="center", color=t["ink"], fontsize=11 * PT)
        ax.text(before, y + 0.3, f"{before:.0%}", ha="center", color=t["muted"], fontsize=10 * PT)
    dot = dict(marker="o", linestyle="none", ms=8 * PT, mec=t["surface"], mew=2 * PT)
    legend(ax, t, [Line2D([], [], color=t["before"], label="Training alone", **dot),
                   Line2D([], [], color=t["after"], label="Aavaran federated", **dot)])
    save(fig, "unseen_fraud_recall", theme)


def chart_rounds(rows: list[dict], histories: dict, t: dict, theme: str) -> None:
    """PR-AUC of the global model after each round, one line per privacy level."""
    if not histories:
        return
    width, height = 800, 420
    order = [f"Aavaran + DP, ε = {e:g}" for e in sorted(DP_EPSILONS)] + ["Aavaran federated"]
    curves = {  # round 0 is the untrained model, so start from round 1
        name: [p for p in histories[name] if p["round"] > 0] for name in order if name in histories
    }
    lowest = min(p["pr_auc"] for h in curves.values() for p in h)
    y_min = max(0.0, (int(lowest * 50) - 1) / 50)
    fig, ax = new_figure(
        t, width, height, "Global model quality by federated round",
        "PR-AUC on the common test set after each round" + (" (y-axis starts above zero)" if y_min > 0 else ""),
        [70 / width, 50 / height, 560 / width, 290 / height],
    )
    ax.grid(axis="y", color=t["grid"], linewidth=1 * PT)
    ax.grid(axis="x", visible=False)
    labels = {"Aavaran federated": "No DP"} | {f"Aavaran + DP, ε = {e:g}": f"ε = {e:g}" for e in DP_EPSILONS}
    color_of = dict(zip(order, t["ramp"]))
    handles = []
    for name, h in curves.items():
        ax.plot([p["round"] for p in h], [p["pr_auc"] for p in h], color=color_of[name], linewidth=2 * PT,
                solid_joinstyle="round", solid_capstyle="round")
        handles.append(Line2D([], [], color=color_of[name], linewidth=2 * PT, label=labels[name]))
    central = next(r for r in rows if r["kind"] == "centralised")["pr_auc"]
    ax.axhline(central, color=t["muted"], linewidth=1 * PT, zorder=1)
    handles.insert(0, Line2D([], [], color=t["muted"], linewidth=1 * PT, label=f"Centralised {central:.3f}"))
    rounds = max(p["round"] for h in curves.values() for p in h)
    ax.set_xlim(1, rounds)
    ax.set_xticks([1] + list(range(5, rounds + 1, 5)))
    ax.set_ylim(y_min, 1.0)
    ax.set_xlabel("Round", color=t["muted"], fontsize=11 * PT)
    legend(ax, t, [handles[0]] + handles[1:][::-1], loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    save(fig, "rounds_pr_auc", theme)


def chart_privacy(rows: list[dict], t: dict, theme: str) -> None:
    """Privacy vs performance: final F1 by privacy budget, against the two reference setups."""
    fed = [r for r in rows if r["kind"] == "federated"]
    if len(fed) < 2:
        return
    fed.sort(key=lambda r: r["epsilon"] if r["epsilon"] is not None else float("inf"))
    width, height = 800, 400
    fig, ax = new_figure(
        t, width, height, "Privacy costs little accuracy",
        "F1 of the final global model by privacy budget ε per bank (smaller ε = stronger privacy)",
        [70 / width, 50 / height, 520 / width, 280 / height],
    )
    ax.grid(axis="y", color=t["grid"], linewidth=1 * PT)
    ax.grid(axis="x", visible=False)
    xs = list(range(len(fed)))
    ys = [r["f1"] for r in fed]
    ax.set_xticks(xs, [f"ε = {r['epsilon']:g}" if r["epsilon"] is not None else "No DP" for r in fed])
    ax.plot(xs, ys, color=t["accent"], linewidth=2 * PT, marker="o", ms=8 * PT, mec=t["surface"], mew=2 * PT)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center",
                    color=t["ink2"], fontsize=11 * PT)
    alone = sum(r["f1"] for r in rows if r["kind"] == "isolated") / sum(r["kind"] == "isolated" for r in rows)
    central = next(r for r in rows if r["kind"] == "centralised")["f1"]
    for value, text in ((central, "Pooled data"), (alone, "Each bank alone (avg)")):
        ax.axhline(value, color=t["muted"], linewidth=1 * PT, zorder=1)
        ax.annotate(f"{text}  {value:.3f}", (1, value), xycoords=("axes fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", va="center", color=t["ink2"], fontsize=10 * PT,
                    annotation_clip=False)
    ax.set_xlim(-0.3, len(fed) - 0.7)
    ax.set_ylim(0, 1.05)
    save(fig, "privacy_tradeoff", theme)


def write_table(rows: list[dict]) -> None:
    lines = [
        "# Aavaran results",
        "",
        "All setups are scored on the same held-out test set: 1,272,524 transactions, 1,642 of them fraud.",
        "Precision, recall and F1 use a 0.5 probability threshold; PR-AUC does not depend on a threshold.",
        "",
        "| Setup | Precision | Recall | F1 | PR-AUC | Recall TRANSFER | Recall CASH_OUT | False alarms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['setup']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['pr_auc']:.3f} "
            f"| {r['recall_TRANSFER']:.3f} | {r['recall_CASH_OUT']:.3f} | {r['false_alarms']:,} |"
        )
    lines += ["", "Banks 1 and 3 never saw CASH_OUT fraud; Banks 2 and 4 never saw TRANSFER fraud.", ""]
    for name in ("f1_by_setup", "unseen_fraud_recall", "rounds_pr_auc", "privacy_tradeoff"):
        if (CHARTS_DIR / f"{name}.png").exists():
            lines.append(f"![{name}](charts/{name}.png)")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    rows, histories = collect()
    for theme, t in THEMES.items():
        chart_setups(rows, t, theme)
        chart_unseen(rows, t, theme)
        chart_rounds(rows, histories, t, theme)
        chart_privacy(rows, t, theme)
    write_table(rows)
    width = max(len(r["setup"]) for r in rows)
    for r in rows:
        print(f"{r['setup']:<{width}}  F1={r['f1']:.3f}  PR-AUC={r['pr_auc']:.3f}  "
              f"recall TRANSFER={r['recall_TRANSFER']:.3f} CASH_OUT={r['recall_CASH_OUT']:.3f}")
    print(f"\nWrote {RESULTS_DIR / 'comparison.md'} and charts in {CHARTS_DIR}")


if __name__ == "__main__":
    main()
