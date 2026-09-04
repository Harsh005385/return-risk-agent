"""Generate docs/architecture.png for the Return Risk Agent."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture.png"


def _box(ax, xy, text, fc="#e8f4fc", ec="#333"):
    x, y, w, h = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Return Risk Agent - Architecture", fontsize=14, fontweight="bold")

    _box(ax, (0.3, 5.5, 2.2, 1), "Case Input\n(customer, order,\nfeatures)", "#fff3cd")
    _box(ax, (3.0, 5.2, 2.8, 1.4), "Agent Orchestrator\n• Fast: parallel tools\n• Deep: sequential ≤4 iter", "#d1ecf1")
    _box(ax, (6.3, 4.8, 2.6, 2.0), "Tools\nhistory · score · SHAP\npattern · order ctx\nshared-signal graph", "#d4edda")
    _box(ax, (9.3, 5.5, 2.2, 1), "Policy Engine\n(deterministic)", "#f8d7da")
    _box(ax, (9.3, 3.8, 2.2, 1), "Action\nALLOW / MONITOR\nREVIEW / HOLD", "#e2e3e5")
    _box(ax, (3.0, 2.5, 3.0, 1), "Audit DB\n(hash-chained SQLite)", "#cfe2ff")
    _box(ax, (6.8, 2.5, 2.8, 1), "Dashboard / API\n+ SHAP + graphs", "#e7d4f5")

    arrows = [
        ((2.5, 6.0), (3.0, 6.0)),
        ((5.8, 6.0), (6.3, 5.8)),
        ((8.9, 5.8), (9.3, 6.0)),
        ((10.4, 5.5), (10.4, 4.8)),
        ((4.5, 5.2), (4.5, 3.5)),
        ((7.5, 4.8), (7.5, 3.5)),
        ((6.0, 3.0), (6.8, 3.0)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
