"""
panel_omic_pathways.py
======================
面板 3：Omic 通路重要性排序图

输入：
  pathway_scores  np.ndarray [n_pathways]  float32
  pathway_names   list[str]                长度 = n_pathways

输出：
  一张 PNG，水平柱状图，Top-N 通路按重要性降序排列
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def save_panel(
    pathway_scores,
    pathway_names: list,
    save_path: str,
    top_n: int = 20,
    cmap: str = "RdYlGn_r",
    figsize: tuple = (9, 7),
    dpi: int = 200,
) -> bool:
    """
    生成通路重要性面板。

    Parameters
    ----------
    pathway_scores  : float32 [n_pathways]，值越大越重要
    pathway_names   : 通路名称列表，与 pathway_scores 对齐
    save_path       : 输出 PNG 路径
    top_n           : 展示前 N 条通路
    cmap            : colormap（颜色越深 = 分数越高）
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if pathway_scores is not None and pathway_names is not None \
            and len(pathway_scores) > 0:

        scores = np.array(pathway_scores, dtype=np.float32)
        names  = list(pathway_names)

        # 截齐
        n = min(len(scores), len(names))
        scores, names = scores[:n], names[:n]

        # Top-N
        k = min(top_n, n)
        top_idx = np.argsort(scores)[-k:][::-1]  # 降序
        top_scores = scores[top_idx]
        top_names  = [_truncate(names[i], 45) for i in top_idx]

        # 颜色映射（归一化到 [0,1]）
        norm_s = (top_scores - top_scores.min()) / \
                 max(top_scores.max() - top_scores.min(), 1e-8)
        color_map = plt.get_cmap(cmap)
        bar_colors = [color_map(v) for v in norm_s]

        y_pos = np.arange(k)[::-1]  # 最高分在顶部
        bars = ax.barh(y_pos, top_scores,
                       color=bar_colors, edgecolor="white",
                       linewidth=0.5, height=0.72)

        # 分数标签
        for bar, score in zip(bars, top_scores):
            ax.text(bar.get_width() + top_scores.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}",
                    va="center", fontsize=7.5, color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("Pathway Importance Score", fontsize=10)
        ax.set_xlim(0, top_scores.max() * 1.18)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Top-{k} Pathway Importance (Omic)",
                     fontsize=13, fontweight="bold", pad=10)

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=color_map,
            norm=plt.Normalize(vmin=top_scores.min(), vmax=top_scores.max())
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Relative Importance", rotation=270,
                       labelpad=13, fontsize=9)

        success = True

    else:
        missing = []
        if pathway_scores is None:  missing.append("pathway_scores")
        if pathway_names is None:   missing.append("pathway_names")
        ax.text(0.5, 0.5,
                f"Pathway Importance Unavailable\nMissing: {', '.join(missing)}",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.set_title("Pathway Importance", fontsize=13, fontweight="bold")
        ax.axis("off")
        success = False

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    status = "✅" if success else "⚠️"
    print(f"  {status} [Omic pathways] → {os.path.basename(save_path)}")
    return success


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"
