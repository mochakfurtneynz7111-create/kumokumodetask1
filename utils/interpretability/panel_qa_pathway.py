"""
panel_qa_pathway.py
===================
面板 4：QA → Pathway 注意力热力图（文本中心）

输入：
  qa2pathway_attn  np.ndarray [6, n_pathways]  float32
  qa_texts         list[str]  len=6
  pathway_names    list[str]  len=n_pathways

输出：
  一张 PNG，热力图：行=QA，列=Top-K 通路
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def save_panel(
    qa2pathway_attn,
    qa_texts: list,
    pathway_names: list,
    save_path: str,
    top_k_pathways: int = 15,
    cmap: str = "YlOrRd",
    figsize: tuple = (14, 5),
    dpi: int = 200,
) -> bool:
    """
    生成 QA→Pathway 注意力热力图面板。

    Parameters
    ----------
    qa2pathway_attn : float32 [6, n_pathways]
    qa_texts        : 6 个 QA 问题文本
    pathway_names   : 通路名称列表
    save_path       : 输出 PNG 路径
    top_k_pathways  : 展示全局 Top-K 通路（列）
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if qa2pathway_attn is not None and pathway_names is not None \
            and len(pathway_names) > 0:

        attn = np.array(qa2pathway_attn, dtype=np.float32)    # [6, n_pw]
        n_qa, n_pw = attn.shape

        qa_labels = [_short_qa(qa_texts[i] if qa_texts and i < len(qa_texts)
                               else f"QA {i+1}")
                     for i in range(n_qa)]

        # 取全局 Top-K 通路（按各QA均值）
        global_scores = attn.mean(axis=0)                     # [n_pw]
        k = min(top_k_pathways, n_pw)
        top_pw_idx = np.argsort(global_scores)[-k:][::-1]     # 降序

        sub_attn  = attn[:, top_pw_idx]                       # [6, k]
        pw_labels = [_truncate(pathway_names[i], 25)
                     for i in top_pw_idx]

        # 行归一化，使每个 QA 的分布可比
        row_max = sub_attn.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        sub_attn_norm = sub_attn / row_max

        im = ax.imshow(sub_attn_norm, cmap=cmap, aspect="auto",
                       vmin=0, vmax=1, interpolation="nearest")

        # 轴标签
        ax.set_xticks(range(k))
        ax.set_xticklabels(pw_labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(n_qa))
        ax.set_yticklabels(qa_labels, fontsize=8.5)

        # 数值注释（仅当格子数不太多时）
        if n_qa * k <= 120:
            for r in range(n_qa):
                for c in range(k):
                    val = sub_attn_norm[r, c]
                    text_color = "white" if val > 0.6 else "black"
                    ax.text(c, r, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=6, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Normalized Attention", rotation=270,
                       labelpad=13, fontsize=9)

        ax.set_xlabel(f"Top-{k} Pathways (by mean QA attention)", fontsize=10)
        ax.set_ylabel("QA Anchors", fontsize=10)
        ax.set_title("Text-Centric: QA → Pathway Attention",
                     fontsize=13, fontweight="bold", pad=10)
        success = True

    else:
        missing = []
        if qa2pathway_attn is None: missing.append("qa2pathway_attn")
        if pathway_names is None:   missing.append("pathway_names")
        ax.text(0.5, 0.5,
                f"QA→Pathway Heatmap Unavailable\nMissing: {', '.join(missing)}",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.set_title("QA → Pathway Attention", fontsize=13, fontweight="bold")
        ax.axis("off")
        success = False

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    status = "✅" if success else "⚠️"
    print(f"  {status} [QA→Pathway] → {os.path.basename(save_path)}")
    return success


def _short_qa(text: str, max_words: int = 7) -> str:
    """截取前几个单词作为简短标签"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"
