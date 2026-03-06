"""
panel_patch_pathway.py
======================
面板 6：Patch ↔ Pathway 对应关系热力图

输入：
  patch_pathway_map  np.ndarray [n_patches, n_pathways]  float32
  pathway_names      list[str]

输出：
  一张 PNG，Top-K patch × Top-K pathway 的热力图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


def save_panel(
    patch_pathway_map,
    pathway_names: list,
    save_path: str,
    top_k_patches: int = 15,
    top_k_pathways: int = 12,
    cmap: str = "YlOrRd",
    annot: bool = False,
    figsize: tuple = (13, 6),
    dpi: int = 200,
) -> bool:
    """
    生成 Patch-Pathway 对应热力图面板。

    Parameters
    ----------
    patch_pathway_map : float32 [n_patches, n_pathways]
    pathway_names     : 通路名称列表
    save_path         : 输出 PNG 路径
    top_k_patches     : 行数（Top-K patches）
    top_k_pathways    : 列数（Top-K pathways）
    annot             : 是否在格子内显示数值
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if patch_pathway_map is not None and pathway_names is not None \
            and len(pathway_names) > 0:

        ppm = np.array(patch_pathway_map, dtype=np.float32)   # [N_p, N_pw]
        n_patches, n_pathways = ppm.shape

        # 全局 Top-K pathways（按列均值）
        kpw = min(top_k_pathways, n_pathways)
        top_pw_idx = np.argsort(ppm.mean(axis=0))[-kpw:][::-1]

        # 全局 Top-K patches（按行均值）
        kp = min(top_k_patches, n_patches)
        top_p_idx = np.argsort(ppm.mean(axis=1))[-kp:][::-1]

        sub = ppm[top_p_idx][:, top_pw_idx]                   # [kp, kpw]

        pw_labels = [_truncate(pathway_names[i], 22) for i in top_pw_idx]
        p_labels  = [f"Patch {i:04d}" for i in top_p_idx]

        if _HAS_SNS:
            import seaborn as sns
            sns.heatmap(
                sub, ax=ax,
                cmap=cmap,
                vmin=0, vmax=sub.max() if sub.max() > 0 else 1,
                xticklabels=pw_labels,
                yticklabels=p_labels,
                annot=annot,
                fmt=".2f" if annot else "",
                linewidths=0.3,
                linecolor="#dddddd",
                cbar_kws={"label": "Correspondence Score",
                           "shrink": 0.8},
            )
        else:
            im = ax.imshow(sub, cmap=cmap, aspect="auto",
                           vmin=0, vmax=sub.max() if sub.max() > 0 else 1,
                           interpolation="nearest")
            ax.set_xticks(range(kpw))
            ax.set_xticklabels(pw_labels, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(kp))
            ax.set_yticklabels(p_labels, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                         label="Correspondence Score")

        ax.tick_params(axis="x", labelsize=7.5, rotation=38)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("Pathways", fontsize=10)
        ax.set_ylabel("Patches", fontsize=10)
        ax.set_title(
            f"Patch ↔ Pathway Correspondence  "
            f"(Top-{kp} patches × Top-{kpw} pathways)",
            fontsize=12, fontweight="bold", pad=10
        )
        success = True

    else:
        missing = []
        if patch_pathway_map is None: missing.append("patch_pathway_map")
        if pathway_names is None:     missing.append("pathway_names")
        ax.text(0.5, 0.5,
                f"Patch-Pathway Map Unavailable\nMissing: {', '.join(missing)}",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.set_title("Patch ↔ Pathway Correspondence",
                     fontsize=13, fontweight="bold")
        ax.axis("off")
        success = False

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    status = "✅" if success else "⚠️"
    print(f"  {status} [Patch↔Pathway] → {os.path.basename(save_path)}")
    return success


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"
