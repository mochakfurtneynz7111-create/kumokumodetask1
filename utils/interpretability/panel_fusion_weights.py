"""
panel_fusion_weights.py
=======================
面板 2：多模态融合权重柱状图

输入：
  fusion_weights  np.ndarray [3]  (path, omic, text)  float32
  pred_label      int
  true_label      int（可选）
  class_names     list[str]

输出：
  一张 PNG，只含柱状图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def save_panel(
    fusion_weights,
    save_path: str,
    pred_label: int = None,
    true_label: int = None,
    class_names: list = None,
    modality_names: list = None,
    figsize: tuple = (6, 5),
    dpi: int = 200,
) -> bool:
    """
    生成融合权重柱状图面板。

    Parameters
    ----------
    fusion_weights  : float32 [3]  三模态权重（path / omic / text）
                      传 None 时保存占位图
    save_path       : 输出 PNG 路径
    pred_label      : 预测标签索引
    true_label      : 真实标签索引（用于标注正确/错误）
    class_names     : 类别名称列表
    modality_names  : 模态名称，默认 ["Pathology", "Genomics", "Text"]
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    modality_names = modality_names or ["Pathology", "Genomics", "Text"]
    colors = ["#E07B6A", "#5BA8A0", "#7DAECC"]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # ── 有数据 ───────────────────────────────────────────────────
    if fusion_weights is not None and len(fusion_weights) >= len(modality_names):
        weights = np.array(fusion_weights[:len(modality_names)], dtype=np.float32)

        bars = ax.bar(modality_names, weights,
                      color=colors[:len(modality_names)],
                      edgecolor="black", linewidth=1.0,
                      width=0.55, alpha=0.88)

        # 数值标签
        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{w:.3f}",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_ylim(0, min(1.15, weights.max() * 1.35 + 0.05))
        ax.set_ylabel("Fusion Weight", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        # 副标题：预测结果
        subtitle = _build_subtitle(pred_label, true_label, class_names)
        if subtitle:
            ax.set_xlabel(subtitle, fontsize=10)

        ax.set_title("Multimodal Fusion Weights", fontsize=13, fontweight="bold", pad=10)
        success = True

    # ── 缺少数据 ─────────────────────────────────────────────────
    else:
        ax.text(0.5, 0.5, "Fusion Weights Unavailable",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.set_title("Multimodal Fusion Weights", fontsize=13, fontweight="bold")
        ax.axis("off")
        success = False

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    status = "✅" if success else "⚠️"
    print(f"  {status} [Fusion weights] → {os.path.basename(save_path)}")
    return success


def _build_subtitle(pred_label, true_label, class_names):
    if pred_label is None:
        return ""
    cn = class_names or []

    def name(idx):
        return cn[idx] if cn and idx < len(cn) else f"Class {idx}"

    pred_str = f"Pred: {name(pred_label)}"
    if true_label is not None:
        correct = "✓" if pred_label == true_label else "✗"
        return f"True: {name(true_label)}   {pred_str}  {correct}"
    return pred_str
