"""
panel_qa_patch.py
=================
面板 5：QA → Patch 空间注意力（文本中心）

输入：
  qa2patch_attn   np.ndarray [6, n_patches]  float32
  qa_texts        list[str]  len=6
  thumbnail       np.ndarray [H, W, 3]        uint8（可选，有则叠加热力图）
  coords          np.ndarray [N, 2]           float32（可选）

输出：
  一张 PNG，6 个子图（每个 QA 一张小 WSI attention 图）
  无 thumbnail 时退化为条形图（Top-K patch index）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def save_panel(
    qa2patch_attn,
    qa_texts: list,
    save_path: str,
    thumbnail=None,
    coords=None,
    patch_size: int = 256,
    top_k_patches: int = 5,
    cmap: str = "hot",
    figsize: tuple = (14, 9),
    dpi: int = 200,
) -> bool:
    """
    生成 QA→Patch 空间注意力面板（6 个 QA，每个一张子图）。

    Parameters
    ----------
    qa2patch_attn   : float32 [6, n_patches]
    qa_texts        : 6 个 QA 文本
    save_path       : 输出 PNG 路径
    thumbnail       : WSI 缩略图（可选）
    coords          : Patch 坐标（可选）
    top_k_patches   : 无 WSI 时展示 Top-K 柱状图
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    if qa2patch_attn is None:
        _save_placeholder(save_path, "QA→Patch Spatial Attention Unavailable\n"
                                     "Missing: qa2patch_attn", figsize, dpi)
        print(f"  ⚠️ [QA→Patch] → {os.path.basename(save_path)}")
        return False

    attn = np.array(qa2patch_attn, dtype=np.float32)   # [6, N]
    n_qa, n_patches = attn.shape
    n_qa = min(n_qa, 6)

    qa_labels = [qa_texts[i] if qa_texts and i < len(qa_texts)
                 else f"QA {i+1}" for i in range(n_qa)]

    has_wsi = (thumbnail is not None and coords is not None
               and len(coords) == n_patches)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    for i in range(n_qa):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        scores_i = attn[i]          # [N]

        if has_wsi:
            _draw_wsi_attention(ax, thumbnail, coords, scores_i,
                                patch_size, cmap, top_k_patches)
        else:
            _draw_bar_attention(ax, scores_i, top_k_patches)

        short_q = _short_qa(qa_labels[i])
        ax.set_title(short_q, fontsize=8.5, fontweight="bold",
                     pad=4, wrap=True)

    fig.suptitle("Text-Centric: QA → Patch Spatial Attention",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ [QA→Patch] → {os.path.basename(save_path)}")
    return True


# ─────────────────────────────────────────────────────────────────
# 绘图工具
# ─────────────────────────────────────────────────────────────────

def _draw_wsi_attention(ax, thumbnail, coords, scores, patch_size, cmap, top_k):
    """在 WSI 缩略图上叠加该 QA 的 patch attention"""
    thumb_h, thumb_w = thumbnail.shape[:2]
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    roi_w = coords[:, 0].max() + patch_size - min_x
    roi_h = coords[:, 1].max() + patch_size - min_y
    if roi_w <= 0 or roi_h <= 0:
        ax.axis("off"); return
    sx = thumb_w / roi_w
    sy = thumb_h / roi_h

    # 归一化
    s = scores.copy()
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min)
    else:
        s = np.ones_like(s) * 0.5

    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    cnt     = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    for (x0, y0), sc in zip(coords, s):
        xs = int((x0 - min_x) * sx);      ys = int((y0 - min_y) * sy)
        xe = int((x0 - min_x + patch_size) * sx)
        ye = int((y0 - min_y + patch_size) * sy)
        xs = max(0, min(xs, thumb_w-1)); ys = max(0, min(ys, thumb_h-1))
        xe = max(xs+1, min(xe, thumb_w)); ye = max(ys+1, min(ye, thumb_h))
        heatmap[ys:ye, xs:xe] += sc
        cnt[ys:ye, xs:xe]     += 1
    mask = cnt > 0
    heatmap[mask] /= cnt[mask]

    ax.imshow(thumbnail)
    if heatmap.max() > 0:
        ax.imshow(heatmap, cmap=cmap, alpha=0.6, vmin=0, vmax=1,
                  interpolation="nearest")

    # Top-K 框
    k = min(top_k, len(scores))
    for rank, idx in enumerate(np.argsort(scores)[-k:][::-1]):
        x0, y0 = coords[idx]
        xt = (x0 - min_x) * sx;   yt = (y0 - min_y) * sy
        wt = patch_size * sx;      ht = patch_size * sy
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((xt, yt), wt, ht,
                                fill=False, edgecolor="cyan",
                                linewidth=1.2))
        ax.text(xt + wt/2, yt - 3, f"#{rank+1}",
                color="cyan", fontsize=6, ha="center", va="bottom",
                fontweight="bold")
    ax.axis("off")


def _draw_bar_attention(ax, scores, top_k):
    """无 WSI 时展示 Top-K patch 柱状图"""
    k = min(top_k, len(scores))
    top_idx = np.argsort(scores)[-k:][::-1]
    top_scores = scores[top_idx]
    ax.bar(range(k), top_scores,
           color="#5BA8A0", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"P{i}" for i in top_idx], fontsize=7, rotation=30)
    ax.set_ylabel("Attention", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def _save_placeholder(save_path, msg, figsize, dpi):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=12, color="#888888", transform=ax.transAxes)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _short_qa(text: str, max_chars: int = 50) -> str:
    return text if len(text) <= max_chars else text[:max_chars-1] + "…"
