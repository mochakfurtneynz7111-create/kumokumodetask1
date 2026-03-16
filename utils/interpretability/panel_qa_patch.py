"""
panel_qa_patch.py
=================
面板 5：QA → Patch 空间注意力热力图

背景重建方式与 panel_wsi_heatmap.py 完全一致：
  1. 通过 patch PNG 文件拼接重建 WSI 背景（最精准）
  2. 若 patch_png_dir 不可用，降级为 thumbnail resize
  3. 每个 QA 独立归一化 + 高斯平滑，叠加在同一背景上
  4. Top-K 黄框标注最高注意力区域

布局：2行 × 3列，共 6 个 QA 子图 + 右侧共享 colorbar
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def save_panel(
    qa2patch_attn,              # np.ndarray [6, n_patches]  float32
    qa_texts: list,             # list[str]  len=6
    save_path: str,
    thumbnail=None,             # np.ndarray [H, W, 3]  uint8  (降级备用)
    coords=None,                # np.ndarray [N, 2]  float32  原始 WSI 坐标
    slide_id: str = "",
    patch_png_dir: str = None,  # patch PNG 目录，结构：{dir}/{slide_id}/{x}x_{y}y.png
    patch_size: int = 256,
    top_k: int = 3,
    cmap: str = "jet",
    alpha: float = 0.55,
    norm_mode: str = "rank",    # "rank" | "percentile" | "linear"
    smooth_level: int = 1,      # 0=无平滑 1=轻微 2=较强
    figsize: tuple = (21, 14),
    dpi: int = 150,
) -> bool:

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True,
    )

    # ── 基本检查 ─────────────────────────────────────────────────
    if qa2patch_attn is None:
        _save_placeholder(save_path, figsize, dpi,
                          "QA→Patch Heatmap Unavailable\nMissing: qa2patch_attn")
        print(f"  ⚠️ [QA→Patch] qa2patch_attn is None")
        return False

    if coords is None or len(coords) == 0:
        _save_placeholder(save_path, figsize, dpi,
                          "QA→Patch Heatmap Unavailable\nMissing: coords")
        print(f"  ⚠️ [QA→Patch] coords is None or empty")
        return False

    attn_all = np.array(qa2patch_attn, dtype=np.float32)   # [6, N]
    n_qa, n_patches = attn_all.shape
    n_qa = min(n_qa, 6)
    coords = np.array(coords, dtype=np.float32)

    if len(coords) != n_patches:
        print(f"  ⚠️ [QA→Patch] coords len {len(coords)} != n_patches {n_patches}")
        _save_placeholder(save_path, figsize, dpi,
                          f"coords len mismatch: {len(coords)} vs {n_patches}")
        return False

    # ── 自动检测 patch stride ─────────────────────────────────────
    stride = _detect_stride(coords)
    if stride:
        patch_size = stride
        print(f"  📐 [QA→Patch] Auto stride = {patch_size}px")

    # ── 坐标系（所有 QA 共享，保证完全对齐）─────────────────────
    x_min = int(coords[:, 0].min())
    y_min = int(coords[:, 1].min())
    x_max = int(coords[:, 0].max()) + patch_size
    y_max = int(coords[:, 1].max()) + patch_size
    wsi_w = x_max - x_min
    wsi_h = y_max - y_min

    max_side = 800
    scale    = min(max_side / wsi_w, max_side / wsi_h)
    canvas_w = int(wsi_w * scale)
    canvas_h = int(wsi_h * scale)
    pw = max(1, int(patch_size * scale))
    ph = max(1, int(patch_size * scale))
    print(f"  📐 [QA→Patch] Canvas: {canvas_w}x{canvas_h}  patch_px: {pw}x{ph}")

    # ── 背景图：patch PNG 拼接（唯一调用，6个子图共享）──────────
    bg = _build_bg_from_patches(
        slide_id, coords, scale, x_min, y_min,
        canvas_w, canvas_h, pw, ph, patch_png_dir,
    )
    if bg is None:
        if thumbnail is not None:
            bg = np.array(
                Image.fromarray(thumbnail).resize((canvas_w, canvas_h), Image.BILINEAR),
                dtype=np.uint8,
            )
            print(f"  ⚠️  [QA→Patch] Patch PNG unavailable, using thumbnail")
        else:
            bg = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
            print(f"  ⚠️  [QA→Patch] No background, using gray canvas")

    # ── 布局 ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(
        2, 3, figure=fig,
        hspace=0.30, wspace=0.06,
        left=0.03, right=0.87, top=0.90, bottom=0.04,
    )

    norm_obj = plt.Normalize(0, 1)
    sm_obj   = cm.ScalarMappable(norm=norm_obj, cmap=cmap)

    qa_labels = [
        qa_texts[i] if qa_texts and i < len(qa_texts) else f"QA {i+1}"
        for i in range(n_qa)
    ]

    for i in range(n_qa):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        scores_i = attn_all[i]   # [N]

        # 每个 QA 独立归一化
        attn_norm = _normalize_attention(scores_i, norm_mode)

        # ── 构建马赛克热力图 canvas ───────────────────────────────
        raw_heatmap = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
        for (x, y), score in zip(coords, attn_norm):
            xs = int((x - x_min) * scale)
            ys = int((y - y_min) * scale)
            xe = min(xs + pw, canvas_w)
            ye = min(ys + ph, canvas_h)
            xs = max(0, xs)
            ys = max(0, ys)
            raw_heatmap[ys:ye, xs:xe] = np.maximum(
                raw_heatmap[ys:ye, xs:xe], score
            )

        smooth_heatmap = _smooth_heatmap(raw_heatmap, pw, smooth_level)

        # ── 绘制 ─────────────────────────────────────────────────
        ax.imshow(bg)
        masked = np.ma.masked_where(smooth_heatmap < 0, smooth_heatmap)
        ax.imshow(
            masked,
            cmap=cmap,
            alpha=alpha,
            norm=norm_obj,
            interpolation="none",
            extent=[0, canvas_w, canvas_h, 0],
        )

        _annotate_top_k(
            ax, coords, scores_i, patch_size,
            scale, x_min, y_min, pw, ph, top_k,
        )

        ax.axis("off")
        short_q = _short_qa(qa_labels[i], max_words=8)
        ax.set_title(short_q, fontsize=9, fontweight="bold", pad=5)

    # ── 右侧共享 colorbar ────────────────────────────────────────
    cbar_ax = fig.add_axes([0.89, 0.10, 0.015, 0.75])
    cbar = fig.colorbar(sm_obj, cax=cbar_ax)
    cbar.set_label(f"Attention ({norm_mode})", rotation=270,
                   labelpad=16, fontsize=10)

    fig.suptitle(
        "Text-Centric: QA -> Patch Spatial Attention",
        fontsize=14, fontweight="bold", y=0.95,
    )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ [QA->Patch] -> {os.path.basename(save_path)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 背景重建（逻辑与 panel_wsi_heatmap._build_bg_from_patches 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

def _build_bg_from_patches(slide_id, coords, scale, x_min, y_min,
                            canvas_w, canvas_h, pw, ph, patch_png_dir):
    if not patch_png_dir:
        return None

    base = slide_id[:-4] if slide_id.endswith(".svs") else slide_id
    slide_dir = None
    for name in [slide_id, base]:
        for subdir in ["", "output"]:
            d = os.path.join(patch_png_dir, subdir, name) if subdir \
                else os.path.join(patch_png_dir, name)
            if os.path.isdir(d):
                slide_dir = d
                break
        if slide_dir:
            break

    if not slide_dir:
        print(f"  ⚠️  [QA->Patch] Patch dir not found under: {patch_png_dir}")
        return None

    patch_files = [f for f in os.listdir(slide_dir) if f.lower().endswith(".png")]
    coord_map = {}
    for fn in patch_files:
        m = re.match(r"(\d+)x_(\d+)y", fn)
        if m:
            coord_map[(int(m.group(1)), int(m.group(2)))] = \
                os.path.join(slide_dir, fn)

    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    placed = 0
    for (x, y) in coords:
        xi, yi = int(x), int(y)
        fpath = coord_map.get((xi, yi))
        # 容差搜索（±4px）
        if fpath is None:
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    fpath = coord_map.get((xi + dx, yi + dy))
                    if fpath:
                        break
                if fpath:
                    break
        if not fpath:
            continue
        try:
            img = Image.open(fpath).convert("RGB").resize((pw, ph), Image.BILINEAR)
            cx  = int((x - x_min) * scale)
            cy  = int((y - y_min) * scale)
            cx2 = min(cx + pw, canvas_w)
            cy2 = min(cy + ph, canvas_h)
            canvas[cy:cy2, cx:cx2] = np.array(img)[:cy2 - cy, :cx2 - cx]
            placed += 1
        except Exception:
            continue

    print(f"  ✅ [QA->Patch] Background: {placed}/{len(coords)} patches placed")
    return canvas if placed > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数（与 panel_wsi_heatmap.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

def _detect_stride(coords: np.ndarray):
    strides = []
    for axis in [0, 1]:
        vals = np.unique(np.sort(coords[:, axis]))
        if len(vals) >= 2:
            diffs = np.diff(vals)
            diffs = diffs[diffs > 10]
            if len(diffs) > 0:
                strides.append(int(np.percentile(diffs, 10)))
    if not strides:
        return None
    s = min(strides)
    return s if 64 <= s <= 8192 else None


def _normalize_attention(scores: np.ndarray, mode: str) -> np.ndarray:
    a = scores.copy().astype(np.float32)
    if mode == "rank":
        ranks = np.argsort(np.argsort(a)).astype(np.float32)
        return ranks / max(len(ranks) - 1, 1)
    elif mode == "percentile":
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        if hi > lo:
            return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return np.ones_like(a) * 0.5
    else:   # linear
        lo, hi = a.min(), a.max()
        if hi > lo:
            return (a - lo) / (hi - lo)
        return np.ones_like(a) * 0.5


def _smooth_heatmap(raw: np.ndarray, pw: int, level: int) -> np.ndarray:
    if level == 0:
        return raw
    sigma = pw * (0.4 if level == 1 else 0.8)
    sigma = max(0.8, sigma)
    filled = raw.copy()
    filled[filled < 0] = 0.0
    smoothed = gaussian_filter(filled, sigma=sigma)
    coverage = gaussian_filter((raw >= 0).astype(np.float32), sigma=sigma * 0.6)
    return np.where(coverage > 0.05, smoothed, -1.0)


def _annotate_top_k(ax, coords, attention_scores, patch_size,
                     scale, x_min, y_min, pw, ph, top_k):
    k = min(top_k, len(attention_scores))
    for rank, idx in enumerate(np.argsort(attention_scores)[-k:][::-1]):
        x, y = coords[idx]
        xt = (x - x_min) * scale
        yt = (y - y_min) * scale
        rect = Rectangle(
            (xt, yt), pw, ph,
            fill=False, edgecolor="yellow", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            xt + pw / 2, yt - 3,
            f"#{rank + 1}\n{attention_scores[idx]:.3f}",
            color="yellow", fontsize=6, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="black", alpha=0.6,
                edgecolor="yellow", linewidth=0.7,
            ),
        )


def _short_qa(text: str, max_words: int = 8) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _save_placeholder(save_path, figsize, dpi, msg):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=13, color="#888", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
