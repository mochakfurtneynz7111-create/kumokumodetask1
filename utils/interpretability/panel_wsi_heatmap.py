"""
panel_wsi_heatmap.py  —  三联图：原图 | 热力图叠加 | 纯热力图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from PIL import Image
import os, re


def save_panel(
    thumbnail: np.ndarray,
    coords: np.ndarray,
    attention_scores: np.ndarray,
    save_path: str,
    slide_id: str = "",
    patch_size: int = 256,
    top_k: int = 5,
    cmap: str = "jet",
    alpha: float = 0.55,
    figsize: tuple = (24, 8),
    dpi: int = 150,
    patch_png_dir: str = None,
    # ── 归一化方式 ────────────────────────────────────────────
    # "rank"      排名归一化（颜色均匀，推荐）
    # "percentile" 百分位裁剪（保留相对大小，高分区更突出）
    # "linear"    原始线性（忠实原始分布，可能大片蓝色）
    norm_mode: str = "rank",
    # ── 平滑强度（0 = 不平滑，纯马赛克；1 = 轻微；2 = 较强）──
    smooth_level: int = 1,
) -> bool:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    if coords is None or attention_scores is None or len(coords) == 0:
        _save_placeholder(save_path, figsize, "Missing coords or attention_scores")
        return False

    n = min(len(coords), len(attention_scores))
    coords            = coords[:n].astype(np.float32)
    attention_scores  = attention_scores[:n].astype(np.float32)

    # ── 自动检测 stride ──────────────────────────────────────────
    stride = _detect_stride(coords)
    if stride:
        patch_size = stride
        print(f"  📐 Auto stride = {patch_size}px")

    # ── 归一化 ──────────────────────────────────────────────────
    attn = _normalize_attention(attention_scores, norm_mode)
    raw = attention_scores.copy().astype(np.float32)
    print(f"  📊 Raw attention   : min={raw.min():.6f}  max={raw.max():.6f}  "
          f"mean={raw.mean():.6f}  sum={raw.sum():.4f}")
    print(f"  📊 Top-5 raw scores: {sorted(raw, reverse=True)[:5]}")
    print(f"  📊 Bot-5 raw scores: {sorted(raw)[:5]}")
    print(f"  📊 After {norm_mode}: min={attn.min():.3f}  max={attn.max():.3f}  mean={attn.mean():.3f}")

    # ── 坐标系 ───────────────────────────────────────────────────
    x_min = int(coords[:, 0].min())
    y_min = int(coords[:, 1].min())
    x_max = int(coords[:, 0].max()) + patch_size
    y_max = int(coords[:, 1].max()) + patch_size
    wsi_w, wsi_h = x_max - x_min, y_max - y_min

    max_side = 1024
    scale    = min(max_side / wsi_w, max_side / wsi_h)
    canvas_w = int(wsi_w * scale)
    canvas_h = int(wsi_h * scale)
    pw = max(1, int(patch_size * scale))
    ph = max(1, int(patch_size * scale))
    print(f"  📐 Canvas: {canvas_w}×{canvas_h}  patch_px: {pw}×{ph}")

    # ── 背景图（patch PNG 拼接）──────────────────────────────────
    bg = _build_bg_from_patches(
        slide_id, coords, scale, x_min, y_min,
        canvas_w, canvas_h, pw, ph, patch_png_dir
    )
    if bg is None:
        if thumbnail is not None:
            bg = np.array(Image.fromarray(thumbnail).resize(
                (canvas_w, canvas_h), Image.BILINEAR))
            print("  ⚠️  Using thumbnail as background")
        else:
            bg = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)

    # ── 马赛克热力图（无平滑版，用于计算） ─────────────────────
    raw_heatmap = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    for (x, y), score in zip(coords, attn):
        xs = int((x - x_min) * scale);  ys = int((y - y_min) * scale)
        xe = min(xs + pw, canvas_w);     ye = min(ys + ph, canvas_h)
        xs = max(0, xs);                 ys = max(0, ys)
        raw_heatmap[ys:ye, xs:xe] = np.maximum(raw_heatmap[ys:ye, xs:xe], score)

    # ── 平滑 ─────────────────────────────────────────────────────
    smooth_heatmap = _smooth_heatmap(raw_heatmap, pw, smooth_level)

    norm   = plt.Normalize(0, 1)
    sm_obj = cm.ScalarMappable(norm=norm, cmap=cmap)

    # ── 绘三联图 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")

    titles = ["Original", "Heatmap Overlay", "Attention Only"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

    short_id = (slide_id[:55] + "...") if len(slide_id) > 55 else slide_id
    fig.suptitle(f"WSI Attention Heatmap\n{short_id}",
                 fontsize=11, y=1.01)

    # 面板 1：原图
    axes[0].imshow(bg)

    # 面板 2：叠加
    axes[1].imshow(bg)
    masked2 = np.ma.masked_where(smooth_heatmap < 0, smooth_heatmap)
    axes[1].imshow(masked2, cmap=cmap, alpha=alpha, norm=norm,
                   interpolation="none",
                   extent=[0, canvas_w, canvas_h, 0])
    _annotate_top_k(axes[1], coords, attention_scores, patch_size,
                    scale, x_min, y_min, pw, ph, top_k)

    # 面板 3：纯热力图（白底）
    axes[2].set_facecolor("white")
    pure_bg = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    axes[2].imshow(pure_bg)
    masked3 = np.ma.masked_where(smooth_heatmap < 0, smooth_heatmap)
    im3 = axes[2].imshow(masked3, cmap=cmap, alpha=1.0, norm=norm,
                          interpolation="none",
                          extent=[0, canvas_w, canvas_h, 0])

    # colorbar 放在第 2、3 面板之间
    cbar = fig.colorbar(sm_obj, ax=axes[1:], fraction=0.015, pad=0.02)
    cbar.set_label("Attention Score (rank)", rotation=270,
                   labelpad=16, fontsize=10)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ [WSI heatmap] → {os.path.basename(save_path)}")
    return True


# ─────────────────────────────────────────────────────────────────

def _normalize_attention(scores: np.ndarray, mode: str) -> np.ndarray:
    """
    三种归一化方式：
      rank        → 按排名均匀分布颜色（推荐，颜色最丰富）
      percentile  → 5%~95% 百分位裁剪（高分区更红，低分区更蓝）
      linear      → 原始 min-max（忠实原始分布，可能大片蓝色）
    """
    a = scores.copy().astype(np.float32)
    if mode == "rank":
        ranks = np.argsort(np.argsort(a)).astype(np.float32)
        return ranks / max(len(ranks) - 1, 1)
    elif mode == "percentile":
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        if hi > lo:
            return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return np.ones_like(a) * 0.5
    else:  # linear
        lo, hi = a.min(), a.max()
        if hi > lo:
            return (a - lo) / (hi - lo)
        return np.ones_like(a) * 0.5


def _smooth_heatmap(raw: np.ndarray, pw: int, level: int) -> np.ndarray:
    """
    level=0  纯马赛克，无平滑
    level=1  轻微：sigma=0.4×patch，填缝隙，保留方块感
    level=2  较强：sigma=0.8×patch，边缘融合，接近连续热力图
    """
    if level == 0:
        return raw
    sigma = pw * (0.4 if level == 1 else 0.8)
    sigma = max(0.8, sigma)
    filled = raw.copy()
    filled[filled < 0] = 0.0
    smoothed = gaussian_filter(filled, sigma=sigma)
    coverage = gaussian_filter((raw >= 0).astype(np.float32), sigma=sigma * 0.6)
    result = np.where(coverage > 0.05, smoothed, -1.0)
    print(f"  📐 Smooth level={level}, sigma={sigma:.1f}px")
    return result


def _annotate_top_k(ax, coords, attention_scores, patch_size,
                     scale, x_min, y_min, pw, ph, top_k):
    k = min(top_k, len(attention_scores))
    for rank, idx in enumerate(np.argsort(attention_scores)[-k:][::-1]):
        x, y = coords[idx]
        xt = (x - x_min) * scale
        yt = (y - y_min) * scale
        rect = Rectangle((xt, yt), pw, ph,
                          fill=False, edgecolor="yellow", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(xt + pw / 2, yt - 3,
                f"#{rank+1}\n{attention_scores[idx]:.3f}",
                color="yellow", fontsize=6, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                          alpha=0.6, edgecolor="yellow", linewidth=0.7))


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
        print(f"  ⚠️  Patch dir not found under: {patch_png_dir}")
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
            cx  = int((x - x_min) * scale);  cy = int((y - y_min) * scale)
            cx2 = min(cx + pw, canvas_w);     cy2 = min(cy + ph, canvas_h)
            canvas[cy:cy2, cx:cx2] = np.array(img)[:cy2-cy, :cx2-cx]
            placed += 1
        except Exception:
            continue

    print(f"  ✅ Background: {placed}/{len(coords)} patches placed")
    return canvas if placed > 0 else None


def _detect_stride(coords):
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


def _save_placeholder(save_path, figsize, msg):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.text(0.5, 0.5, f"WSI Heatmap Unavailable\n{msg}",
            ha="center", va="center", fontsize=13, color="#888888",
            transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)