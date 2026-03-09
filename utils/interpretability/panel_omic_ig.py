"""
panel_omic_ig.py
================
面板7：分子组学 Integrated Gradients 可解释性

生成三张子图：
  子图1：Top-20 基因重要性（CNV + MUT + RNA 合并排名）
  子图2：CNV / MUT / RNA 三组学分层 Top-15
  子图3：通路聚合重要性（需要 gseapy 或自定义通路基因集）

用法：
    from utils.interpretability.panel_omic_ig import save_panel

    save_panel(
        omic_ig_scores = result.omic_ig_scores,   # [omic_dim]
        omic_col_names = result.omic_col_names,   # ['TP53_cnv', 'TP53_mut', ...]
        save_path      = 'panel7_omic_ig.png',
        pathway_gene_sets = pathway_gene_sets,    # dict，可为 None
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def save_panel(
    omic_ig_scores,
    omic_col_names,
    save_path: str,
    pathway_gene_sets: dict = None,   # {'KRAS_SIGNALING': ['KRAS','BRAF',...], ...}
    top_genes: int = 20,
    top_per_type: int = 15,
    top_pathways: int = 20,
    figsize: tuple = (22, 14),
    dpi: int = 150,
    slide_id: str = "",
) -> bool:

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True
    )

    if omic_ig_scores is None or omic_col_names is None:
        _save_placeholder(save_path, figsize, dpi, "Omic IG scores unavailable")
        return False

    scores = np.array(omic_ig_scores, dtype=np.float32)
    names  = list(omic_col_names)
    n = min(len(scores), len(names))
    scores, names = scores[:n], names[:n]

    # ── 拆分三组学 ────────────────────────────────────────────────
    cnv_idx  = [i for i, c in enumerate(names) if c.endswith("_cnv")]
    mut_idx  = [i for i, c in enumerate(names) if c.endswith("_mut")]
    rna_idx  = [i for i, c in enumerate(names) if c.endswith("_rnaseq")]

    print(f"  📊 Omic breakdown: CNV={len(cnv_idx)}, MUT={len(mut_idx)}, RNA={len(rna_idx)}")

    # ── 分组归一化（每组内部独立归一化到 [0,1]）─────────────────
    # 目的：防止 CNV 特征数量少导致分数虚高，RNA 特征多导致分数被稀释
    # 通路聚合用原始分数（保留组间信息），可视化用归一化分数
    scores_normed = scores.copy()

    def _inplace_normalize(idx_list):
        if not idx_list:
            return
        vals = scores[idx_list]
        lo, hi = vals.min(), vals.max()
        if hi - lo > 1e-8:
            scores_normed[idx_list] = (vals - lo) / (hi - lo)
        else:
            scores_normed[idx_list] = np.zeros_like(vals)

    _inplace_normalize(cnv_idx)
    _inplace_normalize(mut_idx)
    _inplace_normalize(rna_idx)

    print(f"  📊 Per-type normalization done: "
          f"CNV max={scores_normed[cnv_idx].max():.3f}, "
          f"MUT max={scores_normed[mut_idx].max() if mut_idx else 0:.3f}, "
          f"RNA max={scores_normed[rna_idx].max() if rna_idx else 0:.3f}")

    # ── 通路聚合（用原始 IG 分数，保留真实量级）────────────────────
    pathway_scores = None
    if pathway_gene_sets is not None and len(pathway_gene_sets) > 0:
        pathway_scores = _aggregate_to_pathways(scores, names, pathway_gene_sets)
        print(f"  📊 Pathways aggregated: {len(pathway_scores)}")

    # ── 布局 ──────────────────────────────────────────────────────
    has_pathway = pathway_scores is not None and len(pathway_scores) > 0
    n_cols = 3 if has_pathway else 2
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(1, n_cols, figure=fig,
                             wspace=0.38, left=0.06, right=0.97,
                             top=0.90, bottom=0.08)

    short_id = (slide_id[:70] + "...") if len(slide_id) > 70 else slide_id
    fig.suptitle(
        f"Omic Integrated Gradients — Gene & Pathway Importance\n{short_id}",
        fontsize=13, fontweight="bold", y=0.97
    )

    # ── 子图1：Top-N 全局基因重要性（用归一化分数，组间公平比较）───
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_top_genes(ax1, scores_normed, names, top_k=top_genes,
                    title=f"Top-{top_genes} Gene Importance\n(Per-type Normalized)")

    # ── 子图2：CNV / MUT / RNA 分层（用归一化分数）───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_by_omic_type(ax2, scores_normed, names,
                       cnv_idx, mut_idx, rna_idx,
                       top_k=top_per_type,
                       title="Top Genes by Omic Type\n(Per-type Normalized)")

    # ── 子图3：通路重要性（可选）─────────────────────────────────
    if has_pathway:
        ax3 = fig.add_subplot(gs[0, 2])
        _plot_pathways(ax3, pathway_scores, top_k=top_pathways,
                       title=f"Top-{top_pathways} Pathway Importance")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ [Omic IG] → {os.path.basename(save_path)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 子图绘制函数
# ─────────────────────────────────────────────────────────────────────────────

def _plot_top_genes(ax, scores, names, top_k=20, title=""):
    """子图1：全局 Top-K 基因重要性水平柱状图"""
    k = min(top_k, len(scores))
    top_idx = np.argsort(scores)[-k:][::-1]
    top_scores = scores[top_idx]
    top_names  = [_short_name(names[i]) for i in top_idx]

    # 颜色：按组学类型区分
    colors = []
    for i in top_idx:
        n = names[i]
        if n.endswith("_cnv"):     colors.append("#E07B6A")   # 红
        elif n.endswith("_mut"):   colors.append("#5BA8A0")   # 绿
        else:                      colors.append("#7DAECC")   # 蓝

    y_pos = np.arange(k)[::-1]
    ax.barh(y_pos, top_scores, color=colors,
            edgecolor="white", linewidth=0.4, height=0.72)

    # 分数标签
    for yp, sc in zip(y_pos, top_scores):
        ax.text(sc + top_scores.max() * 0.01, yp,
                f"{sc:.4f}", va="center", fontsize=7, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("IG Importance Score", fontsize=9)
    ax.set_xlim(0, top_scores.max() * 1.20)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E07B6A", label="CNV"),
        Patch(facecolor="#5BA8A0", label="MUT"),
        Patch(facecolor="#7DAECC", label="RNA"),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc="lower right", framealpha=0.8)


def _plot_by_omic_type(ax, scores, names,
                        cnv_idx, mut_idx, rna_idx,
                        top_k=15, title=""):
    """子图2：CNV / MUT / RNA 三列并排，每列 Top-K"""

    def _top(indices, k):
        if not indices:
            return [], []
        arr = scores[indices]
        ns  = [names[i] for i in indices]
        order = np.argsort(arr)[-k:][::-1]
        return [_short_name(ns[o], strip_suffix=True) for o in order], arr[order]

    cnv_names, cnv_sc = _top(cnv_idx, top_k)
    mut_names, mut_sc = _top(mut_idx, top_k)
    rna_names, rna_sc = _top(rna_idx, top_k)

    # 三列并排（共享 y 轴刻度个数）
    max_k = max(len(cnv_names), len(mut_names), len(rna_names), 1)
    x_groups = np.arange(max_k)

    # 使用嵌套轴实现三列
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # 子轴
    bbox = ax.get_position()
    fig  = ax.get_figure()
    w3   = bbox.width / 3.1
    pad  = bbox.width * 0.02

    specs = [
        (cnv_names, cnv_sc, "#E07B6A", "CNV"),
        (mut_names, mut_sc, "#5BA8A0", "MUT"),
        (rna_names, rna_sc, "#7DAECC", "RNA"),
    ]
    for col_i, (gnames, gsc, color, label) in enumerate(specs):
        left = bbox.x0 + col_i * (w3 + pad)
        sub_ax = fig.add_axes([left, bbox.y0, w3, bbox.height])
        k = len(gnames)
        if k == 0:
            sub_ax.text(0.5, 0.5, f"No {label}\nfeatures",
                        ha="center", va="center", fontsize=9, color="#888")
            sub_ax.axis("off")
            sub_ax.set_title(label, fontsize=10, fontweight="bold", color=color)
            continue

        y_pos = np.arange(k)[::-1]
        sub_ax.barh(y_pos, gsc, color=color,
                    edgecolor="white", linewidth=0.3, height=0.7, alpha=0.85)
        sub_ax.set_yticks(y_pos)
        sub_ax.set_yticklabels(gnames, fontsize=7.5)
        sub_ax.set_xlabel("IG Score", fontsize=8)
        sub_ax.set_xlim(0, max(gsc.max(), 1e-6) * 1.25)
        sub_ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        sub_ax.set_axisbelow(True)
        sub_ax.spines[["top", "right"]].set_visible(False)
        sub_ax.set_title(f"{label} Top-{k}", fontsize=10,
                         fontweight="bold", color=color)


def _plot_pathways(ax, pathway_scores: dict, top_k=20, title=""):
    """子图3：通路聚合重要性"""
    pw_names  = list(pathway_scores.keys())
    pw_scores = np.array(list(pathway_scores.values()), dtype=np.float32)

    k = min(top_k, len(pw_names))
    top_idx = np.argsort(pw_scores)[-k:][::-1]
    top_scores = pw_scores[top_idx]
    top_names  = [_truncate(pw_names[i], 40) for i in top_idx]

    # 颜色渐变
    norm_s = (top_scores - top_scores.min()) / max(top_scores.max() - top_scores.min(), 1e-8)
    cmap   = plt.get_cmap("YlOrRd")
    colors = [cmap(v) for v in norm_s]

    y_pos = np.arange(k)[::-1]
    bars = ax.barh(y_pos, top_scores, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.72)

    for yp, sc in zip(y_pos, top_scores):
        ax.text(sc + top_scores.max() * 0.01, yp,
                f"{sc:.4f}", va="center", fontsize=7, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Aggregated IG Score", fontsize=9)
    ax.set_xlim(0, top_scores.max() * 1.20)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(top_scores.min(), top_scores.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02).set_label(
        "Relative Importance", rotation=270, labelpad=12, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# 通路聚合
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_to_pathways(scores, col_names, pathway_gene_sets: dict) -> dict:
    """
    将基因级 IG 分数聚合到通路级别。

    col_names 格式：['TP53_cnv', 'TP53_mut', 'TP53_rnaseq', 'KRAS_cnv', ...]
    pathway_gene_sets 格式：{'KRAS_SIGNALING': ['KRAS', 'BRAF', ...], ...}

    聚合策略：取该通路内所有匹配基因特征的 IG 分数均值
    """
    # 建立 基因名 → [分数列表] 的映射（同一基因有多种组学类型）
    gene_scores = {}
    for col, sc in zip(col_names, scores):
        gene = col.rsplit("_", 1)[0]   # 'TP53_cnv' → 'TP53'
        gene_scores.setdefault(gene, []).append(float(sc))

    # 每个基因取最大值（CNV/MUT/RNA 中取最重要的那个）
    gene_max = {g: max(v) for g, v in gene_scores.items()}

    pathway_result = {}
    for pathway, genes in pathway_gene_sets.items():
        matched = [gene_max[g] for g in genes if g in gene_max]
        if matched:
            pathway_result[pathway] = float(np.mean(matched))

    return pathway_result


def load_msigdb_hallmark(cache_dir: str = None) -> dict:
    """
    自动下载 MSigDB Hallmark 50 个通路基因集（需要 gseapy）。

    Returns
    -------
    dict  {pathway_name: [gene_list]}
    """
    try:
        import gseapy
        gene_sets = gseapy.get_library("MSigDB_Hallmark_2020",
                                        organism="Human")
        print(f"  ✅ MSigDB Hallmark loaded: {len(gene_sets)} pathways")
        return gene_sets
    except ImportError:
        print("  ⚠️ gseapy not installed. Run: pip install gseapy")
        print("     Falling back to built-in LUAD-relevant pathway set.")
        return _builtin_luad_pathways()
    except Exception as e:
        print(f"  ⚠️ MSigDB download failed: {e}")
        print("     Falling back to built-in LUAD-relevant pathway set.")
        return _builtin_luad_pathways()


def load_kegg_pathways() -> dict:
    """
    自动下载 KEGG 通路基因集（需要 gseapy）。
    """
    try:
        import gseapy
        gene_sets = gseapy.get_library("KEGG_2021_Human")
        print(f"  ✅ KEGG pathways loaded: {len(gene_sets)} pathways")
        return gene_sets
    except Exception as e:
        print(f"  ⚠️ KEGG download failed: {e}")
        return _builtin_luad_pathways()


def load_pathways_from_gmt(gmt_path: str) -> dict:
    """
    从本地 .gmt 文件加载通路基因集（MSigDB 官网可下载）。

    .gmt 格式：每行 = 通路名\t描述\t基因1\t基因2\t...
    """
    gene_sets = {}
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway_name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            gene_sets[pathway_name] = genes
    print(f"  ✅ GMT file loaded: {len(gene_sets)} pathways from {gmt_path}")
    return gene_sets


def _builtin_luad_pathways() -> dict:
    """
    内置的 LUAD 相关通路基因集（不需要网络连接）。
    覆盖最重要的几个 cancer hallmark 通路。
    """
    return {
        "KRAS_SIGNALING": [
            "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2",
            "MAPK1", "MAPK3", "ELK1", "FOS", "JUN", "MYC"
        ],
        "EGFR_SIGNALING": [
            "EGFR", "ERBB2", "ERBB3", "GRB2", "SOS1", "PIK3CA", "AKT1",
            "MTOR", "PTEN", "RPS6KB1", "EIF4EBP1"
        ],
        "TP53_PATHWAY": [
            "TP53", "MDM2", "MDM4", "CDKN1A", "BAX", "BCL2", "PUMA",
            "NOXA", "APAF1", "CASP3", "CASP9"
        ],
        "CELL_CYCLE": [
            "CDK4", "CDK6", "CCND1", "CCND2", "CCND3", "CDKN2A", "RB1",
            "E2F1", "E2F2", "E2F3", "CCNE1", "CDK2", "CDKN1B"
        ],
        "PI3K_AKT_MTOR": [
            "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PTEN", "AKT1", "AKT2",
            "AKT3", "MTOR", "TSC1", "TSC2", "RICTOR", "RAPTOR"
        ],
        "MYC_TARGETS": [
            "MYC", "MYCN", "MAX", "MXD1", "CDK4", "CCND1", "E2F1",
            "NPM1", "NCL", "LDHA", "PKM", "HK2"
        ],
        "HYPOXIA": [
            "HIF1A", "HIF2A", "VEGFA", "VEGFB", "VEGFC", "LDHA", "PKM",
            "SLC2A1", "PDK1", "CA9", "EPO", "EPAS1"
        ],
        "EMT": [
            "CDH1", "CDH2", "VIM", "FN1", "TWIST1", "TWIST2", "SNAI1",
            "SNAI2", "ZEB1", "ZEB2", "MMP2", "MMP9"
        ],
        "DNA_REPAIR": [
            "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51",
            "XRCC1", "MLH1", "MSH2", "MSH6", "PALB2"
        ],
        "IMMUNE_CHECKPOINT": [
            "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
            "CD8A", "CD4", "FOXP3", "IL2", "IFNG", "TNF"
        ],
        "WNT_SIGNALING": [
            "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B", "DVL1", "FZD1",
            "LRP5", "LRP6", "TCF7", "TCF7L2", "LEF1"
        ],
        "NOTCH_SIGNALING": [
            "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2",
            "DLL1", "DLL3", "DLL4", "HES1", "HES5", "MAML1"
        ],
        "ANGIOGENESIS": [
            "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "FLT4", "PDGFRA",
            "PDGFRB", "ANGPT1", "ANGPT2", "TEK", "NRP1"
        ],
        "APOPTOSIS": [
            "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BID", "BIM",
            "CASP3", "CASP7", "CASP8", "CASP9", "XIAP", "SURVIVIN"
        ],
        "METABOLISM": [
            "LDHA", "PKM", "HK2", "GPI", "PFKL", "ALDOA", "GAPDH",
            "PGK1", "ENO1", "PKLR", "IDH1", "IDH2", "SDHA"
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _short_name(col_name: str, strip_suffix: bool = False, max_len: int = 20) -> str:
    """'VERY_LONG_GENE_NAME_cnv' → 'VERY_LONG_G…_cnv' 或 'VERY_LONG_G…'"""
    if strip_suffix:
        # 去掉 _cnv/_mut/_rnaseq
        for suf in ("_rnaseq", "_cnv", "_mut"):
            if col_name.endswith(suf):
                col_name = col_name[:-len(suf)]
                break
    if len(col_name) <= max_len:
        return col_name
    return col_name[:max_len - 1] + "…"


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def _save_placeholder(save_path, figsize, dpi, msg):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=13, color="#888", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
