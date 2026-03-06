"""
generate_interpretability.py
=============================
训练完成后独立生成可解释性面板，每次只生成一个面板（--panel）。

数据结构（与 main.py / dataset_survival.py 完全一致）
----------------------------------------------------
--data_csv     大 CSV 文件，即训练时用的那个（支持 .csv.zip）
               路径示例：datasets_csv/text_tcga_luad_all_clean.csv.zip
               包含列：case_id, slide_id, survival_months, censorship,
                       oncotree_code, age, site, is_female,
                       *_cnv, *_mut, *_rnaseq  （几千列 omic 特征）

--split_csv    splits_N.csv，用来筛选某个 fold 的样本
               路径示例：splits/5foldcv/tcga_luad/splits_0.csv
               包含列：train, val  （每列是 case_id 列表）

--split_key    从 split_csv 取哪个集合：train 或 val（默认 val）

--text_npy     文本嵌入 .npy 文件，shape=[N_samples, 6, 768]
               路径示例：data_text_features/LUAD_text_embeddings_qa_level.npy
               行顺序与 data_csv 过滤后的样本顺序一致

--data_dir     WSI 特征根目录，含 h5_files/ 子目录
               路径示例：/root/autodl-tmp/features/LUAD

--checkpoint   训练好的模型权重 (.pt)
               路径示例：results/fold0/s_0_maxcindex_checkpoint.pt

--panel 可选值与对应文件
------------------------
  wsi_heatmap      →  utils/interpretability/panel_wsi_heatmap.py
  fusion_weights   →  utils/interpretability/panel_fusion_weights.py
  omic_pathways    →  utils/interpretability/panel_omic_pathways.py
  qa_pathway       →  utils/interpretability/panel_qa_pathway.py
  qa_patch         →  utils/interpretability/panel_qa_patch.py
  patch_pathway    →  utils/interpretability/panel_patch_pathway.py

使用示例
--------
# 第 1 步：只生成 WSI 热力图，确认没问题
python generate_interpretability.py \\
    --checkpoint  results/fold0/s_0_maxcindex_checkpoint.pt \\
    --data_csv    datasets_csv/text_tcga_luad_all_clean.csv.zip \\
    --split_csv   splits/5foldcv/tcga_luad/splits_0.csv \\
    --split_key   val \\
    --data_dir    /root/autodl-tmp/features/LUAD \\
    --text_npy    data_text_features/LUAD_text_embeddings_qa_level.npy \\
    --output_dir  interpretability_output/ \\
    --task_type   survival \\
    --panel       wsi_heatmap

# 第 2 步：融合权重（确认上一步没问题后再跑）
python generate_interpretability.py ... --panel fusion_weights
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ════════════════════════════════════════════════════════════════
# 命令行参数
# ════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description="独立可解释性生成脚本（训练后运行，每次只跑一个面板）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 必需参数 ──────────────────────────────────────────────────
    parser.add_argument("--checkpoint", type=str, required=True,
        help="模型权重路径，如 results/fold0/s_0_maxcindex_checkpoint.pt")
    parser.add_argument("--data_csv", type=str, required=True,
        help="训练时用的大 CSV（支持 .csv.zip），"
             "如 datasets_csv/text_tcga_luad_all_clean.csv.zip")
    parser.add_argument("--data_dir", type=str, required=True,
        help="WSI 特征根目录，含 h5_files/ 子目录")
    parser.add_argument("--output_dir", type=str, required=True,
        help="面板图像输出目录")

    # ── 面板选择（每次只跑一个）─────────────────────────────────────
    #
    #   --panel wsi_heatmap    →  panel_wsi_heatmap.py
    #   --panel fusion_weights →  panel_fusion_weights.py
    #   --panel omic_pathways  →  panel_omic_pathways.py
    #   --panel qa_pathway     →  panel_qa_pathway.py
    #   --panel qa_patch       →  panel_qa_patch.py
    #   --panel patch_pathway  →  panel_patch_pathway.py
    parser.add_argument("--panel", type=str, required=True,
        choices=[
            "wsi_heatmap",    # → panel_wsi_heatmap.py
            "fusion_weights", # → panel_fusion_weights.py
            "omic_pathways",  # → panel_omic_pathways.py
            "qa_pathway",     # → panel_qa_pathway.py
            "qa_patch",       # → panel_qa_patch.py
            "patch_pathway",  # → panel_patch_pathway.py
        ],
        help="每次只生成一个面板，出错时精确定位到对应 panel_*.py 文件")

    # ── split 筛选（与 main.py 完全一致）────────────────────────────
    # splits_N.csv 的列是 train / val，每列是 case_id 列表
    parser.add_argument("--split_csv", type=str, default=None,
        help="splits_N.csv 路径，如 splits/5foldcv/tcga_luad/splits_0.csv。"
             "不提供则取 data_csv 前 max_samples 行")
    parser.add_argument("--split_key", type=str, default="val",
        choices=["train", "val", "test"],
        help="从 split_csv 取哪个集合（默认 val）")

    # ── 文本嵌入（与 dataset_survival.py 的 .npy 加载一致）──────────
    parser.add_argument("--text_npy", type=str, default=None,
        help="文本嵌入 .npy 文件，shape=[N, 6, 768]，"
             "如 data_text_features/LUAD_text_embeddings_qa_level.npy")

    # ── 模型结构 ──────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=None,
        help="训练时保存的 args pickle，有此文件可自动填充模型参数")
    parser.add_argument("--model_type", type=str, default="gram_porpoise_mmf",
        choices=["gram_porpoise_mmf", "porpoise_mmf"])
    parser.add_argument("--task_type", type=str, default="survival",
        choices=["survival", "classification", "multi_label"])
    parser.add_argument("--fusion", type=str, default="concat")
    parser.add_argument("--mode", type=str, default="pathomictext",
        choices=["path", "pathomic", "pathomictext"])
    parser.add_argument("--omic_input_dim", type=int, default=None,
        help="不填则自动从 data_csv 的 *_cnv/*_mut/*_rnaseq 列数推断")
    parser.add_argument("--text_input_dim", type=int, default=768)
    parser.add_argument("--path_input_dim", type=int, default=1024)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_labels", type=int, default=5)
    parser.add_argument("--drop_out", action="store_true", default=True)

    # ── 样本选择 ──────────────────────────────────────────────────
    parser.add_argument("--max_samples", type=int, default=5,
        help="最多处理多少个样本（不提供 --split_csv 时生效）")
    parser.add_argument("--sample_ids", nargs="+", default=None,
        help="直接指定 case_id 列表（优先级最高）")
    parser.add_argument("--fold", type=int, default=0,
        help="fold 编号，仅用于输出目录命名")

    # ── 可解释性辅助文件 ──────────────────────────────────────────
    parser.add_argument("--preprocessing_dir", type=str,
        default="./outputs/preprocessing",
        help="WSI 预处理目录。用于查找缩略图（thumbnails/ 子目录）和 patch PNG。"
             "你的路径示例：outputs_LUAD/preprocessing"
             "（其下的 output/{slide_id}/*.png 会被用来重建缩略图）")
    parser.add_argument("--pathway_names_file", type=str, default=None,
        help="通路名称文件，每行一个")
    parser.add_argument("--qa_text_file", type=str, default=None,
        help="QA 文本文件，每行一个问题（共 6 行）")
    parser.add_argument("--class_names", nargs="+", default=None,
        help="分类任务类别名称列表")

    # ── 其他 ──────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda",
        choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════

# 与 dataset_survival.py 一致的元数据列集合，其余列都是 omic 特征
_META_COLS = {
    "Unnamed: 0", "case_id", "slide_id", "site", "is_female",
    "oncotree_code", "age", "survival_months", "censorship",
    "disc_label", "label", "train", "event_time",
}


def _get_omic_cols(df: pd.DataFrame) -> list:
    """
    识别 omic 特征列（*_cnv / *_mut / *_rnaseq），
    与 dataset_survival.py Generic_Split 的列过滤逻辑一致。
    """
    cols = [c for c in df.columns
            if c.endswith("_cnv") or c.endswith("_mut") or c.endswith("_rnaseq")]
    if not cols:
        # 找不到标准后缀时，兜底为所有非元数据列
        cols = [c for c in df.columns if c not in _META_COLS]
    return cols


def load_data_csv(path: str) -> pd.DataFrame:
    """
    加载大 CSV（支持 .csv 和 .csv.zip）。
    与 dataset_survival.py 的 pd.read_csv(csv_path, low_memory=False) 一致。
    """
    print(f"\n📋 Loading data CSV: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ data_csv 文件不存在: {path}\n\n"
            f"  训练时的 CSV 路径格式为：\n"
            f"    datasets_csv/text_<study>_all_clean.csv.zip\n"
            f"  例如：\n"
            f"    datasets_csv/text_tcga_luad_all_clean.csv.zip\n"
            f"    datasets_csv/text_tcga_blca_all_clean.csv.zip\n\n"
            f"  请用 --data_csv 指定完整路径。"
        )
    df = pd.read_csv(path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    print(f"  总行数: {len(df)}，总列数: {len(df.columns)}")
    return df


def filter_by_split(df: pd.DataFrame, split_csv: str, split_key: str) -> pd.DataFrame:
    """
    用 splits_N.csv 筛选指定集合的样本。
    与 main.py 的 dataset.return_splits(csv_path=...) 逻辑一致：
    splits_N.csv 每列是一个集合的 case_id 列表。
    """
    print(f"\n🔀 Filtering by split: {split_csv}  [key={split_key}]")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"split_csv 不存在: {split_csv}")

    splits = pd.read_csv(split_csv)
    if split_key not in splits.columns:
        raise ValueError(
            f"split_csv 中没有 '{split_key}' 列。\n"
            f"  可用列: {list(splits.columns)}\n"
            f"  请用 --split_key 指定（train / val / test）"
        )

    case_ids = [str(c) for c in splits[split_key].dropna().tolist()]

    # data_csv 用 case_id 列匹配
    id_col = "case_id" if "case_id" in df.columns else "slide_id"
    mask = df[id_col].astype(str).isin(case_ids)
    filtered = df[mask].reset_index(drop=True)

    print(f"  {split_key} 集合共 {len(case_ids)} 个 case_id，"
          f"data_csv 匹配到 {len(filtered)} 个样本")

    if len(filtered) == 0:
        raise ValueError(
            f"\n❌ 过滤后样本数为 0！\n"
            f"  split_csv case_id 示例: {case_ids[:3]}\n"
            f"  data_csv {id_col} 示例: {df[id_col].head(3).tolist()}\n"
            f"  请确认格式一致（大小写、是否含 .svs 后缀等）"
        )
    return filtered


def select_samples(df: pd.DataFrame, args) -> pd.DataFrame:
    """根据 --sample_ids / --max_samples 进一步筛选"""
    if args.sample_ids:
        id_col = "case_id" if "case_id" in df.columns else "slide_id"
        mask = df[id_col].astype(str).isin([str(s) for s in args.sample_ids])
        selected = df[mask].reset_index(drop=True)
        print(f"  --sample_ids 指定: {len(selected)} / {len(df)} 个样本")
    else:
        selected = df.head(args.max_samples).reset_index(drop=True)
        print(f"  取前 {len(selected)} 个样本（--max_samples {args.max_samples}）")
    return selected


def load_text_npy(path: str):
    """加载文本嵌入 .npy，shape=[N, 6, 768]"""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"  ⚠️  text_npy 不存在: {path}，文本特征将为 None")
        return None
    emb = np.load(path, allow_pickle=False)
    print(f"  ✅ Text embeddings: {emb.shape}  [{path}]")
    return emb


def load_pathway_names(path: str):
    if path and os.path.exists(path):
        with open(path) as f:
            names = [l.strip() for l in f if l.strip()]
        print(f"  ✅ Pathway names: {len(names)} 条")
        return names
    return None


def load_qa_texts(path: str):
    if path and os.path.exists(path):
        with open(path) as f:
            texts = [l.strip() for l in f if l.strip()][:6]
        print(f"  ✅ QA texts: {len(texts)} 条")
        return texts
    return None


def load_config(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class SampleLoader:
    """
    按行索引逐样本加载数据，与 dataset_survival.py 的读取逻辑一致。

    WSI 特征  → data_dir/h5_files/{slide_id}.h5  （含 coords）
    Omic 特征 → data_csv 中的 *_cnv / *_mut / *_rnaseq 列
    Text 特征 → text_npy[idx]，shape [6, 768]
    """

    def __init__(self, args, df: pd.DataFrame, text_embeddings=None):
        self.args = args
        self.df = df.reset_index(drop=True)
        self.text_embeddings = text_embeddings   # np.ndarray [N, 6, 768]
        self.omic_cols = _get_omic_cols(df)
        print(f"  SampleLoader 就绪: {len(df)} 个样本，"
              f"{len(self.omic_cols)} 个 omic 特征列")

    def load(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # slide_id 优先；如没有则用 case_id
        slide_id = (str(row["slide_id"]) if "slide_id" in row.index
                    else str(row["case_id"]))

        # 生存任务用 disc_label，分类任务用 label
        if "disc_label" in row.index:
            label = int(row["disc_label"])
        elif "label" in row.index:
            label = int(row["label"])
        else:
            label = None

        result = {
            "slide_id":      slide_id,
            "label":         label,
            "survival_time": float(row["survival_months"]) if "survival_months" in row.index else 0.0,
            "event":         int(row["censorship"])         if "censorship"      in row.index else 0,
            "data_WSI":  None,
            "data_omic": None,
            "data_text": None,
            "coords":    None,
        }

        mode = getattr(self.args, "mode", "pathomictext")
        if "path" in mode:
            wsi, coords = self._load_wsi(slide_id)
            result["data_WSI"] = wsi
            result["coords"]   = coords
        if "omic" in mode:
            result["data_omic"] = self._load_omic(idx)
        if "text" in mode:
            result["data_text"] = self._load_text(idx)

        return result

    def _load_wsi(self, slide_id):
        """H5 优先（含坐标），退回 PT。与 dataset_survival.py H5 加载一致。"""
        data_dir = self.args.data_dir
        clean_id = slide_id[:-4] if slide_id.endswith(".svs") else slide_id

        for name in [clean_id, slide_id]:
            h5_path = os.path.join(data_dir, "h5_files", f"{name}.h5")
            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, "r") as f:
                        feat_key = "tile_embeds" if "tile_embeds" in f else "features"
                        features = torch.from_numpy(f[feat_key][:]).float()
                        coords   = f["coords"][:] if "coords" in f else None
                    print(f"  ✅ WSI H5: {features.shape[0]} patches  [{h5_path}]")
                    return features, coords
                except Exception as e:
                    print(f"  ⚠️  H5 读取失败: {e}")

        for name in [clean_id, slide_id]:
            pt_path = os.path.join(data_dir, "pt_files", f"{name}.pt")
            if os.path.exists(pt_path):
                try:
                    features = torch.load(pt_path, map_location="cpu").float()
                    print(f"  ✅ WSI PT: {features.shape[0]} patches（无坐标）")
                    return features, None
                except Exception as e:
                    print(f"  ⚠️  PT 读取失败: {e}")

        print(f"  ❌ WSI 未找到: {slide_id}")
        print(f"     已尝试: {data_dir}/h5_files/{clean_id}.h5")
        return None, None

    def _load_omic(self, idx):
        """
        直接从 data_csv 的 *_cnv / *_mut / *_rnaseq 列读取。
        与 Generic_Split.genomic_features 完全一致。
        """
        if not self.omic_cols:
            print("  ⚠️  data_csv 中未找到 *_cnv/*_mut/*_rnaseq 列")
            return None
        arr = self.df.iloc[idx][self.omic_cols].values.astype(np.float32)
        return torch.from_numpy(arr)

    def _load_text(self, idx):
        """
        从 text_npy[idx] 读取，shape [6, 768]。
        与 dataset_survival.py 的 text_embeddings_raw[idx] 完全一致。
        """
        if self.text_embeddings is None:
            print("  ⚠️  未提供 --text_npy，文本特征为 None")
            return None
        if idx >= len(self.text_embeddings):
            print(f"  ⚠️  text_npy 行数不足: idx={idx}, len={len(self.text_embeddings)}")
            return None
        arr = self.text_embeddings[idx].astype(np.float32)   # [6, 768]
        return torch.from_numpy(arr)


# ════════════════════════════════════════════════════════════════
# 模型加载
# ════════════════════════════════════════════════════════════════

def infer_dims_from_checkpoint(ckpt: dict) -> dict:
    """
    直接从 checkpoint 的 weight shape 反推构造参数，
    永远不会出现 size mismatch。

    推导逻辑：
      path_input_dim  ← path_fc.1.weight       shape [512, path_input_dim]
      omic_input_dim  ← fc_omic.0.fc.0.weight  shape [768, omic_input_dim]
      n_classes       ← classifier.3.weight    shape [n_classes, 128]
    """
    dims = {}

    for key in ("path_fc.1.weight", "path_encoder.1.weight"):
        if key in ckpt:
            dims["path_input_dim"] = ckpt[key].shape[1]
            print(f"  📐 path_input_dim  = {dims['path_input_dim']}  (from {key})")
            break

    for key in ("fc_omic.0.fc.0.weight", "fc_omic.0.weight"):
        if key in ckpt:
            dims["omic_input_dim"] = ckpt[key].shape[1]
            print(f"  📐 omic_input_dim  = {dims['omic_input_dim']}  (from {key})")
            break

    for key in ("classifier.3.weight", "classifier.weight",
                "classifier.3.bias",   "classifier.bias"):
        if key in ckpt:
            dims["n_classes"] = ckpt[key].shape[0]
            print(f"  📐 n_classes       = {dims['n_classes']}  (from {key})")
            break

    # use_gram_fusion: checkpoint 有 volume_fusion 层则说明训练时开了此选项
    has_gram_fusion = any("volume_fusion" in k or "temp_proj_tri" in k
                          for k in ckpt.keys())
    dims["use_gram_fusion"] = has_gram_fusion
    print(f"  📐 use_gram_fusion = {has_gram_fusion}  "
          f"({'volume_fusion keys found' if has_gram_fusion else 'not found in checkpoint'})")

    return dims


def load_model(args, device):
    print("\n📦 Loading model...")
    print(f"  Checkpoint: {args.checkpoint}")

    # ── 先读 checkpoint，从 weight shape 反推维度 ──────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    inferred = infer_dims_from_checkpoint(ckpt)

    # inferred > args 显式传入 > 硬编码默认值
    path_input_dim = inferred.get("path_input_dim",
                     getattr(args, "path_input_dim", 1024))
    omic_input_dim = inferred.get("omic_input_dim",
                     getattr(args, "omic_input_dim") or 256)
    n_classes      = inferred.get("n_classes",
                     getattr(args, "n_classes", 4))

    if args.model_type == "gram_porpoise_mmf":
        from models.gram_porpoise import GRAMPorpoiseMMF

        if args.task_type == "multi_label":
            kw = dict(n_classes=4, n_labels=n_classes)
        else:
            kw = dict(n_classes=n_classes, n_labels=5)

        model = GRAMPorpoiseMMF(
            omic_input_dim=omic_input_dim,
            text_input_dim=getattr(args, "text_input_dim", 768),
            path_input_dim=path_input_dim,
            fusion=args.fusion,
            **kw,
            contra_dim=getattr(args, "contra_dim", 256),
            contra_temp=getattr(args, "contra_temp", 0.07),
            use_contrastive=getattr(args, "use_gram_contrastive", False),
            dropout=0.25 if getattr(args, "drop_out", True) else 0.0,
            task_type=args.task_type,
            path_hidden_dim=getattr(args, "path_hidden_dim", 512),
            path_attention_dim=getattr(args, "path_attention_dim", 256),
            encoder_hidden_dim=getattr(args, "encoder_hidden_dim", 768),
            distance_type=getattr(args, "distance_type", "volume"),
            use_gram_fusion=inferred.get("use_gram_fusion", getattr(args, "use_gram_fusion", False)),
            enable_explainability=getattr(args, "enable_text_centric", False),
            n_qa_pairs=getattr(args, "n_qa_pairs", 6),
            n_pathways=getattr(args, "n_pathways", 50),
        )

    elif args.model_type == "porpoise_mmf":
        from models.model_porpoise import PorpoiseMMF
        model = PorpoiseMMF(
            omic_input_dim=omic_input_dim,
            text_input_dim=getattr(args, "text_input_dim", 768),
            path_input_dim=path_input_dim,
            fusion=args.fusion,
            n_classes=n_classes,
            task_mode=args.task_type,
        )
    else:
        raise NotImplementedError(f"Unknown model_type: {args.model_type}")

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    if not missing and not unexpected:
        print("  ✅ Weights loaded perfectly (no missing / unexpected keys)")

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
    model.eval()

    # 同步回 args，后续步骤可能用到
    args.omic_input_dim = omic_input_dim
    args.path_input_dim = path_input_dim
    args.n_classes      = n_classes

    print(f"  ✅ Model ready")
    return model

# ════════════════════════════════════════════════════════════════
# 单面板调用分发
# ════════════════════════════════════════════════════════════════

def _call_single_panel(module, panel_name: str, data, save_path: str) -> bool:
    """
    将 ExtractedData 路由到指定面板的 save_panel()。

    panel_name        对应文件                     传入字段
    ──────────────    ─────────────────────────    ──────────────────────────────
    wsi_heatmap    →  panel_wsi_heatmap.py         thumbnail, coords, attention_scores
    fusion_weights →  panel_fusion_weights.py      fusion_weights, pred/true_label
    omic_pathways  →  panel_omic_pathways.py       pathway_scores, pathway_names
    qa_pathway     →  panel_qa_pathway.py          qa2pathway_attn, qa_texts, pathway_names
    qa_patch       →  panel_qa_patch.py            qa2patch_attn, qa_texts, thumbnail, coords
    patch_pathway  →  panel_patch_pathway.py       patch_pathway_map, pathway_names
    """
    if panel_name == "wsi_heatmap":
        # patch_png_dir: {preprocessing_dir}/output/{slide_id}/{x}x_{y}y.png
        import os as _os
        _preprocessing_dir = getattr(data, "_preprocessing_dir", None)
        _patch_png_dir = _os.path.join(_preprocessing_dir, "output")             if _preprocessing_dir else None
        return module.save_panel(
            thumbnail=data.thumbnail,
            coords=data.coords,
            attention_scores=data.attention_scores,
            save_path=save_path,
            slide_id=data.slide_id,
            patch_png_dir=_patch_png_dir,
        )
    if panel_name == "fusion_weights":
        return module.save_panel(
            fusion_weights=data.fusion_weights,
            save_path=save_path,
            pred_label=data.pred_label,
            true_label=data.true_label,
            class_names=data.class_names,
        )
    if panel_name == "omic_pathways":
        return module.save_panel(
            pathway_scores=data.pathway_scores,
            pathway_names=data.pathway_names,
            save_path=save_path,
        )
    if panel_name == "qa_pathway":
        return module.save_panel(
            qa2pathway_attn=data.qa2pathway_attn,
            qa_texts=data.qa_texts,
            pathway_names=data.pathway_names,
            save_path=save_path,
        )
    if panel_name == "qa_patch":
        return module.save_panel(
            qa2patch_attn=data.qa2patch_attn,
            qa_texts=data.qa_texts,
            save_path=save_path,
            thumbnail=data.thumbnail,
            coords=data.coords,
        )
    if panel_name == "patch_pathway":
        return module.save_panel(
            patch_pathway_map=data.patch_pathway_map,
            pathway_names=data.pathway_names,
            save_path=save_path,
        )
    raise ValueError(f"Unknown panel_name: {panel_name}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\n🖥️  Device : {device}")
    print(f"🎯 Panel  : {args.panel}")

    # ── 1. 加载训练时的 config ────────────────────────────────────
    if args.config and os.path.exists(args.config):
        print(f"\n⚙️  Loading config: {args.config}")
        train_args = load_config(args.config)
        for key in ("omic_input_dim", "text_input_dim", "path_input_dim",
                    "fusion", "n_classes", "n_labels", "task_type", "mode",
                    "path_hidden_dim", "path_attention_dim", "encoder_hidden_dim",
                    "distance_type", "use_gram_fusion", "use_gram_contrastive",
                    "enable_text_centric", "n_qa_pairs", "n_pathways",
                    "drop_out", "contra_dim", "contra_temp"):
            if hasattr(train_args, key) and getattr(args, key, None) is None:
                setattr(args, key, getattr(train_args, key))
        print("  ✅ Config loaded")

    # ── 2. 加载大 CSV ─────────────────────────────────────────────
    df = load_data_csv(args.data_csv)

    # ── 3. 按 split 筛选 ──────────────────────────────────────────
    if args.split_csv:
        df = filter_by_split(df, args.split_csv, args.split_key)
    else:
        print(f"\n  ℹ️  未提供 --split_csv，将直接取 data_csv 前若干行")

    df = select_samples(df, args)

    # ── 4. 加载文本嵌入 ───────────────────────────────────────────
    text_embeddings = load_text_npy(args.text_npy)

    # ── 5. 推断 omic_input_dim ────────────────────────────────────
    if args.omic_input_dim is None:
        omic_cols = _get_omic_cols(df)
        args.omic_input_dim = len(omic_cols)
        print(f"\n  ℹ️  omic_input_dim 自动推断 = {args.omic_input_dim}"
              f"（来自 *_cnv/*_mut/*_rnaseq 列数）")

    # ── 6. 加载辅助文件 ───────────────────────────────────────────
    pathway_names = load_pathway_names(args.pathway_names_file)
    qa_texts      = load_qa_texts(args.qa_text_file)
    class_names   = args.class_names

    # ── 7. 加载模型 ───────────────────────────────────────────────
    model = load_model(args, device)

    # ── 8. 只 import 指定面板模块 ─────────────────────────────────
    panel_file_map = {
        "wsi_heatmap":    "utils.interpretability.panel_wsi_heatmap",
        "fusion_weights": "utils.interpretability.panel_fusion_weights",
        "omic_pathways":  "utils.interpretability.panel_omic_pathways",
        "qa_pathway":     "utils.interpretability.panel_qa_pathway",
        "qa_patch":       "utils.interpretability.panel_qa_patch",
        "patch_pathway":  "utils.interpretability.panel_patch_pathway",
    }
    import importlib
    panel_module = importlib.import_module(panel_file_map[args.panel])
    print(f"\n✅ Panel module: {panel_file_map[args.panel].split('.')[-1]}.py")

    # ── 9. 初始化数据提取器和加载器 ──────────────────────────────
    from utils.interpretability.data_extractor import InterpretabilityDataExtractor
    extractor = InterpretabilityDataExtractor(
        model=model,
        device=device_str,
        preprocessing_dir=args.preprocessing_dir,
    )
    loader = SampleLoader(args=args, df=df, text_embeddings=text_embeddings)

    # ── 10. 逐样本只生成指定面板 ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀  [{args.panel}]  共 {len(df)} 个样本")
    print(f"    输出目录: {args.output_dir}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_success = 0

    for i in range(len(df)):
        print(f"\n[{i+1}/{len(df)}] 加载样本...")
        sample    = loader.load(i)
        slide_id  = sample["slide_id"]
        sample_id = f"fold{args.fold}_{slide_id.rstrip('.svs')}"

        data_WSI  = sample["data_WSI"].to(device)  if sample["data_WSI"]  is not None else None
        data_omic = sample["data_omic"].to(device) if sample["data_omic"] is not None else None
        data_text = sample["data_text"].to(device) if sample["data_text"] is not None else None

        # 提取数据（唯一调用模型的地方）
        # coords 直接从 H5 传入，不需要 extractor 再去 preprocessing_dir 查找
        data = extractor.extract(
            data_WSI=data_WSI,
            data_omic=data_omic,
            data_text=data_text,
            slide_id=slide_id,
            sample_id=sample_id,
            true_label=sample["label"],
            class_names=class_names,
            pathway_names=pathway_names,
            qa_texts=qa_texts,
            survival_time=sample["survival_time"],
            event=sample["event"],
            coords=sample["coords"],  # 来自 H5，已在 SampleLoader._load_wsi() 里读好
        )

        # 输出路径：{output_dir}/{sample_id}/{panel}.png
        out_dir   = os.path.join(args.output_dir, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{args.panel}.png")

        # 只调用这一个面板，出错日志只和这个面板有关
        try:
            ok = _call_single_panel(panel_module, args.panel, data, save_path)
            if ok:
                n_success += 1
        except Exception as e:
            import traceback
            print(f"  ❌ [{args.panel}] 失败 ({slide_id}): {e}")
            traceback.print_exc()

    # ── 11. 汇总 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅  [{args.panel}]  {n_success}/{len(df)} 个样本成功")
    print(f"    输出目录: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()