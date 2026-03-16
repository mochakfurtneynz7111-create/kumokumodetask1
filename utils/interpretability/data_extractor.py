"""
data_extractor.p
=================
从模型输出中提取原始 numpy 数据，供各面板独立使用。

职责：
  - 接触 PyTorch Tensor / 模型输出 dict
  - 输出纯 numpy 数组 + Python 原生类型
  - 不做任何可视化

使用：
    extractor = InterpretabilityDataExtractor(model, device)
    data = extractor.extract(
        data_WSI, data_omic, data_text,
        slide_id, survival_time, event
    )
    # data 是一个纯 numpy dict，直接传给各面板
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedData:
    """所有面板所需的原始数据，均为 numpy / Python 原生类型"""

    sample_id: str = ""
    slide_id: str = ""
    _preprocessing_dir: str = ""  # 传给 panel_wsi_heatmap 用于找 patch PNG

    # ── WSI ──────────────────────────────────────────────────────
    thumbnail: Optional[np.ndarray] = None          # [H, W, 3]  uint8
    coords: Optional[np.ndarray] = None             # [N, 2]     float32
    attention_scores: Optional[np.ndarray] = None   # [N]        float32
    ig_scores: Optional[np.ndarray] = None 
    omic_ig_scores:  Optional[np.ndarray] = None   # ← 新增 [omic_dim]
    omic_col_names:  Optional[list]       = None   # ← 新增 ['TP53_cnv',...]

    # ── 融合权重 ──────────────────────────────────────────────────
    fusion_weights: Optional[np.ndarray] = None     # [3]  (path, omic, text)

    # ── Omic / Pathway ───────────────────────────────────────────
    pathway_scores: Optional[np.ndarray] = None     # [n_pathways]
    pathway_names: Optional[list] = None            # list[str]

    # ── Text-centric ─────────────────────────────────────────────
    qa2pathway_attn: Optional[np.ndarray] = None    # [6, n_pathways]
    qa2patch_attn: Optional[np.ndarray] = None      # [6, n_patches]
    patch_pathway_map: Optional[np.ndarray] = None  # [n_patches, n_pathways]
    qa_texts: Optional[list] = None                 # list[str], len=6

    # ── 预测结果 ─────────────────────────────────────────────────
    true_label: Optional[int] = None
    pred_label: Optional[int] = None
    class_names: Optional[list] = None
    survival_time: Optional[float] = None
    event: Optional[int] = None


class InterpretabilityDataExtractor:
    """
    从模型输出中提取可解释性所需的所有原始数据。
    提取完成后返回 ExtractedData，后续面板只消费该对象。
    """

    DEFAULT_QA_TEXTS = [
        "What is the tumor grade and differentiation?",
        "Is there immune cell infiltration?",
        "What is the tumor proliferation status?",
        "Is there necrosis or hemorrhage?",
        "What is the stromal composition?",
        "Are there specific morphological features?",
    ]

    def __init__(self, model, device: str = "cuda",
                 preprocessing_dir: str = "./outputs/preprocessing",
                 flip_attention: bool = False,
                 task_type="survival",
                 hallmark_csv=None,    # ← 新增
                 omic_col_names=None): # ← 新增):
        self.model = model
        self.device = device
        self.preprocessing_dir = preprocessing_dir
        self.flip_attention = flip_attention
        self._thumbnail_dirs = [
            os.path.join(preprocessing_dir, "thumbnails"),
            preprocessing_dir,
        ]
        self.task_type = task_type
        # 用户的 patch PNG 目录（每个 slide 一个子目录，文件名格式为 {x}x_{y}y.png）
        self._patch_png_dir = os.path.join(preprocessing_dir, "output")
        self._coord_dir = os.path.join(preprocessing_dir, "output")

        # 加载通路-基因映射
        self.pathway_gene_map = {}   # {pathway_name: [gene_col1, gene_col2, ...]}
        self.pathway_names_ordered = []
        
        if hallmark_csv is not None and omic_col_names is not None:
            import pandas as pd
            hw = pd.read_csv(hallmark_csv, index_col=0)
            omic_set = set(omic_col_names)
            for pw in hw.columns:
                genes = hw[pw].dropna().tolist()
                # 每个基因匹配 _rnaseq 列（RNA表达最能反映通路活性）
                matched = [g + '_rnaseq' for g in genes if g + '_rnaseq' in omic_set]
                if matched:
                    self.pathway_gene_map[pw] = matched
                    self.pathway_names_ordered.append(pw)
            print(f"✅ Loaded {len(self.pathway_names_ordered)} pathways with gene mapping")
        
        self.omic_col_names = omic_col_names or []

    # ─────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────

    def extract(
        self,
        data_WSI,
        data_omic,
        data_text,
        slide_id: str,
        sample_id: str = "",
        true_label=None,
        class_names=None,
        qa_texts=None,
        pathway_names=None,
        survival_time=None,
        event=None,
        coords=None,          # 直接从 H5 传入，优先于 preprocessing_dir 查找
        omic_col_names: list  = None,
    ) -> ExtractedData:
        """
        主入口：运行模型前向传播，提取所有可解释性数据。

        Returns
        -------
        ExtractedData  (纯 numpy / Python，不含任何 Tensor)
        """
        result = ExtractedData(
            sample_id=sample_id,
            slide_id=slide_id,
            class_names=class_names or [],
            qa_texts=qa_texts or self.DEFAULT_QA_TEXTS,
            pathway_names=pathway_names,
            _preprocessing_dir=self.preprocessing_dir,
        )

        if true_label is not None:
            result.true_label = int(true_label) if hasattr(true_label, "item") \
                else int(true_label)
        if survival_time is not None:
            result.survival_time = float(survival_time)
        if event is not None:
            result.event = int(event)

        # 1. 模型前向
        model_output = self._run_forward(data_WSI, data_omic, data_text,
                                         survival_time, event)
        if model_output is None:
            print(f"  ❌ Forward pass failed for {sample_id}")
            return result

        # 2. 预测标签
        result.pred_label = self._extract_pred_label(model_output)

        # 3. WSI thumbnail + coords
        # coords: 优先使用从 H5 直接传入的（已在 SampleLoader 里读好），
        #         否则再去 preprocessing_dir 查找
        if coords is not None:
            result.coords = coords.astype(np.float32) if hasattr(coords, 'astype') else np.array(coords, dtype=np.float32)
        else:
            result.coords = self._load_coords(slide_id)

        # thumbnail: 先找标准缩略图文件，找不到就从 patch PNG 目录重建
        result.thumbnail = self._load_thumbnail(slide_id)
        if result.thumbnail is None and result.coords is not None:
            result.thumbnail = self._build_thumbnail_from_patches(slide_id, result.coords)

        # 4. Attention scores
        result.attention_scores = self._extract_attention(
            model_output, result.coords, flip=getattr(self, "flip_attention", False))

        # 4.5 IG scores（Integrated Gradients，比 attention 更能反映对预测的直接贡献）
        # 4.5 IG scores
        result.ig_scores = self._compute_ig_scores(
            data_WSI, data_omic, data_text,
            survival_time=survival_time,   # ← 新增
            event=event                    # ← 新增
        )

        # 5. Fusion weights
        result.fusion_weights = self._extract_fusion_weights(model_output)

        # 6. Pathway scores (omic importance)
        result.pathway_scores = self._extract_pathway_scores(
            model_output, data_omic, pathway_names
        )

        # 7. Text-centric (qa2pathway, qa2patch, patch_pathway)
        self._extract_text_centric(model_output, result)

        # 8. Omic IG scores（基因级别可解释性）
        if data_omic is not None:
            result.omic_col_names = omic_col_names   # 从外部传入
            result.omic_ig_scores = self._compute_omic_ig_scores(
                data_omic   = data_omic,
                data_WSI    = data_WSI,
                data_text   = data_text,
                survival_time = survival_time,
                event         = event,
            )

        return result

    # ─────────────────────────────────────────────────────────────
    # 私有：模型前向
    # ─────────────────────────────────────────────────────────────

    def _run_forward(self, data_WSI, data_omic, data_text,
                     survival_time, event):
        try:
            kwargs = {
                "compute_loss": False,
                "return_features": True,
                "return_explanation": True,
            }
            if data_WSI is not None:
                kwargs["x_path"] = [data_WSI] if not isinstance(data_WSI, list) else data_WSI
            if data_omic is not None:
                # 确保是 [1, omic_dim]
                x_omic = data_omic
                if x_omic.dim() == 1:          # [omic_dim] → [1, omic_dim]
                    x_omic = x_omic.unsqueeze(0)
                kwargs["x_omic"] = x_omic.to(self.device).float()
            if data_text is not None:
                # 确保是 [1, 6, 768]（batch × n_qa × hidden）
                x_text = data_text
                if x_text.dim() == 2:          # [6, 768] → [1, 6, 768]
                    x_text = x_text.unsqueeze(0)
                elif x_text.dim() == 1:        # [768] → [1, 1, 768]（兜底）
                    x_text = x_text.unsqueeze(0).unsqueeze(0)
                kwargs["x_text"] = x_text.to(self.device).float()
            if survival_time is not None:
                st = torch.tensor([survival_time], dtype=torch.float32) \
                    if not isinstance(survival_time, torch.Tensor) \
                    else survival_time.unsqueeze(0) if survival_time.dim() == 0 else survival_time
                kwargs["survival_time"] = st.to(self.device).float()
            if event is not None:
                ev = torch.tensor([event], dtype=torch.float32) \
                    if not isinstance(event, torch.Tensor) \
                    else event.unsqueeze(0) if event.dim() == 0 else event
                kwargs["event"] = ev.to(self.device).float()

            with torch.no_grad():
                return self.model(**kwargs)
        except Exception as e:
            print(f"  ❌ Forward pass error: {e}")
            import traceback; traceback.print_exc()
            return None

    # ─────────────────────────────────────────────────────────────
    # 私有：各字段提取
    # ─────────────────────────────────────────────────────────────

    def _extract_pred_label(self, output) -> Optional[int]:
        try:
            logits = output["logits"] if isinstance(output, dict) else output
            return int(torch.argmax(logits, dim=-1).item())
        except Exception:
            return None

    def _extract_attention(self, output, coords,
                           flip: bool = False) -> Optional[np.ndarray]:
        """优先从模型临时变量获取真实 attention"""
        # 方式1：_last_batch_attention
        if hasattr(self.model, "_last_batch_attention"):
            ba = self.model._last_batch_attention
            if ba is not None and len(ba) > 0:
                attn = ba[0]
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                # shape 可能是 [N,1] 或 [1,N] 或 [N]，统一展平
                attn = attn.squeeze()          # 去掉所有大小为1的维度
                if attn.ndim == 0:
                    attn = attn.reshape(1)
                if coords is not None and attn.shape[0] == len(coords):
                    if flip:
                        attn = -attn           # 反转方向
                    attn = np.maximum(attn, 0)   # 🔥 新增：ReLU，仅保留正向贡献
                    print(f"  📊 Raw attn: min={attn.min():.4f}  max={attn.max():.4f}"
                          f"  mean={attn.mean():.4f}  {'[FLIPPED]' if flip else ''}")
                    return attn.astype(np.float32)

        # 方式2：fusion_info 里的 patch_attention
        if isinstance(output, dict):
            fi = output.get("fusion_info", {})
            if "patch_attention" in fi:
                attn = fi["patch_attention"]
                if isinstance(attn, torch.Tensor):
                    attn = attn[0].cpu().numpy()
                return attn.astype(np.float32)

        # 方式3：explanation
        exp = self._get_explanation_raw(output)
        if exp is not None and "qa2patch_attention" in exp:
            qa2patch = exp["qa2patch_attention"]
            if isinstance(qa2patch, torch.Tensor):
                qa2patch = qa2patch[0].cpu().numpy()
            # 对6个QA取平均作为整体attention
            return qa2patch.mean(axis=0).astype(np.float32)

        return None

    def _extract_fusion_weights(self, output) -> Optional[np.ndarray]:
        if not isinstance(output, dict):
            return None
        fi = output.get("fusion_info", {})
        if "fusion_weights" in fi:
            fw = fi["fusion_weights"]
            if isinstance(fw, torch.Tensor):
                fw = fw[0].cpu().numpy()
            return fw.astype(np.float32)
        return None

    def _extract_pathway_scores(self, output, data_omic,
                                pathway_names) -> Optional[np.ndarray]:
        """
        提取通路重要性分数。
        优先级：model explanation → omic gradient → omic 特征绝对值均值
        """
        # 方式1：explanation 里有 pathway_scores
        exp = self._get_explanation_raw(output)
        if exp is not None and "pathway_scores" in exp:
            ps = exp["pathway_scores"]
            if isinstance(ps, torch.Tensor):
                ps = ps[0].cpu().numpy()
            return ps.astype(np.float32)

        # 方式2：omic feature 绝对值（降级方案）
        if data_omic is not None:
            omic_np = data_omic.cpu().numpy() if isinstance(data_omic, torch.Tensor) \
                else np.array(data_omic)
            omic_np = omic_np.ravel().astype(np.float32)
            n = len(pathway_names) if pathway_names else len(omic_np)
            n = min(n, len(omic_np))
            # 每段均值作为该通路分数
            if len(omic_np) >= n:
                chunk = len(omic_np) // n
                scores = np.array([
                    np.abs(omic_np[i*chunk:(i+1)*chunk]).mean()
                    for i in range(n)
                ], dtype=np.float32)
                # softmax 归一化
                scores = scores - scores.max()
                scores = np.exp(scores) / np.exp(scores).sum()
                return scores
        return None

    def _extract_text_centric(self, output, result: ExtractedData):
        exp = self._get_explanation_raw(output)
        # ── Post-hoc 方案：用训练好的 fc_text 计算 QA→Pathway 相似度 ──
        # result.qa2pathway_attn = self._compute_posthoc_qa_pathway(output)
        attn, pw_names = self._compute_posthoc_qa_pathway(output)
        result.qa2pathway_attn = attn
        if pw_names:
            result.pathway_names = pw_names  # 覆盖原来的通路名

        result.qa2patch_attn = self._compute_posthoc_qa_patch(output)
        
        # 🔥 新增：Patch ↔ Pathway
        result.patch_pathway_map = self._compute_posthoc_patch_pathway(output)
        
        if exp is None:
            return

        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x[0].cpu().numpy().astype(np.float32) \
                    if x.dim() == 3 else x.cpu().numpy().astype(np.float32)
            return np.array(x, dtype=np.float32)

        if "qa2pathway_attention" in exp:
            result.qa2pathway_attn = to_np(exp["qa2pathway_attention"])
        if "qa2patch_attention" in exp:
            result.qa2patch_attn = to_np(exp["qa2patch_attention"])
        if "patch_pathway_correspondence" in exp:
            result.patch_pathway_map = to_np(exp["patch_pathway_correspondence"])

    def _compute_posthoc_qa_patch(self, output) -> Optional[np.ndarray]:
        """
        Post-hoc 计算 QA → Patch 注意力。
        
        流程：
          1. _temp_text_3d  → [6, 768]  → fc_text → [6, hidden_dim]
          2. _temp_path_features → list[[n_patches, path_hidden_dim]]
                                 → 取第一个样本 → rho → [n_patches, hidden_dim]
          3. 余弦相似度 softmax → [6, n_patches]
        
        Returns:
            np.ndarray [6, n_patches] 或 None（缺少依赖时）
        """
        model  = self.model
        device = next(model.parameters()).device
    
        # ── 1. QA embeddings ──────────────────────────────────────
        if not hasattr(model, '_temp_text_3d') or model._temp_text_3d is None:
            print("  ⚠️ [QA→Patch] _temp_text_3d not found")
            return None
        x_text = model._temp_text_3d[0].float().to(device)   # [6, 768]
    
        # ── 2. Per-patch features ─────────────────────────────────
        if not hasattr(model, '_temp_path_features') or not model._temp_path_features:
            print("  ⚠️ [QA→Patch] _temp_path_features not found")
            return None
        patch_feats = model._temp_path_features[0].float().to(device)
        # patch_feats: [n_patches, path_hidden_dim]，即 path_fc 的输出
    
        with torch.no_grad():
            # ── 3. 编码 QA ────────────────────────────────────────
            if hasattr(model, 'fc_text') and model.fc_text is not None:
                qa_encoded = model.fc_text(x_text)            # [6, hidden_dim]
            else:
                print("  ⚠️ [QA→Patch] fc_text not found")
                return None
    
            # ── 4. 投影 patch 到同一语义空间 ──────────────────────
            # rho: [path_hidden_dim] → [hidden_dim]（和 fc_text 输出维度相同）
            if hasattr(model, 'rho') and model.rho is not None:
                patch_projected = model.rho(patch_feats)       # [n_patches, hidden_dim]
            else:
                # 没有 rho 时 fallback：直接用 patch_feats（维度可能不同，相似度计算会失败）
                print("  ⚠️ [QA→Patch] rho not found, using raw patch features")
                patch_projected = patch_feats
    
            # ── 5. 余弦相似度 → softmax ───────────────────────────
            qa_norm    = F.normalize(qa_encoded,      dim=1)   # [6, hidden_dim]
            patch_norm = F.normalize(patch_projected, dim=1)   # [n_patches, hidden_dim]
    
            sim  = torch.mm(qa_norm, patch_norm.t())            # [6, n_patches]
            attn = torch.softmax(sim * 5.0, dim=-1)             # [6, n_patches]，温度 5
    
        return attn.cpu().numpy().astype(np.float32)

    def _compute_posthoc_qa_pathway(self, output):
        model = self.model
        device = next(model.parameters()).device
    
        # ── 1. 取 [6, 768] QA embeddings ──
        if not hasattr(model, '_temp_text_3d') or model._temp_text_3d is None:
            print("  ⚠️ _temp_text_3d not found")
            return None, []
        print(f"  🔍 _temp_text_3d shape: {model._temp_text_3d.shape}")  
        # 期望输出：torch.Size([1, 6, 768])
        # 如果是 torch.Size([1, 768]) 说明 3D 没被保存，post-hoc 会失效
        x_text = model._temp_text_3d[0].float().to(device)  # [6, 768]
    
        # ── 2. 取原始 omic 向量 ──
        if not hasattr(model, '_temp_omic') or model._temp_omic is None:
            print("  ⚠️ _temp_omic not found")
            return None, []
        x_omic = model._temp_omic[0].float().to(device)  # [omic_dim]
    
        if not self.pathway_gene_map or not self.omic_col_names:
            print("  ⚠️ No pathway-gene mapping loaded")
            return None, []
    
        # ── 3. 为每个通路构建 masked omic 向量，过 fc_omic ──
        col2idx = {col: i for i, col in enumerate(self.omic_col_names)}
        
        pathway_vecs = []
        valid_pathway_names = []
    
        with torch.no_grad():
            # ── 基线：零输入时的 fc_omic 输出（纯 bias 贡献）──
            zero_input = torch.zeros(1, x_omic.shape[0], device=device)
            baseline_vec = model.fc_omic(zero_input)[0]  # [hidden_dim]
            
            for pw_name in self.pathway_names_ordered:
                gene_cols = self.pathway_gene_map[pw_name]
                indices = [col2idx[c] for c in gene_cols if c in col2idx]
                if not indices:
                    continue
                masked = torch.zeros_like(x_omic)
                masked[indices] = x_omic[indices]
                pw_vec = model.fc_omic(masked.unsqueeze(0))[0]
                
                # ← 关键修改：减去基线，只保留该通路基因的净贡献
                pw_vec = pw_vec - baseline_vec
                
                pathway_vecs.append(pw_vec)
                valid_pathway_names.append(pw_name)
    
            if not pathway_vecs:
                return None, []
    
            pathway_mat = torch.stack(pathway_vecs, dim=0)  # [n_pathways, hidden_dim]
    
            # ── 4. QA embeddings 过 fc_text ──
            qa_encoded = model.fc_text(x_text)              # [6, hidden_dim]
    
            # ── 5. 余弦相似度 → softmax ──
            qa_norm  = F.normalize(qa_encoded,  dim=1)      # [6, hidden_dim]
            pw_norm  = F.normalize(pathway_mat, dim=1)      # [n_pathways, hidden_dim]
            sim = torch.mm(qa_norm, pw_norm.t())             # [6, n_pathways]
            attn = torch.softmax(sim * 5.0, dim=-1)          # [6, n_pathways]
    
        return attn.cpu().numpy().astype(np.float32), valid_pathway_names

    def _compute_posthoc_patch_pathway(self, output) -> Optional[np.ndarray]:
        """
        Post-hoc 计算 Patch ↔ Pathway 对应关系。
    
        流程：
          1. _temp_path_features → [n_patches, path_hidden_dim]
                                 → rho → [n_patches, hidden_dim]
          2. 对每条 pathway，用 masked omic → fc_omic → [hidden_dim]
          3. 余弦相似度 → softmax → [n_patches, n_pathways]
    
        Returns:
            np.ndarray [n_patches, n_pathways] 或 None
        """
        model  = self.model
        device = next(model.parameters()).device
    
        # ── 1. Per-patch features ─────────────────────────────────
        if not hasattr(model, '_temp_path_features') or not model._temp_path_features:
            print("  ⚠️ [Patch↔Pathway] _temp_path_features not found")
            return None
        patch_feats = model._temp_path_features[0].float().to(device)
        # patch_feats: [n_patches, path_hidden_dim]
    
        # ── 2. 投影到 hidden 空间 ──────────────────────────────────
        if not hasattr(model, 'rho') or model.rho is None:
            print("  ⚠️ [Patch↔Pathway] rho not found")
            return None
        with torch.no_grad():
            patch_projected = model.rho(patch_feats)      # [n_patches, hidden_dim]
    
        # ── 3. 构建 pathway 向量（与 _compute_posthoc_qa_pathway 完全一致）
        if not hasattr(model, '_temp_omic') or model._temp_omic is None:
            print("  ⚠️ [Patch↔Pathway] _temp_omic not found")
            return None
        x_omic = model._temp_omic[0].float().to(device)   # [omic_dim]
    
        if not self.pathway_gene_map or not self.omic_col_names:
            print("  ⚠️ [Patch↔Pathway] No pathway-gene mapping")
            return None
    
        col2idx = {col: i for i, col in enumerate(self.omic_col_names)}
    
        pathway_vecs = []
        valid_pathway_names = []
    
        with torch.no_grad():
            zero_input   = torch.zeros(1, x_omic.shape[0], device=device)
            baseline_vec = model.fc_omic(zero_input)[0]   # [hidden_dim]
    
            for pw_name in self.pathway_names_ordered:
                gene_cols = self.pathway_gene_map[pw_name]
                indices   = [col2idx[c] for c in gene_cols if c in col2idx]
                if not indices:
                    continue
                masked = torch.zeros_like(x_omic)
                masked[indices] = x_omic[indices]
                pw_vec = model.fc_omic(masked.unsqueeze(0))[0] - baseline_vec
                pathway_vecs.append(pw_vec)
                valid_pathway_names.append(pw_name)
    
            if not pathway_vecs:
                print("  ⚠️ [Patch↔Pathway] No valid pathway vectors")
                return None
    
            pathway_mat = torch.stack(pathway_vecs, dim=0)   # [n_pathways, hidden_dim]
    
            # ── 4. 余弦相似度 → softmax ────────────────────────────
            patch_norm   = F.normalize(patch_projected, dim=1)  # [n_patches, hidden_dim]
            pathway_norm = F.normalize(pathway_mat,     dim=1)  # [n_pathways, hidden_dim]
    
            # [n_patches, n_pathways]
            sim  = torch.mm(patch_norm, pathway_norm.t())
            attn = torch.softmax(sim * 5.0, dim=-1)
    
        print(f"  ✅ [Patch↔Pathway] map: {attn.shape}  max={attn.max():.4f}")
        return attn.cpu().numpy().astype(np.float32)
        
    # ─────────────────────────────────────────────────────────────
    # 私有：WSI 文件加载
    # ─────────────────────────────────────────────────────────────

    def _load_thumbnail(self, slide_id: str) -> Optional[np.ndarray]:
        from PIL import Image

        # 候选命名规则（按优先级）
        base_names = [slide_id]
        if slide_id.endswith(".svs"):
            base_names.append(slide_id[:-4])
        if "TCGA-" in slide_id:
            base_names.append("-".join(slide_id.split("-")[:3]))
        parts = slide_id.split(".")
        if len(parts) >= 2:
            base_names.append(parts[-2])

        suffixes = [
            "_original.png", "_original.PNG",
            "_roi.png", "_roi.PNG",
            ".png", ".PNG", ".jpg", ".JPG",
        ]

        for d in self._thumbnail_dirs:
            for name in base_names:
                for suf in suffixes:
                    p = os.path.join(d, name + suf)
                    if os.path.exists(p):
                        try:
                            img = Image.open(p).convert("RGB")
                            arr = np.array(img, dtype=np.uint8)
                            print(f"  ✅ Thumbnail loaded: {os.path.basename(p)} "
                                  f"[{arr.shape[1]}×{arr.shape[0]}]")
                            return arr
                        except Exception as e:
                            print(f"  ⚠️ Failed to open {p}: {e}")
        print(f"  ⚠️ Thumbnail not found for: {slide_id}")
        return None

    def _load_coords(self, slide_id: str) -> Optional[np.ndarray]:
        """按优先级尝试从 CSV / H5 / PT 加载坐标"""
        base = slide_id[:-4] if slide_id.endswith(".svs") else slide_id

        # CSV
        for d in [self._coord_dir, self.preprocessing_dir]:
            for name in [slide_id, base]:
                p = os.path.join(d, name, "coords.csv")
                if not os.path.exists(p):
                    p = os.path.join(d, f"{name}_coords.csv")
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        col_x = next((c for c in df.columns if "x" in c.lower()), None)
                        col_y = next((c for c in df.columns if "y" in c.lower()), None)
                        if col_x and col_y:
                            coords = df[[col_x, col_y]].values.astype(np.float32)
                            print(f"  ✅ Coords loaded from CSV: {coords.shape[0]} patches")
                            return coords
                    except Exception as e:
                        print(f"  ⚠️ CSV coord load failed: {e}")

        # H5
        for d in [self._coord_dir, self.preprocessing_dir]:
            for name in [slide_id, base]:
                for h5name in [f"{name}.h5", f"{name}_patches.h5"]:
                    p = os.path.join(d, h5name)
                    if os.path.exists(p):
                        try:
                            with h5py.File(p, "r") as f:
                                coords = f["coords"][:].astype(np.float32)
                            print(f"  ✅ Coords loaded from H5: {coords.shape[0]} patches")
                            return coords
                        except Exception as e:
                            print(f"  ⚠️ H5 coord load failed: {e}")

        print(f"  ⚠️ Coords not found for: {slide_id}")
        return None

    def _build_thumbnail_from_patches(self, slide_id: str,
                                        coords: np.ndarray,
                                        target_size: int = 1024) -> Optional[np.ndarray]:
        """
        当找不到标准缩略图时，从 patch PNG 目录重建低分辨率缩略图。
        目录结构：{patch_png_dir}/{slide_id}/{x}x_{y}y.png
        """
        from PIL import Image

        base = slide_id[:-4] if slide_id.endswith(".svs") else slide_id
        slide_dir = None
        for name in [slide_id, base]:
            d = os.path.join(self._patch_png_dir, name)
            if os.path.isdir(d):
                slide_dir = d
                break

        if slide_dir is None:
            print(f"  ⚠️ Patch PNG dir not found for: {slide_id}")
            print(f"     Tried: {self._patch_png_dir}/{base}")
            return None

        # 读取所有 patch 文件，解析坐标
        import re
        patch_files = [f for f in os.listdir(slide_dir) if f.endswith(".png")]
        if not patch_files:
            print(f"  ⚠️ No PNG patches found in: {slide_dir}")
            return None

        # 从文件名解析 (x, y)，格式：{x}x_{y}y.png
        patches = []
        for fn in patch_files:
            m = re.match(r"(\d+)x_(\d+)y", fn)
            if m:
                px, py = int(m.group(1)), int(m.group(2))
                patches.append((px, py, os.path.join(slide_dir, fn)))

        if not patches:
            print(f"  ⚠️ Cannot parse patch coordinates from filenames in: {slide_dir}")
            return None

        # 读一个 patch 获取 patch 尺寸
        try:
            sample_patch = Image.open(patches[0][2])
            patch_w, patch_h = sample_patch.size
        except Exception:
            patch_w, patch_h = 256, 256

        # 计算画布范围
        xs = [p[0] for p in patches]
        ys = [p[1] for p in patches]
        x_min, x_max = min(xs), max(xs) + patch_w
        y_min, y_max = min(ys), max(ys) + patch_h
        full_w = x_max - x_min
        full_h = y_max - y_min

        # 缩放因子
        scale = min(target_size / full_w, target_size / full_h)
        canvas_w = int(full_w * scale)
        canvas_h = int(full_h * scale)
        pw = max(1, int(patch_w * scale))
        ph = max(1, int(patch_h * scale))

        canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)  # 浅灰背景

        placed = 0
        for px, py, fpath in patches:
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((pw, ph), Image.BILINEAR)
                cx = int((px - x_min) * scale)
                cy = int((py - y_min) * scale)
                cx2 = min(cx + pw, canvas_w)
                cy2 = min(cy + ph, canvas_h)
                canvas[cy:cy2, cx:cx2] = np.array(img)[:cy2-cy, :cx2-cx]
                placed += 1
            except Exception:
                continue

        print(f"  ✅ Thumbnail rebuilt from {placed}/{len(patches)} patches "
              f"[{canvas_w}×{canvas_h}]  [{slide_dir}]")
        return canvas

    # ─────────────────────────────────────────────────────────────
    # 工具
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_explanation_raw(output):
        if not isinstance(output, dict):
            return None
        exp = output.get("explanation")
        if exp is None:
            return None
        if isinstance(exp, dict) and "raw" in exp:
            return exp["raw"]
        return exp

    def _compute_ig_scores(self, data_WSI, data_omic, data_text,
                        survival_time=None, event=None,   # ← 新增两个参数
                        n_steps=50) -> Optional[np.ndarray]:
        """用 Integrated Gradients 计算每个 patch 对 risk 预测的直接贡献"""
        if data_WSI is None:
            return None
        try:
            # 统一格式
            if isinstance(data_WSI, list):
                x = data_WSI[0]
            else:
                x = data_WSI
            if x.dim() == 3:
                x = x.squeeze(0)
            x = x.to(self.device).float()   # [N, D]
    
            # baseline：全零（代表"无信息 patch"）
            x_base = torch.zeros_like(x)
            alphas  = torch.linspace(0, 1, n_steps + 1, device=self.device)
            grad_sum = torch.zeros_like(x)
    
            self.model.eval()
            print(f"  🔬 Computing IG scores ({n_steps} steps)...")
    
            for alpha in alphas:
                x_interp = (x_base + alpha * (x - x_base)).detach()
                x_interp.requires_grad_(True)
    
                # 前向（带梯度，不用 no_grad）
                kwargs = {
                    'x_path':          [x_interp],
                    'compute_loss':    False,
                    'return_features': True,
                }
                if data_omic is not None:
                    xo = data_omic
                    if xo.dim() == 1: xo = xo.unsqueeze(0)
                    kwargs['x_omic'] = xo.to(self.device).float()
                if data_text is not None:
                    xt = data_text
                    if xt.dim() == 2: xt = xt.unsqueeze(0)
                    kwargs['x_text'] = xt.to(self.device).float()

                # ↓ 新增这两段，传入生存时间
                if survival_time is not None:
                    st = torch.tensor([survival_time], dtype=torch.float32) \
                         if not isinstance(survival_time, torch.Tensor) else survival_time
                    if st.dim() == 0: st = st.unsqueeze(0)
                    kwargs['survival_time'] = st.to(self.device).float()
                if event is not None:
                    ev = torch.tensor([event], dtype=torch.float32) \
                         if not isinstance(event, torch.Tensor) else event
                    if ev.dim() == 0: ev = ev.unsqueeze(0)
                    kwargs['event'] = ev.to(self.device).float()
    
                output = self.model(**kwargs)
    
                # 提取 risk scalar
                logits = output['logits'] if isinstance(output, dict) else output
                """
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -S.sum()   # 标量
                """
                """
                # 改为：对 WSI pathway 自己的特征范数求梯度
                # 这样梯度 100% 来自 patch，不受 omic/text 稀释
                path_feature = output.get('modal_features', {}).get('path', None)
                if path_feature is not None:
                    scalar = path_feature.norm()   # WSI 特征的 L2 范数作为优化目标
                else:
                    print("还是用了fallbackkkkkkkkkkkkkkkkkkkk!")
                    # fallback：还是用 risk
                    logits = output['logits']
                    hazards = torch.sigmoid(logits)
                    S = torch.cumprod(1 - hazards, dim=1)
                    scalar = -S.sum()
                """
                # 自动根据 task_type 决定
                if getattr(self, 'task_type', 'survival') == 'survival':
                    hazards = torch.sigmoid(logits)
                    S = torch.cumprod(1 - hazards, dim=1)
                    scalar = -S.sum()
                else:
                    pred_class = logits.argmax(dim=-1).item()
                    scalar = logits[0, pred_class]
    
                self.model.zero_grad()
                scalar.backward()
                grad_sum += x_interp.grad.detach()
    
            # IG = (x - baseline) * 平均梯度，取各特征维度之和
            avg_grads = grad_sum / (n_steps + 1)
            ig_attr   = (x - x_base) * avg_grads       # [N, D]
            ig_raw    = ig_attr.sum(dim=-1).cpu().numpy()  # [N]
    
            # 只保留正贡献，归一化到 [0, 1]
            ig_pos = np.clip(ig_raw, 0, None)
            lo, hi = ig_pos.min(), ig_pos.max()
            ig_norm = (ig_pos - lo) / (hi - lo + 1e-8)
    
            print(f"  ✅ IG done | top-5: {sorted(ig_norm, reverse=True)[:5]}")
            return ig_norm.astype(np.float32)
    
        except Exception as e:
            print(f"  ⚠️ IG computation failed: {e}")
            import traceback; traceback.print_exc()
            return None

    def _compute_omic_ig_scores(
        self,
        data_omic,
        data_WSI        = None,
        data_text       = None,
        survival_time   = None,
        event           = None,
        n_steps: int    = 50,
    ) -> Optional[np.ndarray]:
        """
        对每个 omic 特征（基因）计算 Integrated Gradients 分数。
    
        原理：baseline = 全零向量，在 baseline→input 之间均匀插值 n_steps 步，
              对每步的梯度求平均，乘以 (input - baseline) 得到 IG 归因分数，
              取绝对值后归一化到 [0,1]。
    
        Returns
        -------
        np.ndarray  shape=[omic_dim]，每个基因特征的重要性，值越大越重要
        """
        try:
            # ── 准备 omic 张量 ────────────────────────────────────────
            x = data_omic.clone().detach() if isinstance(data_omic, torch.Tensor) \
                else torch.tensor(data_omic, dtype=torch.float32)
            x = x.to(self.device).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)   # [1, omic_dim]
    
            x_base = torch.zeros_like(x)
            alphas  = torch.linspace(0, 1, n_steps + 1, device=self.device)
            grad_sum = torch.zeros_like(x)
    
            self.model.eval()
    
            for alpha in alphas:
                x_interp = (x_base + alpha * (x - x_base)).detach()
                x_interp.requires_grad_(True)
    
                # ── 组装 kwargs ───────────────────────────────────────
                kwargs = {
                    "x_omic":          x_interp,
                    "compute_loss":    False,
                    "return_features": False,
                }
                if data_WSI is not None:
                    wsi = data_WSI.clone().detach() if isinstance(data_WSI, torch.Tensor) \
                          else torch.tensor(data_WSI, dtype=torch.float32)
                    kwargs["x_path"] = [wsi.to(self.device).float()]
    
                if data_text is not None:
                    xt = data_text.clone().detach() if isinstance(data_text, torch.Tensor) \
                         else torch.tensor(data_text, dtype=torch.float32)
                    xt = xt.to(self.device).float()
                    if xt.dim() == 2:
                        xt = xt.unsqueeze(0)
                    kwargs["x_text"] = xt
    
                if survival_time is not None:
                    st = torch.tensor([float(survival_time)],
                                      dtype=torch.float32, device=self.device)
                    kwargs["survival_time"] = st
    
                if event is not None:
                    ev = torch.tensor([float(event)],
                                      dtype=torch.float32, device=self.device)
                    kwargs["event"] = ev
    
                # ── 前向 ──────────────────────────────────────────────
                output = self.model(**kwargs)
    
                # ── 提取 scalar ───────────────────────────────────────
                logits = output["logits"] if isinstance(output, dict) else output
    
                task_type = getattr(self, "task_type", "survival")
                if task_type == "survival":
                    hazards = torch.sigmoid(logits)
                    S       = torch.cumprod(1 - hazards, dim=1)
                    scalar  = -S.sum()
                else:
                    pred_class = logits.argmax(dim=-1).item()
                    scalar     = logits[0, pred_class]
    
                # ── 反向 ──────────────────────────────────────────────
                self.model.zero_grad()
                scalar.backward()
    
                if x_interp.grad is not None:
                    grad_sum += x_interp.grad.detach()
    
            # ── 积分梯度计算 ──────────────────────────────────────────
            avg_grads = grad_sum / (n_steps + 1)
            ig_attr   = ((x - x_base) * avg_grads).squeeze(0)   # [omic_dim]
            ig_scores = ig_attr.abs().cpu().numpy().astype(np.float32)
    
            # ── 归一化到 [0,1] ────────────────────────────────────────
            lo, hi = ig_scores.min(), ig_scores.max()
            ig_norm = (ig_scores - lo) / (hi - lo + 1e-8)
    
            print(f"  ✅ Omic IG computed: {len(ig_norm)} features, "
                  f"max={ig_norm.max():.4f}, mean={ig_norm.mean():.4f}")
            return ig_norm
    
        except Exception as e:
            import traceback
            print(f"  ⚠️ Omic IG computation failed: {e}")
            traceback.print_exc()
            return None
