"""
GRAMPorpoiseMMF - 修复版本
==========================
保留代码1的生存分析和分类功能
保留代码2的多标签改进 (Gram-Gated Concatenation + Asymmetric Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np
  
# ========== 可解释性专用编码器 ==========

class QALevelTextEncoder(nn.Module):
    """
    将整体文本向量分解为6个QA级别的表示
    """
    def __init__(self, text_dim=768, n_qa_pairs=6, output_dim=256):
        super().__init__()
        self.n_qa_pairs = n_qa_pairs
        self.output_dim = output_dim 
        
        # 方案1: 学习一个分解器（从整体文本反推6个QA）
        self.decomposer = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn. Linear(text_dim * 2, n_qa_pairs * output_dim)
        )
        
        # 方案2: 如果有独立的6个QA embedding，直接投影
        self.qa_projector = nn.Linear(text_dim, output_dim)
    
    def forward(self, text_feat, qa_embeddings=None):
        """
        Args:
            text_feat: [B, text_dim] - BioBERT编码的整体文本
            qa_embeddings: [B, 6, text_dim] - (可选) 6个QA的独立编码
        Returns:
            qa_features: [B, 6, output_dim]
        """
        batch_size = text_feat.size(0)
        
        if qa_embeddings is not None:
            # 如果有独立的QA编码，直接投影
            qa_features = self.qa_projector(qa_embeddings)  # [B, 6, output_dim]
        else:
            # 否则从整体文本分解
            decomposed = self.decomposer(text_feat)  # [B, 6*output_dim]
            qa_features = decomposed.view(batch_size, self.n_qa_pairs, self.output_dim)
        
        return qa_features


class PathwayLevelOmicEncoder(nn.Module):
    """
    将基因表达编码为通路级别的表示
    """
    def __init__(self, gene_dim=2007, n_pathways=50, pathway_dim=256, 
                 pathway_gene_mapping=None):
        super().__init__()
        self.n_pathways = n_pathways
        self.pathway_dim = pathway_dim
        
        # 如果提供了通路-基因映射，使用加权聚合
        if pathway_gene_mapping is not None:
            # pathway_gene_mapping: [n_pathways, gene_dim] - 每个通路的基因掩码
            self.register_buffer('pathway_mask', pathway_gene_mapping)
            self.use_mapping = True
        else:
            # 否则，学习一个软映射
            self.gene_to_pathway = nn.Linear(gene_dim, n_pathways * pathway_dim)
            self.use_mapping = False
        
        # 通路特征投影
        self.pathway_encoder = nn.Sequential(
            nn.Linear(gene_dim if self.use_mapping else pathway_dim, pathway_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, gene_expressions, pathway_scores=None):
        """
        Args:
            gene_expressions: [B, gene_dim] - 原始基因表达
            pathway_scores: [B, n_pathways] - (可选) 预计算的通路得分(如GSVA)
        Returns:
            pathway_features: [B, n_pathways, pathway_dim]
        """
        batch_size = gene_expressions.size(0)
        
        if pathway_scores is not None:
            # 如果有预计算的通路得分，直接使用
            pathway_features = self.pathway_encoder(
                pathway_scores.unsqueeze(-1).expand(-1, -1, gene_expressions.size(1))
            )
        elif self.use_mapping:
            # 使用通路-基因映射加权聚合
            pathway_features = []
            for i in range(self.n_pathways):
                # 提取该通路的基因表达
                pathway_genes = gene_expressions * self.pathway_mask[i].unsqueeze(0)
                pathway_feat = self.pathway_encoder(pathway_genes)
                pathway_features.append(pathway_feat)
            pathway_features = torch.stack(pathway_features, dim=1)  # [B, n_pathways, pathway_dim]
        else:
            # 学习软映射
            mapped = self.gene_to_pathway(gene_expressions)  # [B, n_pathways*pathway_dim]
            pathway_features = mapped.view(batch_size, self.n_pathways, self.pathway_dim)
        
        return pathway_features


class TrimodalExplainabilityModule(nn.Module):
    """
    三模态可解释性模块（手动注意力实现）
    """
    def __init__(self, feature_dim=256, n_qa_pairs=6, n_pathways=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_qa_pairs = n_qa_pairs
        self.n_pathways = n_pathways
        
        # 🔥 手动注意力：Query/Key/Value投影
        self.qa_to_pathway_query = nn.Linear(feature_dim, feature_dim)
        self.pathway_key = nn.Linear(feature_dim, feature_dim)
        self.pathway_value = nn.Linear(feature_dim, feature_dim)
        
        self.qa_to_patch_query = nn.Linear(feature_dim, feature_dim)
        self.patch_key = nn.Linear(feature_dim, feature_dim)
        self.patch_value = nn.Linear(feature_dim, feature_dim)
        
        self.patch_to_pathway_query = nn.Linear(feature_dim, feature_dim)
        
        # 🔥🔥🔥 调整温度
        # 不使用标准的 d^-0.5 缩放，使用更大的值
        self.scale = 2.0  # 🔥 改这里！原来是 feature_dim ** -0.5 ≈ 0.0625
        
        print(f"✅ TrimodalExplainabilityModule initialized with manual attention")
        print(f"   feature_dim: {feature_dim}, scale: {self.scale}")
    
    def forward(self, qa_features, pathway_features, patch_features, fusion_weights=None):
        """
        手动计算注意力
        
        Args:
            qa_features: [B, 6, 256]
            pathway_features: [B, 50, 256]
            patch_features: [B, N, 256]
        
        Returns:
            explanation dict
        """
        batch_size = qa_features.size(0)
        
        # 🔥 输入检查
        print(f"\n     🔍 Explainability Module Inputs:")
        print(f"        qa: std={qa_features.std():.4f}")
        print(f"        pathway: std={pathway_features.std():.4f}")
        print(f"        patch: std={patch_features.std():.4f}")
        
        # === 1. QA ↔ Pathway Attention (手动实现) ===
        Q_qa = self.qa_to_pathway_query(qa_features)      # [B, 6, 256]
        K_pathway = self.pathway_key(pathway_features)    # [B, 50, 256]
        V_pathway = self.pathway_value(pathway_features)  # [B, 50, 256]
        
        # 计算attention scores
        # [B, 6, 256] @ [B, 256, 50] = [B, 6, 50]
        qa2pathway_scores = torch.bmm(Q_qa, K_pathway.transpose(1, 2)) * self.scale
        
        print(f"\n     🔍 QA->Pathway Attention:")
        print(f"        Scores (before softmax): min={qa2pathway_scores.min():.4f}, max={qa2pathway_scores.max():.4f}, std={qa2pathway_scores.std():.4f}")
        
        # Softmax
        qa2pathway_attn = F.softmax(qa2pathway_scores, dim=-1)  # [B, 6, 50]
        
        print(f"        Attention (after softmax): min={qa2pathway_attn.min():.6f}, max={qa2pathway_attn.max():.6f}, std={qa2pathway_attn.std():.6f}")
        print(f"        Unique values: {len(torch.unique(qa2pathway_attn))}")
        
        # Weighted sum
        qa2pathway_output = torch.bmm(qa2pathway_attn, V_pathway)  # [B, 6, 256]
        
        # === 2. QA ↔ Patch Attention ===
        Q_qa_patch = self.qa_to_patch_query(qa_features)  # [B, 6, 256]
        K_patch = self.patch_key(patch_features)          # [B, N, 256]
        V_patch = self.patch_value(patch_features)        # [B, N, 256]
        
        # [B, 6, 256] @ [B, 256, N] = [B, 6, N]
        qa2patch_scores = torch.bmm(Q_qa_patch, K_patch.transpose(1, 2)) * self.scale
        qa2patch_attn = F.softmax(qa2patch_scores, dim=-1)
        
        print(f"\n     🔍 QA->Patch Attention:")
        print(f"        Scores std: {qa2patch_scores.std():.4f}")
        print(f"        Attention std: {qa2patch_attn.std():.6f}")
        
        qa2patch_output = torch.bmm(qa2patch_attn, V_patch)
        
        # === 3. Patch ↔ Pathway Correspondence ===
        Q_patch = self.patch_to_pathway_query(patch_features)  # [B, N, 256]
        
        # [B, N, 256] @ [B, 256, 50] = [B, N, 50]
        patch2pathway_scores = torch.bmm(Q_patch, K_pathway.transpose(1, 2)) * self.scale
        patch2pathway_corr = F.softmax(patch2pathway_scores, dim=-1)
        
        print(f"\n     🔍 Patch->Pathway Correspondence:")
        print(f"        Correlation std: {patch2pathway_corr.std():.6f}")
        
        # === 返回结果 ===
        explanation = {
            'qa2pathway_attention': qa2pathway_attn,           # [B, 6, 50]
            'qa2patch_attention': qa2patch_attn,               # [B, 6, N]
            'patch_pathway_correspondence': patch2pathway_corr, # [B, N, 50]
            'qa_features': qa_features,
            'pathway_features': pathway_features,
            'patch_features': patch_features,
            'fusion_weights': fusion_weights,
        }
        
        return explanation
    
    def get_top_k_explanations(self, explanation, k=5, qa_texts=None, pathway_names=None):
        """
        提取Top-K解释
        """
        qa2pathway_attn = explanation['qa2pathway_attention'][0].cpu().numpy()  # [6, 50]
        qa2patch_attn = explanation['qa2patch_attention'][0].cpu().numpy()      # [6, N]
        
        results = []
        
        for qa_idx in range(self.n_qa_pairs):
            qa_text = qa_texts[qa_idx] if qa_texts else f"QA_{qa_idx}"
            
            # Top-K pathways
            pathway_scores = qa2pathway_attn[qa_idx]
            top_pathway_indices = np.argsort(pathway_scores)[-k:][::-1]
            top_pathways = [
                {
                    'pathway_idx': int(idx),
                    'pathway_name': pathway_names[idx] if pathway_names else f"Pathway_{idx}",
                    'score': float(pathway_scores[idx])
                }
                for idx in top_pathway_indices
            ]
            
            # Top-K patches
            patch_scores = qa2patch_attn[qa_idx]
            top_patch_indices = np.argsort(patch_scores)[-k:][::-1]
            top_patches = [
                {
                    'patch_idx': int(idx),
                    'score': float(patch_scores[idx])
                }
                for idx in top_patch_indices
            ]
            
            results.append({
                'qa_idx': qa_idx,
                'qa_text': qa_text,
                'top_pathways': top_pathways,
                'top_patches': top_patches
            })
        
        return results


# ========== 生存分析专用GRAM融合模块 ==========

class SurvivalAwareVolumeCalculator(nn.Module):
    """
    生存分析专用的Volume计算器
    考虑时间维度的几何关系
    """
    def __init__(self, feature_dim, normalize=True, epsilon=1e-8):
        super().__init__()
        self.normalize = normalize
        self.epsilon = epsilon
        
        # 🔥 添加梯度裁剪
        self.max_volume = 50.0  # 限制volume的最大值
          
        # 时间感知的特征投影 - 🔥 添加LayerNorm稳定训练
        self.time_aware_projection = nn.Sequential(
            nn.Linear(feature_dim + 64, feature_dim),
            nn.LayerNorm(feature_dim),  # 🔥 添加
            nn.Tanh()
        )
    
    def compute_time_aware_gram(self, v1, v2, v3, time_emb):
        """
        计算时间感知的Gram矩阵
        """
        batch_size = v1.size(0)
        
        # 🔥 添加特征裁剪，防止极端值
        v1 = torch.clamp(v1, -10, 10)
        v2 = torch.clamp(v2, -10, 10)
        v3 = torch.clamp(v3, -10, 10)
        time_emb = torch.clamp(time_emb, -10, 10)
        
        # 将时间信息融入每个模态
        v1_time = self.time_aware_projection(torch.cat([v1, time_emb], dim=1))
        v2_time = self.time_aware_projection(torch.cat([v2, time_emb], dim=1))
        v3_time = self.time_aware_projection(torch.cat([v3, time_emb], dim=1))
        
        # 🔥 再次裁剪投影后的特征
        v1_time = torch.clamp(v1_time, -10, 10)
        v2_time = torch.clamp(v2_time, -10, 10)
        v3_time = torch.clamp(v3_time, -10, 10)
        
        # 构建Gram矩阵
        A = torch.stack([v1_time, v2_time, v3_time], dim=-1)  # [B, D, 3]
        G = torch.bmm(A.transpose(1, 2), A)  # [B, 3, 3]
        
        return G, v1_time, v2_time, v3_time
    
    def compute_survival_volume(self, G, event, survival_time):
        """
        计算考虑删失和生存时间的Volume
        🔥 添加数值稳定性检查
        """
        # 基础volume
        det_G = torch.det(G)
        
        # 🔥 关键修复：处理负的行列式（数值误差导致）
        det_G = torch.abs(det_G)  # 取绝对值
        det_G = torch.clamp(det_G, min=self.epsilon, max=self.max_volume**2)
        
        base_volume = torch.sqrt(det_G)
        
        # 🔥 检查NaN
        if torch.isnan(base_volume).any():
            print(f"⚠️ Warning: NaN detected in base_volume")
            base_volume = torch.nan_to_num(base_volume, nan=1.0)
        
        # 🔥 归一化生存时间，避免极端值
        time_normalized = survival_time / (survival_time.max() + self.epsilon)
        time_normalized = torch.clamp(time_normalized, 0, 1)
        
        # 时间加权：生存时间越长，volume应该越小（表示更一致的预测）
        time_weight = torch.sigmoid(-time_normalized * 5.0)  # 🔥 降低系数，从10降到5
        
        # 事件加权：发生事件的样本volume应该更小（需要更准确）
        event_weight = event.float() * 0.3 + 0.7  # 🔥 降低权重差异，从0.5改为0.3
        
        # 综合volume
        weighted_volume = base_volume * (0.5 + 0.5 * time_weight * event_weight)  # 🔥 限制权重范围
        
        # 🔥 最终裁剪
        weighted_volume = torch.clamp(weighted_volume, self.epsilon, self.max_volume)
        
        return weighted_volume, base_volume
    
    def forward(self, feat1, feat2, feat3, time_emb, event=None, survival_time=None):
        """
        前向传播 - 🔥 添加完整的NaN检查
        """
        # 🔥 输入检查
        if torch.isnan(feat1).any() or torch.isnan(feat2).any() or torch.isnan(feat3).any():
            print("⚠️ Warning: NaN in input features")
            feat1 = torch.nan_to_num(feat1, nan=0.0)
            feat2 = torch.nan_to_num(feat2, nan=0.0)
            feat3 = torch.nan_to_num(feat3, nan=0.0)
        
        if time_emb is not None and torch.isnan(time_emb).any():
            print("⚠️ Warning: NaN in time_emb")
            time_emb = torch.nan_to_num(time_emb, nan=0.0)
        
        if self.normalize:
            feat1 = F.normalize(feat1, dim=1, eps=self.epsilon)
            feat2 = F.normalize(feat2, dim=1, eps=self.epsilon)
            feat3 = F.normalize(feat3, dim=1, eps=self.epsilon)
        
        # 计算时间感知的Gram矩阵
        G, v1_time, v2_time, v3_time = self.compute_time_aware_gram(
            feat1, feat2, feat3, time_emb
        )
        
        # 🔥 检查Gram矩阵
        if torch.isnan(G).any():
            print("⚠️ Warning: NaN in Gram matrix")
            G = torch.nan_to_num(G, nan=1.0)
        
        # 计算volume
        if event is not None and survival_time is not None:
            weighted_volume, base_volume = self.compute_survival_volume(
                G, event, survival_time
            )
        else:
            det_G = torch.det(G)
            det_G = torch.abs(det_G)  # 🔥 取绝对值
            det_G = torch.clamp(det_G, min=self.epsilon, max=self.max_volume**2)
            weighted_volume = torch.sqrt(det_G)
            base_volume = weighted_volume
        
        # 🔥 最终NaN检查
        if torch.isnan(weighted_volume).any():
            print("⚠️ Warning: NaN in weighted_volume, replacing with 1.0")
            weighted_volume = torch.nan_to_num(weighted_volume, nan=1.0)
        
        # 对齐分数：volume越小，对齐越好
        alignment_score = torch.sigmoid(-weighted_volume * 5.0 + 2.5)  # 🔥 降低系数
        
        result = {
            'volume': weighted_volume,
            'base_volume': base_volume,
            'alignment_score': alignment_score,
            'gram_matrix': G,
            'time_aware_features': {
                'v1': v1_time,
                'v2': v2_time,
                'v3': v3_time
            }
        }
        
        return result

class SurvivalVolumeGuidedFusion(nn.Module):
    """
    生存分析专用的Volume引导融合
    """
    def __init__(self, feature_dim, time_emb_dim=64, use_risk_aware=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.time_emb_dim = time_emb_dim
        self.use_risk_aware = use_risk_aware
        
        # 生存感知的Volume计算器
        self.volume_calculator = SurvivalAwareVolumeCalculator(
            feature_dim=feature_dim,
            normalize=True
        )
        
        # 🔥 添加LayerNorm
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 3 + time_emb_dim + 2, 256),
            nn.LayerNorm(256),  # 🔥 添加
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # 🔥 添加
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
        if use_risk_aware:
            self.risk_confidence_scorer = nn.Sequential(
                nn.Linear(feature_dim + time_emb_dim, 128),
                nn.LayerNorm(128),  # 🔥 添加
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def compute_risk_aware_weights(self, feat1, feat2, feat3, volume_info, 
                                   time_emb, risk_score=None):
        """
        计算风险感知的融合权重
        """
        batch_size = feat1.size(0)
        volume = volume_info['volume']
        
        # 🔥 输入检查
        if torch.isnan(volume).any():
            print("⚠️ Warning: NaN in volume")
            volume = torch.nan_to_num(volume, nan=1.0)
        
        # 如果没有提供risk_score，使用volume作为代理
        if risk_score is None:
            risk_score = volume
        
        # 🔥 检查risk_score
        if torch.isnan(risk_score).any():
            print("⚠️ Warning: NaN in risk_score")
            risk_score = torch.nan_to_num(risk_score, nan=0.0)
        
        # 🔥 裁剪所有输入
        feat1 = torch.clamp(feat1, -10, 10)
        feat2 = torch.clamp(feat2, -10, 10)
        feat3 = torch.clamp(feat3, -10, 10)
        time_emb = torch.clamp(time_emb, -10, 10)
        volume = torch.clamp(volume, -10, 10)
        risk_score = torch.clamp(risk_score, -10, 10)
        
        # 构建输入特征
        combined = torch.cat([
            feat1, feat2, feat3, 
            time_emb,
            volume.unsqueeze(1),
            risk_score.unsqueeze(1) if risk_score.dim() == 1 else risk_score
        ], dim=1)
        
        # 🔥 检查combined
        if torch.isnan(combined).any():
            print("⚠️ Warning: NaN in combined features")
            combined = torch.nan_to_num(combined, nan=0.0)
        
        # 生成权重
        weights = self.weight_generator(combined)
        
        # 🔥 检查weights
        if torch.isnan(weights).any():
            print("⚠️ Warning: NaN in weights, using uniform")
            weights = torch.ones_like(weights) / 3.0
        
        # 如果使用风险感知，调整权重
        if self.use_risk_aware and hasattr(self, 'risk_confidence_scorer'):
            conf1 = self.risk_confidence_scorer(torch.cat([feat1, time_emb], dim=1))
            conf2 = self.risk_confidence_scorer(torch.cat([feat2, time_emb], dim=1))
            conf3 = self.risk_confidence_scorer(torch.cat([feat3, time_emb], dim=1))
            
            confidences = torch.cat([conf1, conf2, conf3], dim=1)
            
            # 🔥 检查confidences
            if torch.isnan(confidences).any():
                print("⚠️ Warning: NaN in confidences")
                confidences = torch.ones_like(confidences)
            
            # 用置信度调整权重
            weights = weights * confidences
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # 🔥 添加epsilon
        
        return weights
    
    def forward(self, feat_path, feat_omic, feat_text, time_emb, 
                event=None, survival_time=None, risk_score=None):
        """
        执行生存感知的Volume引导融合
        """
        # 计算时间感知的volume
        volume_info = self.volume_calculator(
            feat_path, feat_omic, feat_text, time_emb,
            event=event, survival_time=survival_time
        )
        
        # 使用时间感知的特征
        feat_path_time = volume_info['time_aware_features']['v1']
        feat_omic_time = volume_info['time_aware_features']['v2']
        feat_text_time = volume_info['time_aware_features']['v3']
        
        # 🔥 检查时间感知特征
        if torch.isnan(feat_path_time).any() or torch.isnan(feat_omic_time).any() or torch.isnan(feat_text_time).any():
            print("⚠️ Warning: NaN in time-aware features")
            feat_path_time = torch.nan_to_num(feat_path_time, nan=0.0)
            feat_omic_time = torch.nan_to_num(feat_omic_time, nan=0.0)
            feat_text_time = torch.nan_to_num(feat_text_time, nan=0.0)
        
        # 计算风险感知的权重
        weights = self.compute_risk_aware_weights(
            feat_path_time, feat_omic_time, feat_text_time,
            volume_info, time_emb, risk_score
        )
        
        # 加权融合
        fused = (
            weights[:, 0:1] * feat_path_time +
            weights[:, 1:2] * feat_omic_time +
            weights[:, 2:3] * feat_text_time
        )
        
        # 🔥 检查融合结果
        if torch.isnan(fused).any():
            print("⚠️ Warning: NaN in fused features")
            fused = torch.nan_to_num(fused, nan=0.0)
        
        # 返回详细信息
        info = {
            'volume': volume_info['volume'],
            'base_volume': volume_info['base_volume'],
            'alignment_score': volume_info['alignment_score'],
            'fusion_weights': weights,
            'gram_matrix': volume_info['gram_matrix']
        }
        
        return fused, info


class SurvivalTaskAdaptiveFusion(nn.Module):
    """
    生存分析任务自适应融合
    """
    def __init__(self, feature_dim, time_emb_dim=64):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.time_emb_dim = time_emb_dim
        
        self.volume_fusion = SurvivalVolumeGuidedFusion(
            feature_dim=feature_dim,
            time_emb_dim=time_emb_dim,
            use_risk_aware=True
        )
        
        # 🔥 降低正则化权重
        self.volume_reg_weight = 0.01  # 从0.05降到0.01
        self.risk_consistency_weight = 0.01  # 从0.03降到0.01
    
    def compute_volume_regularization(self, volume_info, event, survival_time):
        """
        计算生存任务的Volume正则化损失
        🔥 添加数值稳定性
        """
        volume = volume_info['volume']
        
        # 🔥 检查输入
        if torch.isnan(volume).any():
            print("⚠️ Warning: NaN in volume for regularization")
            return torch.tensor(0.0, device=volume.device)
        
        # 🔥 归一化生存时间
        time_normalized = survival_time / (survival_time.max() + 1e-8)
        time_normalized = torch.clamp(time_normalized, 0, 1)
        
        # 基于生存时间和事件状态定义风险层级
        risk_labels = (event.float() * (1.0 - time_normalized)).detach()
        
        # 🔥 检查risk_labels
        if torch.isnan(risk_labels).any():
            print("⚠️ Warning: NaN in risk_labels")
            return torch.tensor(0.0, device=volume.device)
        
        # 对风险分数进行分层
        try:
            risk_quantiles = torch.quantile(risk_labels, torch.tensor([0.33, 0.67], device=volume.device))
        except:
            return torch.tensor(0.0, device=volume.device)
        
        low_risk = risk_labels <= risk_quantiles[0]
        high_risk = risk_labels >= risk_quantiles[1]
        mid_risk = ~(low_risk | high_risk)
        
        reg_loss = 0.0
        count = 0
        
        # 同层内volume应该小
        for mask in [low_risk, mid_risk, high_risk]:
            if mask.sum() > 1:
                intra_volume = volume[mask].mean()
                
                # 🔥 检查NaN
                if not torch.isnan(intra_volume):
                    reg_loss += intra_volume
                    count += 1
        
        if count > 0:
            reg_loss = reg_loss / count
            # 🔥 裁剪正则化损失
            reg_loss = torch.clamp(reg_loss, 0, 10.0)
        else:
            reg_loss = torch.tensor(0.0, device=volume.device)
        
        return reg_loss * self.volume_reg_weight
    
    def compute_risk_consistency_loss(self, volume_info, risk_score, event):
        """
        计算风险一致性损失
        🔥 添加数值稳定性
        """
        volume = volume_info['volume']
        
        # 🔥 检查输入
        if torch.isnan(volume).any() or torch.isnan(risk_score).any():
            print("⚠️ Warning: NaN in volume or risk_score for consistency loss")
            return torch.tensor(0.0, device=volume.device)
        
        event_mask = event.bool()
        
        if event_mask.sum() > 1:
            event_volume = volume[event_mask]
            event_risk = risk_score[event_mask]
            
            # 🔥 检查是否有有效数据
            if event_volume.numel() == 0 or event_risk.numel() == 0:
                return torch.tensor(0.0, device=volume.device)
            
            # volume小的样本，risk方差应该小
            volume_weights = torch.softmax(-event_volume, dim=0)
            
            # 🔥 检查weights
            if torch.isnan(volume_weights).any():
                return torch.tensor(0.0, device=volume.device)
            
            weighted_risk_var = torch.sum(
                volume_weights * (event_risk - event_risk.mean()) ** 2
            )
            
            # 🔥 检查结果
            if torch.isnan(weighted_risk_var):
                return torch.tensor(0.0, device=volume.device)
            
            # 🔥 裁剪
            weighted_risk_var = torch.clamp(weighted_risk_var, 0, 10.0)
            
            return weighted_risk_var * self.risk_consistency_weight
        
        return torch.tensor(0.0, device=volume.device)
    
    def forward(self, feat_path, feat_omic, feat_text, time_emb,
                event=None, survival_time=None, risk_score=None):
        """
        任务自适应融合
        """
        # 生存感知的融合
        fused, volume_info = self.volume_fusion(
            feat_path, feat_omic, feat_text, time_emb,
            event=event, survival_time=survival_time,
            risk_score=risk_score
        )
        
        # 计算正则化损失
        reg_loss = torch.tensor(0.0, device=fused.device)
        risk_cons_loss = torch.tensor(0.0, device=fused.device)
        
        if event is not None and survival_time is not None:
            try:
                reg_loss = self.compute_volume_regularization(
                    volume_info, event, survival_time
                )
            except Exception as e:
                print(f"⚠️ Warning: Error in volume regularization: {e}")
                reg_loss = torch.tensor(0.0, device=fused.device)
            
            if risk_score is not None:
                try:
                    risk_cons_loss = self.compute_risk_consistency_loss(
                        volume_info, risk_score, event
                    )
                except Exception as e:
                    print(f"⚠️ Warning: Error in risk consistency loss: {e}")
                    risk_cons_loss = torch.tensor(0.0, device=fused.device)
        
        total_reg_loss = reg_loss + risk_cons_loss
        
        # 🔥 最终检查
        if torch.isnan(total_reg_loss):
            print("⚠️ Warning: NaN in total_reg_loss")
            total_reg_loss = torch.tensor(0.0, device=fused.device)
        
        return fused, volume_info, total_reg_loss


# ========== GRAM-Fusion核心模块 (来自代码1) ==========

class GramVolumeCalculator(nn.Module):
    """Gram矩阵Volume计算器"""
    
    def __init__(self, normalize=True, epsilon=1e-8):
        super().__init__()
        self.normalize = normalize
        self.epsilon = epsilon
    
    def compute_gram_matrix(self, v1, v2, v3):
        """计算Gram矩阵"""
        batch_size = v1.size(0)
        A = torch.stack([v1, v2, v3], dim=-1)  # [B, D, 3]
        G = torch.bmm(A.transpose(1, 2), A)  # [B, 3, 3]
        return G
    
    def compute_volume_from_gram(self, G):
        """从Gram矩阵计算Volume"""
        det_G = torch.det(G)  # [B]
        det_G = torch.clamp(det_G, min=self.epsilon)
        volume = torch.sqrt(det_G)
        return volume
    
    def compute_pairwise_volumes(self, v1, v2, v3):
        """计算两两Volume"""
        def pairwise_vol(a, b):
            norm_a = torch.norm(a, dim=1)
            norm_b = torch.norm(b, dim=1)
            dot_ab = (a * b).sum(dim=1)
            vol_squared = norm_a**2 * norm_b**2 - dot_ab**2
            vol_squared = torch.clamp(vol_squared, min=self.epsilon)
            return torch.sqrt(vol_squared)
        
        vol_12 = pairwise_vol(v1, v2)
        vol_13 = pairwise_vol(v1, v3)
        vol_23 = pairwise_vol(v2, v3)
        
        return vol_12, vol_13, vol_23
    
    def forward(self, feat1, feat2, feat3, return_pairwise=False):
        """完整的Volume计算"""
        if self.normalize:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            feat3 = F.normalize(feat3, dim=1)
        
        G = self.compute_gram_matrix(feat1, feat2, feat3)
        volume = self.compute_volume_from_gram(G)
        alignment_score = torch.sigmoid(-volume * 10.0 + 5.0)
        
        result = {
            'volume': volume,
            'alignment_score': alignment_score
        }
        
        if return_pairwise:
            vol_12, vol_13, vol_23 = self.compute_pairwise_volumes(feat1, feat2, feat3)
            result['pairwise_volumes'] = {
                'path_omic': vol_12,
                'path_text': vol_13,
                'omic_text': vol_23
            }
        
        return result


class VolumeGuidedFusion(nn.Module):
    """Volume引导的自适应融合"""
    
    def __init__(self, feature_dim, use_mlp=True, temperature=1.0, primary_modality='path'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.primary_modality = primary_modality
        
        self.volume_calculator = GramVolumeCalculator(normalize=True)
        
        if use_mlp:
            self.weight_generator = nn.Sequential(
                nn.Linear(feature_dim * 3 + 1, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),
                nn.Softmax(dim=-1)
            )
        else:
            self.weight_generator = None
        
        self.confidence_scorer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def compute_adaptive_weights(self, feat1, feat2, feat3, volume_info):
        """根据Volume计算自适应权重"""
        batch_size = feat1.size(0)
        volume = volume_info['volume']
        alignment_score = volume_info['alignment_score']
        
        if self.weight_generator is not None:
            combined = torch.cat([feat1, feat2, feat3, volume.unsqueeze(1)], dim=1)
            weights = self.weight_generator(combined)
        else:
            conf1 = self.confidence_scorer(feat1).squeeze(-1)
            conf2 = self.confidence_scorer(feat2).squeeze(-1)
            conf3 = self.confidence_scorer(feat3).squeeze(-1)
            
            base_weights = torch.stack([conf1, conf2, conf3], dim=1)
            base_weights = F.softmax(base_weights / self.temperature, dim=1)
            
            primary_boost = torch.zeros_like(base_weights)
            if self.primary_modality == 'path':
                primary_boost[:, 0] = 1.0 - alignment_score
            elif self.primary_modality == 'omic':
                primary_boost[:, 1] = 1.0 - alignment_score
            elif self.primary_modality == 'text':
                primary_boost[:, 2] = 1.0 - alignment_score
            
            alpha = 0.3
            weights = (1 - alpha) * base_weights + alpha * primary_boost
            weights = weights / weights.sum(dim=1, keepdim=True)
        
        return weights
    
    def forward(self, feat_path, feat_omic, feat_text):
        """执行Volume引导的融合"""
        volume_info = self.volume_calculator(feat_path, feat_omic, feat_text, return_pairwise=True)
        weights = self.compute_adaptive_weights(feat_path, feat_omic, feat_text, volume_info)
        
        fused = (
            weights[:, 0:1] * feat_path +
            weights[:, 1:2] * feat_omic +
            weights[:, 2:3] * feat_text
        )
        
        info = {
            'volume': volume_info['volume'],
            'alignment_score': volume_info['alignment_score'],
            'pairwise_volumes': volume_info['pairwise_volumes'],
            'fusion_weights': weights
        }
        
        return fused, info


class HierarchicalGRAMFusion(nn.Module):
    """分层GRAM融合"""
    
    def __init__(self, feature_dim, dropout=0.25):
        super().__init__()
        
        self.volume_calculator = GramVolumeCalculator(normalize=True)
        
        # 三条融合路径
        self.fuse_path_omic = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fuse_po_text = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fuse_path_text = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fuse_pt_omic = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fuse_omic_text = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fuse_ot_path = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, feat_path, feat_omic, feat_text):
        """分层融合"""
        volume_info = self.volume_calculator(feat_path, feat_omic, feat_text, return_pairwise=True)
        pairwise = volume_info['pairwise_volumes']
        
        vol_po = pairwise['path_omic']
        vol_pt = pairwise['path_text']
        vol_ot = pairwise['omic_text']
        
        # 三条路径
        h_po = self.fuse_path_omic(torch.cat([feat_path, feat_omic], dim=1))
        out1 = self.fuse_po_text(torch.cat([h_po, feat_text], dim=1))
        
        h_pt = self.fuse_path_text(torch.cat([feat_path, feat_text], dim=1))
        out2 = self.fuse_pt_omic(torch.cat([h_pt, feat_omic], dim=1))
        
        h_ot = self.fuse_omic_text(torch.cat([feat_omic, feat_text], dim=1))
        out3 = self.fuse_ot_path(torch.cat([h_ot, feat_path], dim=1))
        
        volume_scores = torch.stack([vol_po, vol_pt, vol_ot], dim=1)
        inv_volumes = 1.0 / (volume_scores + 1e-6)
        gate_weights = self.gate_net(inv_volumes)
        
        outputs = torch.stack([out1, out2, out3], dim=1)
        fused = torch.einsum('bd,bdn->bn', gate_weights, outputs)
        
        info = {
            'pairwise_volumes': pairwise,
            'pathway_weights': gate_weights,
            'selected_pathway': torch.argmin(volume_scores, dim=1)
        }
        
        return fused, info


class TaskAdaptiveFusion(nn.Module):
    """任务自适应融合"""
    
    def __init__(self, feature_dim, task_type='classification'):
        super().__init__()
        
        self.task_type = task_type
        
        self.volume_fusion = VolumeGuidedFusion(
            feature_dim,
            primary_modality='path' if task_type == 'classification' else None
        )
        
        if task_type == 'classification':
            self.volume_reg_weight = 0.1
        elif task_type == 'survival':
            self.volume_reg_weight = 0.05
        elif task_type == 'multi_label':
            self.volume_reg_weight = 0.08
        else:
            self.volume_reg_weight = 0.0
    
    def compute_volume_regularization(self, volume_info, labels=None):
        """计算Volume正则化损失"""
        volume = volume_info['volume']
        
        if self.task_type == 'classification' and labels is not None:
            reg_loss = 0.0
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() > 1:
                    class_volume = volume[mask].mean()
                    reg_loss += class_volume
            
            return reg_loss * self.volume_reg_weight
        
        elif self.task_type == 'survival':
            reg_loss = volume.mean()
            return reg_loss * self.volume_reg_weight
        
        else:
            return torch.tensor(0.0, device=volume.device)
    
    def forward(self, feat_path, feat_omic, feat_text, labels=None):
        """任务自适应融合"""
        fused, volume_info = self.volume_fusion(feat_path, feat_omic, feat_text)
        reg_loss = self.compute_volume_regularization(volume_info, labels)
        
        return fused, volume_info, reg_loss


# ========== 多标签专用模块 (来自代码2) ==========

class GramGatedConcatFusion(nn.Module):
    """
    多标签专用：Gram引导的通道门控拼接
    """
    def __init__(self, feature_dim, reduction=16):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.gram_calc = GramVolumeCalculator(normalize=True)
        
        self.gram_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim // 2),
            nn.ReLU()
        )
        
        self.global_pool = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU()
        )
        
        self.gate_generator = nn.Sequential(
            nn.Linear(feature_dim // 2 + feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim * 3),
            nn.Sigmoid()
        )
        
    def forward(self, feat_path, feat_omic, feat_text):
        batch_size = feat_path.size(0)
        
        # 使用 return_pairwise=False 的简化版本
        feat1_norm = F.normalize(feat_path, dim=1)
        feat2_norm = F.normalize(feat_omic, dim=1)
        feat3_norm = F.normalize(feat_text, dim=1)
        
        A = torch.stack([feat1_norm, feat2_norm, feat3_norm], dim=-1)
        G = torch.bmm(A.transpose(1, 2), A)
        gram_flat = G.view(batch_size, -1)
        
        gram_emb = self.gram_encoder(gram_flat)
        
        raw_concat = torch.cat([feat_path, feat_omic, feat_text], dim=1)
        global_ctx = self.global_pool(raw_concat)
        
        combined_info = torch.cat([gram_emb, global_ctx], dim=1)
        all_gates = self.gate_generator(combined_info)
        
        gate_path = all_gates[:, :self.feature_dim]
        gate_omic = all_gates[:, self.feature_dim:2*self.feature_dim]
        gate_text = all_gates[:, 2*self.feature_dim:]
        
        f_path_gated = feat_path * gate_path
        f_omic_gated = feat_omic * gate_omic
        f_text_gated = feat_text * gate_text
        
        fused = torch.cat([f_path_gated, f_omic_gated, f_text_gated], dim=1)
        
        info = {
            'gram_matrix': G,
            'gates_mean': torch.stack([gate_path.mean(), gate_omic.mean(), gate_text.mean()])
        }
        
        return fused, info


class AsymmetricLoss(nn.Module):
    """多标签专用：Asymmetric Loss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.size(0)


# ========== 原有辅助模块 (来自代码1) ==========

class Contra_head(nn.Module):
    """对比学习投影头"""
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.contra_head = nn.Sequential(
            nn.Linear(input_dim, contra_dim),
            LayerNorm(contra_dim)
        )
    
    def forward(self, x):
        return self.contra_head(x)


class Attn_Net_Gated(nn.Module):
    """门控注意力网络"""
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class SNN_Block(nn.Module):
    """自归一化神经网络块"""
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False)
        )
    
    def forward(self, x):
        return self.fc(x)


class MLP_Block(nn.Module):
    """标准MLP块"""
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False)
        )
    
    def forward(self, x):
        return self.fc(x)


class BatchedAttn_Net_Gated(nn.Module):
    """批量门控注意力网络"""
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_net = Attn_Net_Gated(L, D, dropout, n_classes)
    
    def forward(self, x_batch):
        batch_A = []
        batch_h = []
        batch_attention_features = []
        
        for x in x_batch:
            A, h = self.attention_net(x)
            
            if A.shape[1] == 1:
                A = A.squeeze(1)
            elif A.shape[0] == 1:
                A = A.squeeze(0)
            else:
                A = A[:, 0]
            
            A = F.softmax(A, dim=0)
            attention_feat = torch.einsum('n,nd->d', A, h)
            batch_attention_features.append(attention_feat)
            
            batch_A.append(A)
            batch_h.append(h)
        
        batched_attention = torch.stack(batch_attention_features, dim=0)
        return batch_A, batch_h, batched_attention


class CPathOmniSequentialFusion(nn.Module):
    """序列级Transformer融合"""
    
    def __init__(self, path_dim=512, omic_dim=256, text_dim=256,
                 hidden_dim=256, nhead=8, num_layers=2, dropout=0.25):
        super().__init__()
        
        self.path_proj = nn.Linear(path_dim, hidden_dim)
        self.omic_proj = nn.Linear(omic_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.modality_embedding = nn.Embedding(3, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, h_path, h_omic, h_text):
        batch_size = h_path.size(0)
        device = h_path.device
        
        path_token = self.path_proj(h_path).unsqueeze(1)
        omic_token = self.omic_proj(h_omic).unsqueeze(1)
        text_token = self.text_proj(h_text).unsqueeze(1)
        
        modality_ids = torch.tensor([0, 1, 2], device=device).unsqueeze(0).expand(batch_size, -1)
        modality_emb = self.modality_embedding(modality_ids)
        
        sequence = torch.cat([path_token, omic_token, text_token], dim=1)
        sequence = sequence + modality_emb
        
        transformed = self.transformer(sequence)
        output = transformed.mean(dim=1)
        
        return self.output_proj(output)


class TimeEmbedding(nn.Module):
    """时间嵌入层"""
    def __init__(self, hidden_dim=64, num_freqs=32):
        super().__init__()
        self.num_freqs = num_freqs
        self.hidden_dim = hidden_dim
        
        self.freq_weights = nn.Parameter(torch.randn(num_freqs) * 0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, time):
        time_norm = time / (time.max() + 1e-8)
        time_norm = time_norm.unsqueeze(-1)
        
        freqs = torch.exp(self.freq_weights)
        angles = time_norm * freqs.unsqueeze(0)
        
        sin_emb = torch.sin(2 * np.pi * angles)
        cos_emb = torch.cos(2 * np.pi * angles)
        
        time_feat = torch.cat([sin_emb, cos_emb], dim=-1)
        time_emb = self.mlp(time_feat)
        
        return time_emb


class RiskStratification(nn.Module):
    """风险分层模块"""
    def __init__(self, input_dim, num_strata=3):
        super().__init__()
        self.num_strata = num_strata
        
        self.risk_scorer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, features):
        risk_scores = self.risk_scorer(features).squeeze(-1)
        strata_logits = self._compute_strata_logits(risk_scores)
        return risk_scores, strata_logits
    
    def _compute_strata_logits(self, risk_scores):
        quantiles = torch.quantile(
            risk_scores,
            torch.tensor([0.33, 0.67], device=risk_scores.device)
        )
        
        low_prob = torch.sigmoid(-(risk_scores - quantiles[0]) * 10)
        high_prob = torch.sigmoid((risk_scores - quantiles[1]) * 10)
        mid_prob = 1 - low_prob - high_prob
        
        strata_logits = torch.stack([low_prob, mid_prob, high_prob], dim=-1)
        return strata_logits


def volume_computation3(feat1, feat2, feat3):
    """计算三个特征向量的volume距离"""
    batch_size = feat1.shape[0]
    
    volumes = []
    for i in range(batch_size):
        for j in range(batch_size):
            v1 = torch.stack([feat1[i], feat2[i], feat3[i]], dim=0)
            v2 = torch.stack([feat1[j], feat2[j], feat3[j]], dim=0)
            
            diff = v1 - v2
            vol = torch.abs(torch.det(diff @ diff.T))
            volumes.append(vol)
    
    volume_matrix = torch.stack(volumes).reshape(batch_size, batch_size)
    return volume_matrix


# ========== 主模型 ==========

class GRAMPorpoiseMMF(nn.Module):
    """GRAMPorpoiseMMF - 修复版本"""
    
    def __init__(self, 
             omic_input_dim,
             text_input_dim=768, 
             path_input_dim=1024,
             fusion='concat',
             dropout=0.25,
             n_classes=4,
             n_labels=5,
             contra_dim=256,
             contra_temp=0.07,
             use_contrastive=False,
             contrastive_weight=0.1,
             task_type='classification',
             size_arg="small",
             dropinput=0.10,
             use_mlp=False,
             distance_type='volume',
             path_hidden_dim=512,
             path_attention_dim=256,
             encoder_hidden_dim=768,
             survival_time_threshold=12.0,
             use_simple_survival_loss=True,
             survival_loss_debug=False,
             use_time_embedding=True,
             use_risk_stratification=True,
             use_hard_negative_mining=True,
             use_adaptive_temperature=True,
             time_scales=[6.0, 12.0, 24.0],
             use_gram_fusion=False,
             ## 新增的参数
             enable_explainability=False,    # 是否启用可解释性
             n_qa_pairs=6,                   # QA对数量
             n_pathways=50,                  # 通路数量
             pathway_gene_mapping=None,      # 通路-基因映射矩阵 
             **kwargs):
    
        super().__init__()

        # 🔥🔥🔥 关键：在调用_build_encoders之前设置这些属性
        self.enable_explainability = enable_explainability
        self.n_qa_pairs = n_qa_pairs
        self.n_pathways = n_pathways
        
        # 🔥 关键：先保存输入维度（在调用 _build_fusion_layer 之前）
        self.path_input_dim = path_input_dim
        self.omic_input_dim = omic_input_dim
        self.text_input_dim = text_input_dim
        
        # 保存其他配置
        self.fusion = fusion
        self.task_type = task_type
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.contra_dim = contra_dim
        self.contra_temp = contra_temp
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.distance_type = distance_type
        self.path_hidden_dim = path_hidden_dim
        self.path_attention_dim = path_attention_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.survival_time_threshold = survival_time_threshold
        self.use_simple_survival_loss = use_simple_survival_loss
        self.survival_loss_debug = survival_loss_debug
        
        self.use_time_embedding = use_time_embedding and (task_type == 'survival')
        self.use_risk_stratification = use_risk_stratification and (task_type == 'survival')
        self.use_hard_negative_mining = use_hard_negative_mining
        self.use_adaptive_temperature = use_adaptive_temperature
        self.time_scales = time_scales
        
        self.use_gram_fusion = use_gram_fusion
        
        self._forward_count = 0
        self._loss_call_count = 0
        
        print(f"\n{'='*70}")
        print(f"Initializing GRAMPorpoiseMMF (Fixed Version):")
        print(f"   Task type: {task_type}")
        print(f"   Fusion strategy: {fusion}")
        print(f"   Path input dim: {path_input_dim}")  # 🔥 添加维度打印
        print(f"   Omic input dim: {omic_input_dim}")  # 🔥 添加维度打印
        print(f"   Text input dim: {text_input_dim}")  # 🔥 添加维度打印
        if use_gram_fusion:
            print(f"   🔥 GRAM-Fusion: ENABLED")
        else:
            print(f"   GRAM-Fusion: DISABLED")
        print(f"   Use contrastive: {use_contrastive}")
        print(f"{'='*70}\n")
        
        # 构建编码器
        self._build_encoders(path_input_dim, omic_input_dim, text_input_dim, 
                            dropout, size_arg, dropinput, use_mlp)
        
        # 🔥 现在可以安全调用 _build_fusion_layer 了
        self._build_fusion_layer(dropout)

        
        # 其他模块
        if self.use_time_embedding:
            self.time_embedding = TimeEmbedding(hidden_dim=64, num_freqs=32)
            self.time_fusion = nn.Sequential(
                nn.Linear(self.fusion_output_dim + 64, self.fusion_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        if self.use_risk_stratification:
            self.risk_stratification = RiskStratification(
                input_dim=self.fusion_output_dim, num_strata=3
            )
        
        if self.use_adaptive_temperature:
            self.temp_predictor = nn.Sequential(
                nn.Linear(self.fusion_output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        if self.use_contrastive:
            self.contra_head_path = Contra_head(encoder_hidden_dim, contra_dim)
            self.contra_head_omic = Contra_head(encoder_hidden_dim, contra_dim)
            self.contra_head_text = Contra_head(encoder_hidden_dim, contra_dim)
        
        self._build_classifier(dropout)
 
        # 多标签专用损失函数
        if self.task_type == 'multi_label':
            self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
        
        if enable_explainability:
            print(f"\n{'='*70}")
            print(f"🔍 Initializing Explainability Module:")
            print(f"   n_qa_pairs: {n_qa_pairs}")
            print(f"   n_pathways: {n_pathways}")
            print(f"   encoder_hidden_dim: {self.encoder_hidden_dim}")
            
            # 🔥🔥🔥 关键修复：使用固定的可解释性特征维度
            # 不依赖 encoder_hidden_dim，避免维度不匹配
            EXPLAIN_FEATURE_DIM = 256  # 固定为256
            
            print(f"   🔧 Using fixed explainability feature dim: {EXPLAIN_FEATURE_DIM}")
            print(f"{'='*70}\n")
            
            # QA级别的文本编码器
            self.text_qa_encoder = QALevelTextEncoder(
                text_dim=self.text_input_dim,      # 输入: 768
                n_qa_pairs=n_qa_pairs,             # 6
                output_dim=EXPLAIN_FEATURE_DIM     # 输出: 256 ✅
            )
            
            # 通路级别的Omic编码器
            self.omic_pathway_encoder = PathwayLevelOmicEncoder(
                gene_dim=self.omic_input_dim,      # 输入: 2007
                n_pathways=n_pathways,             # 50
                pathway_dim=EXPLAIN_FEATURE_DIM,   # 输出: 256 ✅
                pathway_gene_mapping=pathway_gene_mapping
            )
            
            # Patch投影器
            self.patch_projector_explain = nn.Linear(
                self.path_hidden_dim,              # 输入: 512
                EXPLAIN_FEATURE_DIM                # 输出: 256 ✅
            )
            
            # 可解释性模块
            self.explainability_module = TrimodalExplainabilityModule(
                feature_dim=EXPLAIN_FEATURE_DIM,   # 统一: 256 ✅
                n_qa_pairs=n_qa_pairs,
                n_pathways=n_pathways
            )
            
            print(f"✅ Explainability modules initialized:")
            print(f"   Text QA Encoder: [B, 6, 768] -> [B, 6, {EXPLAIN_FEATURE_DIM}]")
            print(f"   Omic Pathway Encoder: [B, {self.omic_input_dim}] -> [B, 50, {EXPLAIN_FEATURE_DIM}]")
            print(f"   Patch Projector: [B, N, {self.path_hidden_dim}] -> [B, N, {EXPLAIN_FEATURE_DIM}]")

            # 🔥 检查权重是否被初始化
            print(f"\n🔍 Checking explainability module weights:")
            print(f"   text_qa_encoder.qa_projector.weight: mean={self.text_qa_encoder.qa_projector.weight.mean():.6f}")
            print(f"   omic_pathway_encoder.gene_to_pathway.weight: mean={self.omic_pathway_encoder.gene_to_pathway.weight.mean():.6f}")
            print(f"   patch_projector_explain.weight: mean={self.patch_projector_explain.weight.mean():.6f}")

    def _build_encoders(self, path_input_dim, omic_input_dim, text_input_dim, 
                   dropout, size_arg, dropinput, use_mlp):
        """构建编码器 - 只构建维度 > 0 的模态编码器"""
        
        # 🔥 Path 编码器（只在 path_input_dim > 0 时构建）
        if path_input_dim > 0:
            size_dict_path = {
                "small": [path_input_dim, self.path_hidden_dim, self.encoder_hidden_dim], 
                "big": [1024, self.path_hidden_dim, self.encoder_hidden_dim]
            }
            size = size_dict_path[size_arg]
            
            self.path_fc = nn.Sequential(
                nn.Dropout(dropinput), 
                nn.Linear(size[0], size[1]),
                nn.ReLU(), 
                nn.Dropout(dropout)
            )
            
            self.batched_attention_net = BatchedAttn_Net_Gated(
                L=self.path_hidden_dim,
                D=self.path_attention_dim,
                dropout=dropout, 
                n_classes=1
            )
            
            self.rho = nn.Sequential(
                nn.Linear(self.path_hidden_dim, self.encoder_hidden_dim),
                nn.ReLU(), 
                nn.Dropout(dropout)
            )
            
            self.path_output_dim = size[2]
            print(f"   ✅ Path encoder built (dim: {path_input_dim} -> {self.encoder_hidden_dim})")
        else:
            self.path_fc = None
            self.batched_attention_net = None
            self.rho = None
            self.path_output_dim = 0
            print(f"   ⏭️  Path encoder skipped (dim: 0)")
        
        # 🔥 Omic 编码器（只在 omic_input_dim > 0 时构建）
        if omic_input_dim > 0:
            Block = MLP_Block if use_mlp else SNN_Block
            hidden = [self.encoder_hidden_dim, self.encoder_hidden_dim]
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
            self.omic_output_dim = self.encoder_hidden_dim
            print(f"   ✅ Omic encoder built (dim: {omic_input_dim} -> {self.encoder_hidden_dim})")
        else:
            self.fc_omic = None
            self.omic_output_dim = 0
            print(f"   ⏭️  Omic encoder skipped (dim: 0)")
        
        # 🔥 Text 编码器（只在 text_input_dim > 0 时构建）
        # 🔥🔥🔥 Text 编码器（支持QA级别）
        if text_input_dim > 0:
            # 🔥 检测是否启用QA级别的文本
            self.enable_qa_text = getattr(self, 'enable_explainability', False) and getattr(self, 'n_qa_pairs', 1) > 1
            
            if self.enable_qa_text:
                # 🔥 QA级别编码器：[batch, 6, 768] -> [batch, hidden_dim]
                n_qa = getattr(self, 'n_qa_pairs', 6)
                
                # 方案A：用Attention聚合6个QA
                self.qa_attention = nn.MultiheadAttention(
                    embed_dim=text_input_dim,
                    num_heads=4,
                    batch_first=True
                )
                # 投影到目标维度
                self.fc_text = nn.Sequential(
                    nn.Linear(text_input_dim, self.encoder_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                print(f"   ✅ Text encoder built for QA-level:")
                print(f"      Input: [batch, {n_qa}, {text_input_dim}]")
                print(f"      Output: [batch, {self.encoder_hidden_dim}]")
                print(f"      Method: Multi-head Attention aggregation")
                
            else:
                # 🔥 常规编码器：[batch, 768] -> [batch, hidden_dim]
                self.qa_attention = None
                self.fc_text = nn.Sequential(
                    nn.Linear(text_input_dim, self.encoder_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                print(f"   ✅ Text encoder built (dim: {text_input_dim} -> {self.encoder_hidden_dim})")
            
            self.text_output_dim = self.encoder_hidden_dim
            
        else:
            self.fc_text = None
            self.qa_attention = None
            self.text_output_dim = 0
            print(f"   ⏭️  Text encoder skipped (dim: 0)")

    def _build_fusion_layer(self, dropout):
        """构建融合层 - 支持单/双/三模态"""
        
        # 🔥 新增：检测实际使用的模态数量
        self.num_modalities = 0
        if self.path_input_dim > 0:
            self.num_modalities += 1
        if self.omic_input_dim > 0:
            self.num_modalities += 1
        if self.text_input_dim > 0:
            self.num_modalities += 1
        
        print(f"   Detected {self.num_modalities} modalities")
        
        # 🔥 单模态：不需要融合层
        if self.num_modalities == 1:
            print("   Single modality mode - no fusion needed")
            self._is_gram_fusion = False
            self._is_multilabel_gram = False
            self.mm = None  # 不需要融合
            self.fusion_output_dim = self.encoder_hidden_dim
            return
        
        # 多标签任务使用 GramGatedConcatFusion
        if self.task_type == 'multi_label' and self.use_gram_fusion:
            print("   Using GramGatedConcatFusion (Multi-label optimized)")
            self.mm = GramGatedConcatFusion(self.encoder_hidden_dim)
            self._is_gram_fusion = True
            self._is_multilabel_gram = True
            self.fusion_output_dim = self.encoder_hidden_dim * 3
        
        # 🔥 新增：生存任务专用融合
        elif self.task_type == 'survival' and self.use_gram_fusion:
            print("   Using SurvivalTaskAdaptiveFusion (GRAM for Survival)")
            self.mm = SurvivalTaskAdaptiveFusion(
                feature_dim=self.encoder_hidden_dim,
                time_emb_dim=64  # 与TimeEmbedding输出维度一致
            )
            self._is_gram_fusion = True
            self._is_multilabel_gram = False
            self._is_survival_gram = True  # 新增标志
            self.fusion_output_dim = self.encoder_hidden_dim
            
        elif self.fusion == 'concat':
            if self.use_gram_fusion:
                print("   Using TaskAdaptiveFusion (GRAM)")
                self.mm = TaskAdaptiveFusion(
                    feature_dim=self.encoder_hidden_dim,
                    task_type=self.task_type
                )
                self._is_gram_fusion = True
            else:
                # 🔥 动态调整concat维度
                self.mm = nn.Sequential(
                    nn.Linear(self.encoder_hidden_dim * self.num_modalities, self.encoder_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self._is_gram_fusion = False
            
            self._is_multilabel_gram = False
            self.fusion_output_dim = self.encoder_hidden_dim
            
        elif self.fusion == 'cpathomni_sequential':
            if self.use_gram_fusion:
                print("   Using HierarchicalGRAMFusion")
                self.mm = HierarchicalGRAMFusion(
                    feature_dim=self.encoder_hidden_dim,
                    dropout=dropout
                )
                self._is_gram_fusion = True
            else:
                self.mm = CPathOmniSequentialFusion(
                    path_dim=self.path_hidden_dim,
                    omic_dim=self.encoder_hidden_dim,
                    text_dim=self.encoder_hidden_dim,
                    hidden_dim=self.encoder_hidden_dim,
                    num_layers=2,
                    dropout=dropout
                )
                self._is_gram_fusion = False
            
            self._is_multilabel_gram = False
            self.fusion_output_dim = self.encoder_hidden_dim
            
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
            
    def _build_classifier(self, dropout):
        """构建分类器 - 恢复代码1的结构"""
        classifier_input_dim = self.fusion_output_dim
        
        if self.task_type == 'survival':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.n_classes)
            )
        elif self.task_type == 'classification':
            self.classifier = nn.Linear(classifier_input_dim, self.n_classes)
        elif self.task_type == 'multi_label':
            # 多标签使用更深的分类器
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.n_labels)
            )

    def get_contrastive_features(self, h_path, h_omic, h_text):
        """获取对比学习特征"""
        if not self.use_contrastive:
            raise RuntimeError("Contrastive learning not enabled")
        
        feat_path = self.contra_head_path(h_path)
        feat_omic = self.contra_head_omic(h_omic)
        feat_text = self.contra_head_text(h_text)
        
        feat_path = F.normalize(feat_path, dim=-1)
        feat_omic = F.normalize(feat_omic, dim=-1)
        feat_text = F.normalize(feat_text, dim=-1)
        
        return feat_path, feat_omic, feat_text

    def compute_volume_distance(self, feat_path, feat_omic, feat_text):
        """计算volume距离"""
        volume = volume_computation3(feat_path, feat_omic, feat_text)
        volume = torch.clamp(volume, min=-50, max=50)
        return volume
    
    def compute_cosine_distance(self, feat_path, feat_omic, feat_text):
        """计算余弦距离"""
        sim_po = feat_path @ feat_omic.T
        sim_pt = feat_path @ feat_text.T
        sim_ot = feat_omic @ feat_text.T
        dist = (1-sim_po + 1-sim_pt + 1-sim_ot) / 3.0
        return -dist
    
    def compute_gram_loss(self, feat_path, feat_omic, feat_text):
        """原始GRAM损失"""
        batch_size = feat_path.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=feat_path.device)
        
        if self.distance_type == 'volume':
            similarity_logits = self.compute_volume_distance(feat_path, feat_omic, feat_text)
        elif self.distance_type == 'cosine':
            similarity_logits = self.compute_cosine_distance(feat_path, feat_omic, feat_text)
        elif self.distance_type == 'hybrid':
            vol = self.compute_volume_distance(feat_path, feat_omic, feat_text)
            cos = self.compute_cosine_distance(feat_path, feat_omic, feat_text)
            similarity_logits = (vol + cos) / 2.0
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")
        
        logits = similarity_logits / self.contra_temp
        
        targets = torch.arange(batch_size, device=logits.device)
        loss = (
            F.cross_entropy(logits, targets, label_smoothing=0.1) +
            F.cross_entropy(logits.T, targets, label_smoothing=0.1)
        ) / 2.0
        
        return loss
    
    def forward(self, compute_loss=True, return_features=False,
            survival_time=None, event=None,
            # 🔥 新增参数
            return_explanation=False,      # 是否返回可解释性结果
            qa_embeddings=None,            # (可选) 6个QA的独立编码 [B, 6, 768]
            pathway_scores=None,           # (可选) 预计算的通路得分 [B, n_pathways]
            qa_texts=None,                 # (可选) 6个QA的文本，用于可视化
            pathway_names=None,            # (可选) 通路名称，用于可视化
            **kwargs):
        """前向传播 - 支持单/双/三模态"""
        device = next(self.parameters()).device
        self._forward_count += 1
        
        # 🔥 提取各模态特征（可能为None）
        h_path_enc = None
        h_omic_enc = None
        h_text_enc = None
        
        # 🔥 Path 模态（只在编码器存在且输入存在时处理）
        if self.path_fc is not None and 'x_path' in kwargs and kwargs['x_path'] is not None:
            x_path_batch = kwargs['x_path']
            
            if not isinstance(x_path_batch, list):
                if x_path_batch.dim() == 3:
                    x_path_batch = [x_path_batch[i] for i in range(x_path_batch.size(0))]
                else:
                    x_path_batch = [x_path_batch]
            
            processed_path_batch = [self.path_fc(x) for x in x_path_batch]
            batch_A, batch_h, h_path_attention = self.batched_attention_net(processed_path_batch)
            h_path_enc = self.rho(h_path_attention)

            # 🔥🔥🔥 在这之后添加：
            # 保存attention权重供可解释性使用
            self._last_batch_attention = batch_A  # 列表，每个元素是一个样本的attention [n_patches]
            print(f"  💾 Saved attention weights: {len(batch_A)} samples, first sample has {len(batch_A[0]) if batch_A else 0} patches")
        
        # 🔥 Omic 模态（只在编码器存在且输入存在时处理）
        if self.fc_omic is not None and 'x_omic' in kwargs and kwargs['x_omic'] is not None:
            x_omic = kwargs['x_omic']
            h_omic_enc = self.fc_omic(x_omic)
        
        # 🔥 Text 模态（只在编码器存在且输入存在时处理）
        # 🔥🔥🔥 Text 模态处理（完整修复版）
        if self.fc_text is not None and 'x_text' in kwargs and kwargs.get('x_text') is not None:
            x_text = kwargs['x_text']
            
            # Step 1: 统一维度
            if x_text.dim() == 1:
                x_text = x_text.unsqueeze(0)  # [768] -> [1, 768]
            
            # Step 2: 处理3D输入（QA级别）
            if x_text.dim() == 3:  # [batch, 6, 768]
                print(f"  🔍 Text input is 3D: {x_text.shape}")
                
                # 保存原始3D数据（用于可解释性）
                self._temp_text_3d = x_text.to(device).float()
                
                # 聚合为2D
                if self.enable_qa_text and self.qa_attention is not None:
                    # 方案A: 用Attention聚合
                    query = x_text.mean(dim=1, keepdim=True)  # [batch, 1, 768]
                    aggregated, _ = self.qa_attention(
                        query=query, 
                        key=x_text, 
                        value=x_text
                    )
                    x_text_2d = aggregated.squeeze(1)  # [batch, 768]
                    print(f"     Aggregated via attention: {x_text_2d.shape}")
                else:
                    # 方案B: 直接平均
                    x_text_2d = x_text.mean(dim=1)  # [batch, 768]
                    print(f"     Aggregated via mean: {x_text_2d.shape}")
                
                # 编码
                h_text_enc = self.fc_text(x_text_2d)  # [batch, hidden_dim]
                
            else:  # 2D [batch, 768]
                self._temp_text_3d = None
                h_text_enc = self.fc_text(x_text.to(device).float())
            
            # Step 3: 🔥🔥🔥 最终安全检查（关键！）
            if h_text_enc.dim() == 3:
                print(f"  ⚠️ h_text_enc is still 3D: {h_text_enc.shape}, squeezing...")
                h_text_enc = h_text_enc.squeeze(1)
            
            # Step 4: 验证最终形状
            assert h_text_enc.dim() == 2, f"h_text_enc must be 2D, got {h_text_enc.shape}"
            print(f"  ✅ Final h_text_enc: {h_text_enc.shape}")
        
        # 🔥 统计实际存在的模态
        available_features = []
        available_names = []
        
        if h_path_enc is not None:
            available_features.append(h_path_enc)
            available_names.append('path')
        if h_omic_enc is not None:
            available_features.append(h_omic_enc)
            available_names.append('omic')
        if h_text_enc is not None:
            available_features.append(h_text_enc)
            available_names.append('text')
        
        num_available = len(available_features)
        
        if num_available == 0:
            raise ValueError("At least one modality must be provided!")
        
        if self._forward_count == 1:
            print(f"\n🔍 Forward pass info:")
            print(f"   Available modalities: {available_names}")
            print(f"   Number of modalities: {num_available}")
        
        # 🔥🔥🔥 【关键修改】生存任务：在融合之前计算时间嵌入
        time_emb = None
        if self.task_type == 'survival' and self.use_time_embedding and survival_time is not None:
            time_emb = self.time_embedding(survival_time)
        
        # 🔥 融合逻辑
        fusion_info = {}
        
        if num_available == 1:
            # 单模态：直接使用
            h_mm = available_features[0]
            
        elif num_available == 2:
            # 双模态融合
            if self.mm is None:
                # 如果没有融合模块，直接拼接
                h_mm = torch.cat(available_features, dim=1)
                # 需要一个投影层降维
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(
                        self.encoder_hidden_dim * 2, 
                        self.encoder_hidden_dim
                    ).to(available_features[0].device)
                h_mm = self.projection(h_mm)
            else:
                # 使用融合模块
                if self._is_gram_fusion:
                    # GRAM融合需要3个模态，用零向量填充
                    dummy_feat = torch.zeros_like(available_features[0])
                    feat1, feat2 = available_features[0], available_features[1]
                    
                    labels = kwargs.get('labels', None)
                    
                    # 🔥🔥🔥 生存任务的双模态融合
                    if self.task_type == 'survival' and hasattr(self, '_is_survival_gram') and self._is_survival_gram:
                        # 计算临时风险分数
                        temp_concat = torch.cat([feat1, feat2], dim=1)
                        if not hasattr(self, 'temp_proj_dual'):
                            self.temp_proj_dual = nn.Linear(
                                self.encoder_hidden_dim * 2,
                                self.encoder_hidden_dim
                            ).to(feat1.device)
                        temp_feat = self.temp_proj_dual(temp_concat)
                        
                        if self.use_risk_stratification:
                            temp_risk_score, _ = self.risk_stratification(temp_feat)
                        else:
                            temp_risk_score = None
                        
                        # 使用生存专用融合（传入time_emb）
                        h_mm, vol_info, reg_loss = self.mm(
                            feat1, feat2, dummy_feat,
                            time_emb=time_emb,
                            event=event,
                            survival_time=survival_time,
                            risk_score=temp_risk_score
                        )
                        fusion_info['volume_reg_loss'] = reg_loss
                        fusion_info.update(vol_info)
                        
                    elif isinstance(self.mm, TaskAdaptiveFusion):
                        h_mm, vol_info, reg_loss = self.mm(feat1, feat2, dummy_feat, labels)
                        fusion_info['volume_reg_loss'] = reg_loss
                    else:
                        h_mm, vol_info = self.mm(feat1, feat2, dummy_feat)
                        fusion_info.update(vol_info)
                else:
                    # 普通concat融合
                    concatenated = torch.cat(available_features, dim=1)
                    h_mm = self.mm(concatenated)
                    
        else:  # num_available == 3
            # 三模态融合
            if hasattr(self, '_is_multilabel_gram') and self._is_multilabel_gram:
                # 多标签专用融合（不动）
                h_mm, info = self.mm(h_path_enc, h_omic_enc, h_text_enc)
                fusion_info.update(info)
                
            elif hasattr(self, '_is_survival_gram') and self._is_survival_gram:
                # 生存任务专用融合
                labels = kwargs.get('labels', None)
                
                # 🔥 不再需要展平，直接拼接
                temp_concat = torch.cat([h_path_enc, h_omic_enc, h_text_enc], dim=1)
                
                if not hasattr(self, 'temp_proj_tri'):
                    total_dim = temp_concat.size(1)
                    self.temp_proj_tri = nn.Linear(
                        total_dim, 
                        self.encoder_hidden_dim
                    ).to(temp_concat.device)
                temp_feat = self.temp_proj_tri(temp_concat)
                
                if self.use_risk_stratification:
                    temp_risk_score, _ = self.risk_stratification(temp_feat)
                else:
                    temp_risk_score = None
                
                # 传入融合模块（都是2D）
                h_mm, vol_info, reg_loss = self.mm(
                    h_path_enc, h_omic_enc, h_text_enc,
                    time_emb=time_emb,
                    event=event,
                    survival_time=survival_time,
                    risk_score=temp_risk_score
                )
                
                fusion_info['volume_reg_loss'] = reg_loss
                fusion_info['volume'] = vol_info['volume']
                fusion_info['base_volume'] = vol_info.get('base_volume', vol_info['volume'])
                fusion_info['alignment_score'] = vol_info['alignment_score']
                fusion_info['fusion_weights'] = vol_info['fusion_weights']
                
            elif self._is_gram_fusion:
                # 其他GRAM融合（分类任务等）
                labels = kwargs.get('labels', None)
                
                if isinstance(self.mm, TaskAdaptiveFusion):
                    h_mm, vol_info, reg_loss = self.mm(h_path_enc, h_omic_enc, h_text_enc, labels)
                    fusion_info['volume_reg_loss'] = reg_loss
                    fusion_info['volume'] = vol_info['volume']
                    fusion_info['alignment_score'] = vol_info['alignment_score']
                    fusion_info['fusion_weights'] = vol_info['fusion_weights']
                    
                elif isinstance(self.mm, HierarchicalGRAMFusion):
                    h_mm, vol_info = self.mm(h_path_enc, h_omic_enc, h_text_enc)
                    fusion_info['pathway_weights'] = vol_info['pathway_weights']
                    fusion_info['selected_pathway'] = vol_info['selected_pathway']
                
                else:
                    h_mm, vol_info = self.mm(h_path_enc, h_omic_enc, h_text_enc)
                    fusion_info['volume'] = vol_info['volume']
                    fusion_info['fusion_weights'] = vol_info['fusion_weights']
            
            else:
                # 非GRAM融合
                if self.fusion == 'concat':
                    concatenated = torch.cat([h_path_enc, h_omic_enc, h_text_enc], dim=1)
                    h_mm = self.mm(concatenated)
                elif self.fusion == 'cpathomni_sequential':
                    h_mm = self.mm(h_path_attention, h_omic_enc, h_text_enc)
                else:
                    raise ValueError(f"Unknown fusion: {self.fusion}")
        
        # 🔥🔥🔥 【关键修改】非生存任务：在融合之后处理时间嵌入（保持原有逻辑）
        # 生存任务的时间信息已经在融合中处理，这里跳过
        if self.use_time_embedding and survival_time is not None and self.task_type != 'survival':
            time_emb_post = self.time_embedding(survival_time)
            h_mm_with_time = torch.cat([h_mm, time_emb_post], dim=-1)
            h_mm = self.time_fusion(h_mm_with_time)
        
        # 分类
        logits = self.classifier(h_mm)
        
        # 返回结果
        result = {'logits': logits}
        
        if fusion_info:
            result['fusion_info'] = fusion_info
        
        if return_features:
            result['fused_features'] = h_mm
            result['modal_features'] = {
                'path': h_path_enc,
                'omic': h_omic_enc,
                'text': h_text_enc
            }
        
        # 对比学习损失（只在三模态时启用）
        if compute_loss and self.use_contrastive and num_available == 3:
            feat_path, feat_omic, feat_text = self.get_contrastive_features(
                h_path_enc, h_omic_enc, h_text_enc
            )
            
            contrastive_loss = self.compute_gram_loss(feat_path, feat_omic, feat_text)
            result['contrastive_loss'] = contrastive_loss
            
            if return_features:
                result['contrastive_features'] = {
                    'path': feat_path,
                    'omic': feat_omic,
                    'text': feat_text
                }



        # ===== 🔥 可解释性计算（新增）=====
        # 🔥🔥🔥 在return result之前，添加可解释性计算
        # 🔥 调试信息
        if return_explanation:
            print(f"  🔍 return_explanation=True, enable_explainability={self.enable_explainability}")

        # ===== 🔥🔥🔥 可解释性计算（完整修复版）=====
        if return_explanation and self.enable_explainability:
            try:
                print(f"  🔍 return_explanation=True, enable_explainability=True")
                print(f"  📊 Computing explainability...")
                
                qa_features = None
                pathway_features = None
                patch_features_explain = None
                
                # === 1. QA级别的文本特征 ===
                if h_text_enc is not None and hasattr(self, '_temp_text_3d') and self._temp_text_3d is not None:
                    try:
                        # 🔥 传入原始3D数据
                        # 🔥🔥🔥 检查输入特征
                        print(f"     🔍 Input QA embeddings statistics:")
                        print(f"        _temp_text_3d: min={self._temp_text_3d.min():.4f}, max={self._temp_text_3d.max():.4f}, mean={self._temp_text_3d.mean():.4f}, std={self._temp_text_3d.std():.4f}")
                        
                        qa_features = self.text_qa_encoder(
                            h_text_enc,
                            qa_embeddings=self._temp_text_3d
                        )
                        
                        # 🔥 检查输出特征
                        print(f"     🔍 Output QA features statistics:")
                        print(f"        qa_features: min={qa_features.min():.4f}, max={qa_features.max():.4f}, mean={qa_features.mean():.4f}, std={qa_features.std():.4f}")
                        print(f"     ✅ QA features: {qa_features.shape}")  # 应该是 [B, 6, 256]
                        
                        # 🔥 维度检查
                        if qa_features.shape[-1] != 256:
                            print(f"     ⚠️ WARNING: QA feature dim is {qa_features.shape[-1]}, expected 256!")
                            
                    except Exception as e:
                        print(f"     ❌ QA encoding failed: {e}")
                        qa_features = None
                else:
                    print(f"     ⚠️ No text features for QA encoding")
                    if h_text_enc is None:
                        print(f"        - h_text_enc is None")
                    if not hasattr(self, '_temp_text_3d'):
                        print(f"        - _temp_text_3d not found")
                    elif self._temp_text_3d is None:
                        print(f"        - _temp_text_3d is None")
                
                # === 2. 通路级别的Omic特征 ===
                if h_omic_enc is not None and 'x_omic' in kwargs:
                    try:
                        pathway_features = self.omic_pathway_encoder(
                            kwargs['x_omic']  # [B, 2007] 原始基因表达
                        )
                        print(f"     ✅ Pathway features: {pathway_features.shape}")  # 应该是 [B, 50, 256]
                        
                        # 🔥 维度检查
                        if pathway_features.shape[-1] != 256:
                            print(f"     ⚠️ WARNING: Pathway feature dim is {pathway_features.shape[-1]}, expected 256!")
                            
                    except Exception as e:
                        print(f"     ❌ Pathway encoding failed: {e}")
                        pathway_features = None
                else:
                    print(f"     ⚠️ No omic features for pathway encoding")
                
                # === 3. Patch级别的WSI特征 ===
                if h_path_enc is not None and 'x_path' in kwargs and kwargs['x_path'] is not None:
                    try:
                        x_path_batch = kwargs['x_path']
                        
                        # 转换为列表
                        if not isinstance(x_path_batch, list):
                            if x_path_batch.dim() == 3:
                                x_path_batch = [x_path_batch[i] for i in range(x_path_batch.size(0))]
                            else:
                                x_path_batch = [x_path_batch]
                        
                        # 🔥 使用原始patch特征（path_fc的输出）
                        patch_feats_raw = self.path_fc(x_path_batch[0])  # [n_patches, 512]
                        
                        # 🔥 投影到可解释性维度
                        patch_features_explain = self.patch_projector_explain(patch_feats_raw)  # [n_patches, 256]
                        patch_features_explain = patch_features_explain.unsqueeze(0)  # [1, n_patches, 256]
                        
                        print(f"     ✅ Patch features: {patch_features_explain.shape}")  # 应该是 [1, N, 256]
                        
                        # 🔥 维度检查
                        if patch_features_explain.shape[-1] != 256:
                            print(f"     ⚠️ WARNING: Patch feature dim is {patch_features_explain.shape[-1]}, expected 256!")
                            
                    except Exception as e:
                        print(f"     ❌ Patch encoding failed: {e}")
                        import traceback
                        traceback.print_exc()
                        patch_features_explain = None
                else:
                    print(f"     ⚠️ No path features for patch encoding")
                
                # === 4. 计算交叉注意力（QA ↔ Pathway ↔ Patch）===
                if (qa_features is not None and 
                    pathway_features is not None and 
                    patch_features_explain is not None):
                    
                    try:
                        # 🔥 最终维度验证
                        print(f"     🔍 Final dimension check before attention:")
                        print(f"        QA: {qa_features.shape} (expected: [B, 6, 256])")
                        print(f"        Pathway: {pathway_features.shape} (expected: [B, 50, 256])")
                        print(f"        Patch: {patch_features_explain.shape} (expected: [B, N, 256])")
                        
                        # 获取融合权重（如果有）
                        fusion_weights = None
                        if 'fusion_info' in result and 'fusion_weights' in result['fusion_info']:
                            fusion_weights = result['fusion_info']['fusion_weights']
                            print(f"        Fusion weights: {fusion_weights.shape}")
                        
                        # 🔥 调用可解释性模块
                        explanation = self.explainability_module(
                            qa_features=qa_features,
                            pathway_features=pathway_features,
                            patch_features=patch_features_explain,
                            fusion_weights=fusion_weights
                        )
                        
                        # 提取Top-K结果（用于日志）
                        top_k_results = self.explainability_module.get_top_k_explanations(
                            explanation, 
                            k=5, 
                            qa_texts=qa_texts, 
                            pathway_names=pathway_names
                        )
                        
                        # 🔥 保存结果
                        result['explanation'] = {
                            'raw': explanation,
                            'top_k': top_k_results
                        }
                        
                        print(f"     ✅ Explainability computed successfully")
                        
                        # 🔥 打印注意力统计
                        qa2pathway_attn = explanation['qa2pathway_attention'][0]  # [6, 50]
                        qa2patch_attn = explanation['qa2patch_attention'][0]      # [6, N]
                        
                        print(f"     📊 Attention statistics:")
                        print(f"        QA->Pathway: mean={qa2pathway_attn.mean():.4f}, std={qa2pathway_attn.std():.4f}")
                        print(f"        QA->Patch: mean={qa2patch_attn.mean():.4f}, std={qa2patch_attn.std():.4f}")
                        
                    except Exception as e:
                        print(f"     ❌ Explainability module failed: {e}")
                        import traceback
                        traceback.print_exc()
                        result['explanation'] = None
                else:
                    print(f"     ⚠️ Cannot compute explanation - missing modalities")
                    missing = []
                    if qa_features is None: missing.append('QA')
                    if pathway_features is None: missing.append('Pathway')
                    if patch_features_explain is None: missing.append('Patch')
                    print(f"        Missing: {', '.join(missing)}")
                    result['explanation'] = None
                    
            except Exception as e:
                print(f"     ❌ Explainability computation failed: {e}")
                import traceback
                traceback.print_exc()
                result['explanation'] = None
        
        return result

    def relocate(self):
        """移动到GPU"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)


# ========== 训练辅助函数 ==========

def compute_total_loss(outputs, labels=None, survival_time=None, event=None, 
                      task_type='classification',
                      contrastive_weight=0.1,
                      survival_loss_fn=None,
                      volume_reg_weight=0.05,
                      model=None):
    """统一的损失计算函数 - 恢复代码1的完整逻辑"""
    device = outputs['logits'].device if isinstance(outputs, dict) else outputs.device
    
    if isinstance(outputs, dict):
        logits = outputs['logits']
        contrastive_loss = outputs.get('contrastive_loss', torch.tensor(0.0, device=device))
    else:
        logits = outputs
        contrastive_loss = torch.tensor(0.0, device=device)
    
    # 主任务损失
    if task_type == 'classification':
        main_loss = F.cross_entropy(logits, labels)
    elif task_type == 'multi_label':
        # 使用模型内置的 Asymmetric Loss
        if model and hasattr(model, 'criterion'):
            main_loss = model.criterion(logits, labels)
        else:
            main_loss = F.binary_cross_entropy_with_logits(logits, labels)
    elif task_type == 'survival':
        if survival_loss_fn is None:
            raise ValueError("survival_loss_fn required for survival task")
        main_loss = survival_loss_fn(h=logits, y=labels, t=survival_time, c=event)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Volume正则化损失
    volume_reg_loss = torch.tensor(0.0, device=device)
    if isinstance(outputs, dict) and 'fusion_info' in outputs:
        if 'volume_reg_loss' in outputs['fusion_info']:
            volume_reg_loss = outputs['fusion_info']['volume_reg_loss']
    
    # 总损失
    total_loss = (
        main_loss + 
        contrastive_weight * contrastive_loss +
        volume_reg_weight * volume_reg_loss
    )
    
    # 🔥🔥🔥 关键修改：统一转换为 Python float
    def safe_item(x):
        """安全地提取标量值"""
        if isinstance(x, torch.Tensor):
            return x.item()
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return 0.0
    
    # 损失字典 - 使用 safe_item
    loss_dict = {
        'main_loss': safe_item(main_loss),
        'contrastive_loss': safe_item(contrastive_loss),
        'weighted_contrastive_loss': safe_item(contrastive_weight * contrastive_loss),
        'volume_reg_loss': safe_item(volume_reg_loss),
        'total_loss': safe_item(total_loss)
    }
    
    return total_loss, loss_dict

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing GRAMPorpoiseMMF (Fixed Version)")
    print("="*70)
    
    # 测试数据
    batch_size = 8
    n_patches = 100
    
    x_path = [torch.randn(n_patches, 1024) for _ in range(batch_size)]
    x_omic = torch.randn(batch_size, 2007)
    x_text = torch.randn(batch_size, 768)
    
    # ===== 测试1: 分类任务 =====
    print("\n>>> Test 1: Classification Task")
    config_cls = {
        'omic_input_dim': 2007,
        'text_input_dim': 768,
        'path_input_dim': 1024,
        'fusion': 'concat',
        'n_classes': 4,
        'encoder_hidden_dim': 256,
        'use_contrastive': True,
        'task_type': 'classification',
        'use_gram_fusion': True
    }
    
    model_cls = GRAMPorpoiseMMF(**config_cls)
    labels_cls = torch.randint(0, 4, (batch_size,))
    
    outputs_cls = model_cls(
        x_path=x_path,
        x_omic=x_omic,
        x_text=x_text,
        labels=labels_cls,
        compute_loss=True,
        return_features=True
    )
    
    print(f"✅ Logits: {outputs_cls['logits'].shape}")
    if 'fusion_info' in outputs_cls:
        print(f"✅ Has fusion_info with volume_reg_loss: {'volume_reg_loss' in outputs_cls['fusion_info']}")
    
    total_loss, loss_dict = compute_total_loss(
        outputs_cls, labels_cls, 
        task_type='classification',
        model=model_cls
    )
    print(f"✅ Loss: {loss_dict}")
    
    # ===== 测试2: 生存分析任务 =====
    print("\n>>> Test 2: Survival Task")
    config_surv = {
        'omic_input_dim': 2007,
        'text_input_dim': 768,
        'path_input_dim': 1024,
        'fusion': 'concat',
        'n_classes': 4,
        'encoder_hidden_dim': 256,
        'use_contrastive': False,
        'task_type': 'survival',
        'use_gram_fusion': True,
        'use_time_embedding': True,
        'use_risk_stratification': True
    }
    
    model_surv = GRAMPorpoiseMMF(**config_surv)
    survival_time = torch.rand(batch_size) * 100
    event = torch.randint(0, 2, (batch_size,))
    labels_surv = torch.randint(0, 4, (batch_size,))
    
    outputs_surv = model_surv(
        x_path=x_path,
        x_omic=x_omic,
        x_text=x_text,
        survival_time=survival_time,
        event=event,
        compute_loss=True,
        return_features=True
    )
    
    print(f"✅ Logits: {outputs_surv['logits'].shape}")
    if 'fusion_info' in outputs_surv:
        print(f"✅ Has fusion_info: True")
    
    # ===== 测试3: 多标签任务 =====
    print("\n>>> Test 3: Multi-label Task")
    config_ml = {
        'omic_input_dim': 2007,
        'text_input_dim': 768,
        'path_input_dim': 1024,
        'fusion': 'concat',
        'n_labels': 5,
        'encoder_hidden_dim': 256,
        'use_contrastive': False,
        'task_type': 'multi_label',
        'use_gram_fusion': True
    }
    
    model_ml = GRAMPorpoiseMMF(**config_ml)
    labels_ml = torch.randint(0, 2, (batch_size, 5)).float()
    
    outputs_ml = model_ml(
        x_path=x_path,
        x_omic=x_omic,
        x_text=x_text,
        labels=labels_ml,
        compute_loss=True
    )
    
    print(f"✅ Logits: {outputs_ml['logits'].shape}")
    if 'fusion_info' in outputs_ml:
        print(f"✅ Has fusion_info with gates_mean: {'gates_mean' in outputs_ml['fusion_info']}")
    
    total_loss, loss_dict = compute_total_loss(
        outputs_ml, labels_ml, 
        task_type='multi_label',
        model=model_ml
    )
    print(f"✅ Loss (Asymmetric): {loss_dict}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)