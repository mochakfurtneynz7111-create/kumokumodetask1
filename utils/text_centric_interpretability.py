"""
文本为纽带的三模态可解释性
====================================
核心思想：
1. 文本（6个QA）作为语义锚点
2. 找出哪些通路支持每个QA
3. 找出哪些Patch支持每个QA
4. 建立Patch与通路的联系
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from PIL import Image
import pandas as pd

class TextCentricExplainer:
    """文本为纽带的可解释性分析器"""
    
    def __init__(self, model, save_dir, device='cuda',
                 preprocessing_dir='./outputs_COADREAD/preprocessing',
                 n_qa_pairs=6, n_pathways=50, heatmap_generator=None):
        
        self.model = model
        self.save_dir = save_dir
        self.device = device
        self.preprocessing_dir = preprocessing_dir
        self.n_qa_pairs = n_qa_pairs
        self.n_pathways = n_pathways

        # 🔥🔥🔥 关键修复：使用共享的 WSIHeatmapGenerator
        if heatmap_generator is not None:
            self.heatmap_gen = heatmap_generator
            print(f"✅ TextCentricExplainer: Using shared WSIHeatmapGenerator")
        else:
            # 🔥 如果没有传入，自己创建一个（但应该避免）
            print(f"⚠️ TextCentricExplainer: No heatmap_generator provided, creating new one")
            # 🔥 这里需要导入 WSIHeatmapGenerator
            # 选项1：如果在同一个文件
            # self.heatmap_gen = WSIHeatmapGenerator(preprocessing_dir)
            
            # 选项2：如果在 interpretability_module.py
            try:
                from interpretability_module import WSIHeatmapGenerator
                self.heatmap_gen = WSIHeatmapGenerator(preprocessing_dir)
            except ImportError:
                print(f"❌ Could not import WSIHeatmapGenerator!")
                self.heatmap_gen = None
        
        os.makedirs(save_dir, exist_ok=True)

        print(f"✅ TextCentricExplainer initialized")
        print(f"   Save dir: {save_dir}")
        print(f"   n_qa_pairs: {n_qa_pairs}")
        print(f"   n_pathways: {n_pathways}")
    
    def explain_single_sample(self,
                         data_WSI,
                         data_omic,
                         data_text,
                         slide_id,
                         sample_id,
                         qa_texts=None,
                         pathway_names=None,
                         coords=None,
                         survival_time=None,
                         event=None):
        """
        为单个样本生成文本纽带的解释
        
        Args:
            data_WSI: WSI特征 [n_patches, feature_dim]
            data_omic: Omic特征 [omic_dim]
            data_text: Text特征 [text_dim]
            slide_id: Slide ID (用于加载缩略图和坐标)
            sample_id: Sample ID (用于保存文件)
            qa_texts: 6个QA的文本描述 (可选)
            pathway_names: 通路名称列表 (可选)
            coords: Patch坐标 [n_patches, 2] (可选，会自动加载)
            survival_time: 生存时间 (可选)
            event: 事件指示器 (可选)
        
        Returns:
            explanation_dict: 包含所有解释信息的字典
        """
        print(f"\n🔍 Generating text-centric explanation for {sample_id}...")
        
        # === Step 1: 准备数据 ===
        if qa_texts is None:
            qa_texts = [
                "What is the tumor grade and differentiation?",
                "Is there immune cell infiltration?",
                "What is the tumor proliferation status?",
                "Is there necrosis or hemorrhage?",
                "What is the stromal composition?",
                "Are there specific morphological features?"
            ]
        
        if pathway_names is None:
            pathway_names = [f"Pathway_{i+1}" for i in range(self.n_pathways)]
        
        # === Step 2: 调用模型的可解释性forward ===
        with torch.no_grad():
            forward_kwargs = {
                'x_path': [data_WSI],
                'x_omic': data_omic.unsqueeze(0) if data_omic.dim() == 1 else data_omic,
                'compute_loss': False,
                'return_explanation': True,  # 🔥 关键参数
                'qa_texts': qa_texts,
                'pathway_names': pathway_names,
            }
            
            if data_text is not None:
                # 🔥🔥🔥 关键修复
                print(f"  🔍 Input data_text shape: {data_text.shape}")
                
                if data_text.dim() == 1:
                    # [768] -> [1, 768]
                    forward_kwargs['x_text'] = data_text.unsqueeze(0)
                    print(f"     1D->2D: {forward_kwargs['x_text'].shape}")
                    
                elif data_text.dim() == 2:
                    # 检查是否是QA级别 [6, 768]
                    if data_text.size(0) == 6 or data_text.size(0) == self.n_qa_pairs:
                        # [6, 768] -> [1, 6, 768]
                        forward_kwargs['x_text'] = data_text.unsqueeze(0)
                        print(f"     QA-level [6,768]->[1,6,768]: {forward_kwargs['x_text'].shape}")
                    else:
                        # [batch, 768] -> 取第一个
                        forward_kwargs['x_text'] = data_text[0:1]
                        print(f"     Batch mode, take first: {forward_kwargs['x_text'].shape}")
                        
                elif data_text.dim() == 3:
                    # [batch, 6, 768] -> 取第一个
                    forward_kwargs['x_text'] = data_text[0:1]
                    print(f"     3D batch mode, take first: {forward_kwargs['x_text'].shape}")
                    
                else:
                    print(f"  ⚠️ Unexpected text dimension: {data_text.shape}")
                    forward_kwargs['x_text'] = data_text
            
            if survival_time is not None:
                forward_kwargs['survival_time'] = survival_time.unsqueeze(0) if survival_time.dim() == 0 else survival_time
            
            if event is not None:
                forward_kwargs['event'] = event.unsqueeze(0) if event.dim() == 0 else event
            
            # 🔥 检查模型是否启用了可解释性
            if not hasattr(self.model, 'enable_explainability') or not self.model.enable_explainability:
                print("  ⚠️ Model does not have explainability enabled!")
                return self._fallback_explanation(data_WSI, data_omic, data_text, 
                                                 qa_texts, pathway_names)
            
            outputs = self.model(**forward_kwargs)
        
        # === Step 3: 提取可解释性结果 ===
        if 'explanation' not in outputs or outputs['explanation'] is None:
            print("  ⚠️ Model did not return explanation, using fallback")
            return self._fallback_explanation(data_WSI, data_omic, data_text, 
                                             qa_texts, pathway_names)
        
        explanation = outputs['explanation']['raw']
        
        # === Step 4: 可视化 ===
        self._visualize_text_centric_explanation(
            explanation=explanation,
            slide_id=slide_id,
            sample_id=sample_id,
            qa_texts=qa_texts,
            pathway_names=pathway_names,
            coords=coords
        )
        
        return explanation   
        
    def _visualize_text_centric_explanation(self,
                                           explanation,
                                           slide_id,
                                           sample_id,
                                           qa_texts,
                                           pathway_names,
                                           coords=None):
        """
        可视化文本为纽带的解释
        
        布局：
        ┌────────────────┬────────────────┐
        │  QA1           │  QA2           │
        │  - Pathways    │  - Pathways    │
        │  - Patches     │  - Patches     │
        ├────────────────┼────────────────┤
        │  QA3           │  QA4           │
        │  - Pathways    │  - Pathways    │
        │  - Patches     │  - Patches     │
        ├────────────────┼────────────────┤
        │  QA5           │  QA6           │
        │  - Pathways    │  - Pathways    │
        │  - Patches     │  - Patches     │
        └────────────────┴────────────────┘
        
        下方：Patch-Pathway对应热力图
        """
        
        # 提取数据
        qa2pathway_attn = explanation['qa2pathway_attention'][0].cpu().numpy()  # [6, n_pathways]
        qa2patch_attn = explanation['qa2patch_attention'][0].cpu().numpy()      # [6, n_patches]
        patch_pathway_map = explanation['patch_pathway_correspondence'][0].cpu().numpy()  # [n_patches, n_pathways]
        
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3,
                     height_ratios=[1, 1, 1, 1.2])
        
        # === 为每个QA创建子图 ===
        for qa_idx in range(min(self.n_qa_pairs, 6)):
            row = qa_idx // 2
            col = qa_idx % 2
            
            ax = fig.add_subplot(gs[row, col])
            
            # QA标题
            qa_text = qa_texts[qa_idx]
            ax.text(0.5, 0.98, qa_text, 
                   ha='center', va='top', fontsize=11, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # === 左半部分：Top通路 ===
            pathway_scores = qa2pathway_attn[qa_idx]
            top_pathway_indices = np.argsort(pathway_scores)[-5:][::-1]
            top_pathway_scores = pathway_scores[top_pathway_indices]
            top_pathway_names = [pathway_names[i][:30] for i in top_pathway_indices]
            
            y_positions = np.arange(5)
            colors = plt.cm.Reds(top_pathway_scores / top_pathway_scores.max())
            
            ax.barh(y_positions, top_pathway_scores, color=colors, alpha=0.7, height=0.6)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(top_pathway_names, fontsize=8)
            ax.set_xlabel('Pathway Importance', fontsize=9)
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # 添加分数标签
            for i, score in enumerate(top_pathway_scores):
                ax.text(score + 0.02, i, f'{score:.3f}', 
                       va='center', fontsize=8, fontweight='bold')
            
            ax.set_title(f'Top-5 Related Pathways', fontsize=9, pad=20)
        
            # === 右侧区域：WSI概览 + Top Patches细节 ===
            if coords is not None:
                # 🔥🔥🔥 上半部分：缩略图 + 标注框（空间上下文）
                ax_inset_overview = ax.inset_axes([0.60, 0.50, 0.38, 0.45])
                
                # 加载缩略图
                thumbnail = self._load_thumbnail(slide_id)
                
                if thumbnail is not None:
                    ax_inset_overview.imshow(thumbnail)
                    
                    # 🔥 标注Top-3 patches
                    patch_scores = qa2patch_attn[qa_idx]
                    top_patch_indices = np.argsort(patch_scores)[-3:][::-1]
                    
                    # 计算坐标变换
                    min_x = coords[:, 0].min()
                    min_y = coords[:, 1].min()
                    roi_width = coords[:, 0].max() + 256 - min_x
                    roi_height = coords[:, 1].max() + 256 - min_y
                    scale_x = thumbnail.shape[1] / roi_width
                    scale_y = thumbnail.shape[0] / roi_height
                    
                    colors = ['red', 'orange', 'yellow']
                    for rank, patch_idx in enumerate(top_patch_indices):
                        x, y = coords[patch_idx]
                        score = patch_scores[patch_idx]
                        
                        x_rel = x - min_x
                        y_rel = y - min_y
                        x_thumb = x_rel * scale_x
                        y_thumb = y_rel * scale_y
                        w_thumb = 256 * scale_x
                        h_thumb = 256 * scale_y
                        
                        rect = plt.Rectangle((x_thumb, y_thumb), w_thumb, h_thumb,
                                            fill=False, edgecolor=colors[rank], linewidth=2.5)
                        ax_inset_overview.add_patch(rect)
                        
                        # 🔥 标注编号
                        ax_inset_overview.text(x_thumb + w_thumb/2, y_thumb + h_thumb/2,
                                              f'{rank+1}',
                                              color='white', fontsize=12, fontweight='bold',
                                              ha='center', va='center',
                                              bbox=dict(boxstyle='circle', facecolor=colors[rank], 
                                                       alpha=0.8, edgecolor='white', linewidth=2))
                    
                    ax_inset_overview.set_title('WSI Overview (Top-3)', fontsize=9, fontweight='bold')
                    ax_inset_overview.axis('off')
                else:
                    ax_inset_overview.text(0.5, 0.5, 'No\nThumbnail', 
                                          ha='center', va='center', fontsize=9)
                    ax_inset_overview.axis('off')
                
                # 🔥🔥🔥 下半部分：Top-3 Patches的高清图（细节）
                ax_inset_detail = ax.inset_axes([0.60, 0.02, 0.38, 0.45])
                
                patch_scores = qa2patch_attn[qa_idx]
                top_patch_indices = np.argsort(patch_scores)[-3:][::-1]
                
                # 🔥 加载Top-3 patches的原始图像
                patch_images = []
                patch_labels = []
                
                for rank, patch_idx in enumerate(top_patch_indices):
                    tile_x, tile_y = coords[patch_idx]
                    score = patch_scores[patch_idx]
                    
                    # 🔥 加载原始tile图像
                    patch_img = self._load_patch_image(slide_id, tile_x, tile_y)
                    
                    if patch_img is not None:
                        patch_images.append(patch_img)
                        patch_labels.append(f'#{rank+1}\nScore: {score:.3f}')
                    else:
                        # 🔥 如果加载失败，用占位符
                        placeholder = np.ones((256, 256, 3), dtype=np.uint8) * 200
                        patch_images.append(placeholder)
                        patch_labels.append(f'#{rank+1}\nN/A')
                
                if len(patch_images) > 0:
                    # 🔥 拼接成1x3的网格（横向排列）
                    combined = np.concatenate(patch_images, axis=1)
                    
                    ax_inset_detail.imshow(combined)
                    
                    # 🔥 在每个patch下方添加标签
                    patch_width = patch_images[0].shape[1]
                    colors = ['red', 'orange', 'yellow']
                    
                    for i, label in enumerate(patch_labels):
                        x_center = patch_width * i + patch_width / 2
                        ax_inset_detail.text(x_center, combined.shape[0] + 20,
                                            label,
                                            ha='center', va='top', fontsize=8,
                                            fontweight='bold', color=colors[i],
                                            bbox=dict(boxstyle='round', facecolor='white',
                                                     alpha=0.9, edgecolor=colors[i], linewidth=2))
                    
                    ax_inset_detail.set_title('Top-3 Patches (High-Res)', fontsize=9, fontweight='bold')
                    ax_inset_detail.axis('off')
                    ax_inset_detail.set_xlim(0, combined.shape[1])
                    ax_inset_detail.set_ylim(combined.shape[0] + 40, 0)  # 留空间给标签
                else:
                    ax_inset_detail.text(0.5, 0.5, 'No Patch\nImages', 
                                        ha='center', va='center', fontsize=9)
                    ax_inset_detail.axis('off')
        
        # === 底部：Patch-Pathway对应热力图 ===
        ax_heatmap = fig.add_subplot(gs[3, :])
        
        # 选择Top pathways和Top patches
        top_pathways_global = qa2pathway_attn.mean(axis=0).argsort()[-10:][::-1]
        top_patches_global = qa2patch_attn.mean(axis=0).argsort()[-15:][::-1]
        
        heatmap_data = patch_pathway_map[top_patches_global][:, top_pathways_global]
        
        sns.heatmap(heatmap_data, 
                   cmap='YlOrRd', 
                   ax=ax_heatmap,
                   cbar_kws={'label': 'Correspondence Score'},
                   xticklabels=[pathway_names[i][:20] for i in top_pathways_global],
                   yticklabels=[f'Patch {i}' for i in top_patches_global],
                   annot=False)
        
        ax_heatmap.set_title('Patch-Pathway Correspondence Heatmap\n(Shows which patches correspond to which pathways)', 
                            fontsize=12, fontweight='bold', pad=10)
        ax_heatmap.set_xlabel('Pathways', fontsize=10)
        ax_heatmap.set_ylabel('Patches', fontsize=10)
        
        # 总标题
        fig.suptitle(f'Text-Centric Multimodal Explanation\n{sample_id} | {slide_id[:40]}...', 
                    fontsize=14, fontweight='bold', y=0.99)
        
        # 保存
        save_path = os.path.join(self.save_dir, f'{sample_id}_text_centric.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved text-centric visualization: {os.path.basename(save_path)}")
    
    def _load_thumbnail(self, slide_id):
        """加载缩略图 - 使用共享的 WSIHeatmapGenerator"""
        
        # 🔥🔥🔥 直接使用 WSIHeatmapGenerator 的强大加载器
        if self.heatmap_gen is not None:
            return self.heatmap_gen.load_thumbnail(slide_id)
        else:
            print(f"    ⚠️ No heatmap_gen available!")
            return None

    def _load_patch_image(self, slide_id, tile_x, tile_y):
        """
        加载单个patch的原始图像
        
        Args:
            slide_id: Slide ID
            tile_x: Tile的x坐标
            tile_y: Tile的y坐标
        
        Returns:
            numpy array [H, W, 3] 或 None
        """
        output_dir = os.path.join(self.preprocessing_dir, 'output')
        
        # 🔥 构建图片路径
        # 格式：output/slide_id/15984x_03792y.png
        tile_filename = f"{int(tile_x):05d}x_{int(tile_y):05d}y.png"
        tile_path = os.path.join(output_dir, slide_id, tile_filename)
        
        if os.path.exists(tile_path):
            try:
                img = Image.open(tile_path)
                return np.array(img.convert('RGB'))
            except Exception as e:
                print(f"    ⚠️ Failed to load tile {tile_filename}: {e}")
                return None
        else:
            print(f"    ⚠️ Tile not found: {tile_path}")
            return None
    
    def _fallback_explanation(self, data_WSI, data_omic, data_text,
                             qa_texts, pathway_names):
        """
        降级方案：如果模型没有提供可解释性，生成模拟数据
        （仅用于演示，实际应该修改模型）
        """
        n_patches = data_WSI.size(0)
        
        # 生成随机attention（仅作演示）
        qa2pathway_attn = torch.softmax(torch.randn(self.n_qa_pairs, self.n_pathways), dim=-1)
        qa2patch_attn = torch.softmax(torch.randn(self.n_qa_pairs, n_patches), dim=-1)
        patch_pathway_map = torch.softmax(torch.randn(n_patches, self.n_pathways), dim=-1)
        
        return {
            'qa2pathway_attention': qa2pathway_attn.unsqueeze(0),
            'qa2patch_attention': qa2patch_attn.unsqueeze(0),
            'patch_pathway_correspondence': patch_pathway_map.unsqueeze(0)
        }
    
    def analyze_batch(self, data_WSI_list, data_omic, data_text,
                     slide_ids, sample_ids, max_visualize=2,
                     qa_texts=None, pathway_names=None,
                     survival_times=None, events=None):
        """批量分析"""
        
        num_samples = min(len(slide_ids), max_visualize)
        
        print(f"\n📊 Starting text-centric explanation for {num_samples} samples...")
        
        for i in range(num_samples):
            try:
                sample_wsi = data_WSI_list[i] if isinstance(data_WSI_list, list) else data_WSI_list[i]
                sample_omic = data_omic[i]
                
                # 🔥🔥🔥 正确提取text
                if data_text is not None:
                    if data_text.dim() == 3:
                        sample_text = data_text[i]  # [batch, 6, 768] -> [6, 768]
                    elif data_text.dim() == 2:
                        sample_text = data_text[i]  # [batch, 768] -> [768]
                    else:
                        sample_text = data_text
                else:
                    sample_text = None
                    
                # 尝试加载坐标
                coords = self._load_coords(slide_ids[i])
                
                self.explain_single_sample(
                    data_WSI=data_WSI_list[i] if isinstance(data_WSI_list, list) else data_WSI_list[i:i+1],
                    data_omic=data_omic[i],
                    data_text=data_text[i] if data_text is not None else None,
                    slide_id=slide_ids[i],
                    sample_id=sample_ids[i],
                    qa_texts=qa_texts,
                    pathway_names=pathway_names,
                    coords=coords,
                    survival_time=survival_times[i] if survival_times is not None else None,
                    event=events[i] if events is not None else None
                )
            except Exception as e:
                print(f"  ⚠️ Failed for sample {i}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Text-centric explanation complete!\n")
    
    def _load_coords(self, slide_id):
        """加载坐标 - 使用共享的 WSIHeatmapGenerator"""
        
        # 🔥🔥🔥 直接使用 WSIHeatmapGenerator 的方法
        if self.heatmap_gen is not None:
            return self.heatmap_gen.load_coords_from_csv(slide_id)
        else:
            print(f"    ⚠️ No heatmap_gen available!")
            return None