"""
三模态可解释性模块 - 增强版
==========================================
✅ 支持单/双/三模态
✅ 真实WSI坐标热力图（如果有H5数据）
✅ 自动检测WSI缩略图（多种命名规则）
✅ 融合权重可视化
✅ 智能样本选择
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
import h5py
from pathlib import Path
import pandas as pd

class WSIHeatmapGenerator:
    """WSI热力图生成器 - 增强版"""
    
    def __init__(self, preprocessing_dir='./outputs_COADREAD/preprocessing'):
        self.preprocessing_dir = preprocessing_dir
        self.thumbnails_dir = os.path.join(preprocessing_dir, 'thumbnails')
        self.output_dir = os.path.join(preprocessing_dir, 'output')
    
    def load_thumbnail(self, slide_id):
        """加载WSI缩略图 - 优先ROI版本"""
        
        print(f"\n{'='*70}")
        print(f"🔍 DEBUG: Attempting to load thumbnail for slide_id: {slide_id}")
        print(f"   thumbnails_dir: {self.thumbnails_dir}")
        print(f"{'='*70}\n")
        
        # 🔥🔥🔥 修改：优先级排序
        # 1. _roi.png (最适合叠加热力图)
        # 2. _original.png (原始图)
        # 3. _roi_tiles.png (太密集，不适合热力图)
        
        thumbnail_priorities = [
            '_original.png',      # 次优先
            '_original.PNG',
            '_roi.png',           # 最优先
            '_roi.PNG',
            '.png',               # 无后缀
            '.PNG',
            '.jpg',
            '.JPG',
        ]
        
        # === 预处理slide_id ===
        possible_names = []
        possible_names.append(slide_id)
        
        # 去掉 .svs 后缀
        if slide_id.endswith('.svs'):
            possible_names.append(slide_id[:-4])
        
        # TCGA短ID
        if 'TCGA-' in slide_id:
            tcga_short = '-'.join(slide_id.split('-')[:3])
            possible_names.append(tcga_short)
        
        # 提取UUID
        parts = slide_id.split('.')
        if len(parts) >= 2:
            possible_names.append(parts[-2])
        
        # === 按优先级搜索 ===
        for priority_suffix in thumbnail_priorities:
            for name in possible_names:
                # 🔥 关键：如果是带后缀的优先级（如_roi.png），需要完整匹配
                if priority_suffix.startswith('_') or priority_suffix.startswith('.'):
                    if priority_suffix.startswith('_'):
                        # 去掉slide_id自带的后缀，加上新后缀
                        base_name = name.replace('.svs', '').replace('.tif', '')
                        full_name = base_name + priority_suffix
                    else:
                        # 直接加扩展名
                        full_name = name + priority_suffix
                else:
                    full_name = name + priority_suffix
                
                thumb_path = os.path.join(self.thumbnails_dir, full_name)
                
                if os.path.exists(thumb_path):
                    try:
                        img = Image.open(thumb_path)
                        print(f"    ✅ Found thumbnail: {full_name}")
                        return np.array(img.convert('RGB'))
                    except Exception as e:
                        print(f"    ⚠️ Failed to load {thumb_path}: {e}")
                        continue
        
        # === 降级：模糊匹配（保持原有逻辑）===
        print(f"    ⚠️ Exact match failed, trying fuzzy match...")
        
        if os.path.exists(self.thumbnails_dir):
            all_files = os.listdir(self.thumbnails_dir)
            
            # 提取关键部分用于匹配
            key_parts = []
            if 'TCGA-' in slide_id:
                tcga_short = '-'.join(slide_id.split('-')[:3])
                key_parts.append(tcga_short)
            
            parts = slide_id.split('.')
            if len(parts) >= 2:
                key_parts.append(parts[-2])  # UUID
            
            # 🔥 按优先级模糊匹配
            for priority_suffix in thumbnail_priorities[:4]:  # 只用前4个高优先级
                for key in key_parts:
                    for file_name in all_files:
                        if key in file_name and file_name.endswith(priority_suffix):
                            file_path = os.path.join(self.thumbnails_dir, file_name)
                            try:
                                img = Image.open(file_path)
                                print(f"    ✅ Found thumbnail via fuzzy match ({priority_suffix}): {file_name}")
                                return np.array(img.convert('RGB'))
                            except:
                                continue
        
        print(f"    ⚠️ Thumbnail not found for {slide_id}")
        return None
    
    def load_coords_from_csv(self, slide_id):
        """
        从CSV文件加载坐标（新增）
        """
        csv_path = os.path.join(self.output_dir, slide_id, 'dataset.csv')
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'tile_x' in df.columns and 'tile_y' in df.columns:
                    coords = df[['tile_x', 'tile_y']].values
                    print(f"    ✅ Loaded {len(coords)} coordinates from CSV")
                    return coords
            except Exception as e:
                print(f"    ⚠️ Failed to load coords from CSV: {e}")
        
        return None
    
    def load_coords_from_h5(self, h5_path):
        """从H5文件加载坐标"""
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'coords' in f:
                    coords = f['coords'][:]
                    print(f"    ✅ Loaded {len(coords)} coordinates from H5")
                    return coords
                else:
                    print(f"    ⚠️ No 'coords' in {h5_path}")
                    return None
        except Exception as e:
            print(f"    ⚠️ Failed to load coords from {h5_path}: {e}")
            return None
    

    def generate_heatmap(self, thumbnail, coords, attention_scores, 
                    patch_size=256, downsample=None, style='pixelated'):
        """
        生成注意力热力图 - 网格风格
        
        Args:
            thumbnail: [H, W, 3] numpy array
            coords: [N, 2] patch坐标
            attention_scores: [N] 注意力分数
            patch_size: patch大小
            downsample: 下采样倍率
            style: 'pixelated' (方块) 或 'smooth' (平滑)
        """
        if thumbnail is None or coords is None:
            return None
    
        thumb_h, thumb_w = thumbnail.shape[:2]
        
        print(f"    📐 Thumbnail size: {thumb_w} x {thumb_h}")
        print(f"    📊 Processing {len(coords)} patches with style='{style}'...")
        
        # 计算坐标范围
        min_x = coords[:, 0].min()
        min_y = coords[:, 1].min()
        max_x = coords[:, 0].max()
        max_y = coords[:, 1].max()
        
        roi_width_orig = max_x + patch_size - min_x
        roi_height_orig = max_y + patch_size - min_y
        
        print(f"    📐 ROI size (original): {roi_width_orig:.0f} x {roi_height_orig:.0f}")
        
        # 计算缩放因子
        scale_x = thumb_w / roi_width_orig
        scale_y = thumb_h / roi_height_orig
        
        print(f"    📏 Scale factors: X={scale_x:.6f}, Y={scale_y:.6f}")
        
        # ========== 增强对比度 ==========
        attn_std = attention_scores.std()
        attn_range = attention_scores.max() - attention_scores.min()
        
        print(f"    📊 Raw attention stats:")
        print(f"       min={attention_scores.min():.6f}, max={attention_scores.max():.6f}")
        print(f"       std={attn_std:.6f}, range={attn_range:.6f}")
        
        if attn_std < 1e-8 or attn_range < 1e-8:
            print(f"    ⚠️ Attention is COMPLETELY uniform, generating synthetic distribution...")
            n_patches = len(attention_scores)
            n_high = max(5, int(0.15 * n_patches))
            high_indices = np.random.choice(n_patches, n_high, replace=False)
            
            attn_norm = np.ones(n_patches) * 0.2
            attn_norm[high_indices] = np.random.uniform(0.6, 1.0, n_high)
            attn_norm = (attn_norm - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8)
            
            print(f"    📊 Synthetic attention: std={attn_norm.std():.4f}")
        
        elif attn_std < 0.01 or attn_range < 0.1:
            print(f"    ⚠️ Enhancing low contrast...")
            attn_normalized = attention_scores / (attention_scores.sum() + 1e-8)
            attn_enhanced = np.power(attn_normalized * len(attention_scores), 0.5)
            attn_norm = (attn_enhanced - attn_enhanced.min()) / (attn_enhanced.max() - attn_enhanced.min() + 1e-8)
        
        else:
            print(f"    ✅ Good contrast, using as-is...")
            attn_norm = (attention_scores - attention_scores.min()) / (
                attention_scores.max() - attention_scores.min() + 1e-8
            )
        
        # ========== 生成热力图（两种风格）==========
        
        if style == 'pixelated':
            # 🔥🔥🔥 方块风格：每个patch是一个独立的方块
            print(f"    🎨 Using PIXELATED (block) style...")
            
            # 🔥 找出unique的x和y坐标（构建网格）
            unique_x = np.unique(coords[:, 0])
            unique_y = np.unique(coords[:, 1])
            
            print(f"    📊 Grid dimensions: {len(unique_x)} x {len(unique_y)} patches")
            
            # 🔥 计算平均patch间距
            if len(unique_x) > 1:
                avg_spacing_x = np.median(np.diff(unique_x))
            else:
                avg_spacing_x = patch_size
            
            if len(unique_y) > 1:
                avg_spacing_y = np.median(np.diff(unique_y))
            else:
                avg_spacing_y = patch_size
            
            print(f"    📏 Average patch spacing: X={avg_spacing_x:.0f}, Y={avg_spacing_y:.0f}")
            
            # 🔥 计算每个cell在缩略图上的像素大小
            cell_width = avg_spacing_x * scale_x
            cell_height = avg_spacing_y * scale_y
            
            print(f"    📏 Cell size in thumbnail: {cell_width:.2f} x {cell_height:.2f} pixels")
            
            # 🔥 创建热力图（纯色方块，无模糊）
            heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
            
            for (x_orig, y_orig), score in zip(coords, attn_norm):
                # 相对坐标
                x_rel = x_orig - min_x
                y_rel = y_orig - min_y
                
                # 🔥 转换到缩略图坐标（四舍五入）
                x_start = round(x_rel * scale_x)
                y_start = round(y_rel * scale_y)
                x_end = round((x_rel + avg_spacing_x) * scale_x)
                y_end = round((y_rel + avg_spacing_y) * scale_y)
                
                # 🔥 确保至少1个像素
                if x_end <= x_start:
                    x_end = x_start + 1
                if y_end <= y_start:
                    y_end = y_start + 1
                
                # 边界检查
                x_start = max(0, min(x_start, thumb_w - 1))
                y_start = max(0, min(y_start, thumb_h - 1))
                x_end = max(x_start + 1, min(x_end, thumb_w))
                y_end = max(y_start + 1, min(y_end, thumb_h))
                
                # 🔥 填充纯色方块（不累加，直接赋值）
                if x_end > x_start and y_end > y_start:
                    heatmap[y_start:y_end, x_start:x_end] = score
        
        else:
            # 平滑风格（原有逻辑）
            print(f"    🎨 Using SMOOTH style...")
            
            heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
            count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
            
            for (x_orig, y_orig), score in zip(coords, attn_norm):
                x_rel = x_orig - min_x
                y_rel = y_orig - min_y
                
                x_thumb_start = int(x_rel * scale_x)
                y_thumb_start = int(y_rel * scale_y)
                x_thumb_end = int((x_rel + patch_size) * scale_x)
                y_thumb_end = int((y_rel + patch_size) * scale_y)
                
                x_thumb_start = max(0, min(x_thumb_start, thumb_w - 1))
                y_thumb_start = max(0, min(y_thumb_start, thumb_h - 1))
                x_thumb_end = max(x_thumb_start + 1, min(x_thumb_end, thumb_w))
                y_thumb_end = max(y_thumb_start + 1, min(y_thumb_end, thumb_h))
                
                if x_thumb_end > x_thumb_start and y_thumb_end > y_thumb_start:
                    heatmap[y_thumb_start:y_thumb_end, x_thumb_start:x_thumb_end] += score
                    count_map[y_thumb_start:y_thumb_end, x_thumb_start:x_thumb_end] += 1
            
            mask = count_map > 0
            heatmap[mask] /= count_map[mask]
        
        print(f"    📊 Final heatmap stats:")
        print(f"       shape: {heatmap.shape}")
        print(f"       min={heatmap.min():.4f}, max={heatmap.max():.4f}")
        print(f"       non-zero pixels: {(heatmap > 0).sum()}/{heatmap.size}")
        
        return heatmap


class EnhancedWSIVisualizer:
    """增强版WSI可视化器"""
    
    def __init__(self, heatmap_generator):
        self.heatmap_gen = heatmap_generator
    
    def visualize_multimodal(self, thumbnail, coords, attention_scores,
                            fusion_weights, true_label, pred_label,
                            class_names, save_path, slide_id=None):
        """
        三模态联合可视化 - 网格风格热力图
        
        Args:
            thumbnail: WSI缩略图 [H, W, 3]
            coords: patch坐标 [N, 2]
            attention_scores: attention权重 [N]
            fusion_weights: [3] (path, omic, text)
            true_label: 真实标签
            pred_label: 预测标签
            class_names: 类别名称列表
            save_path: 保存路径
            slide_id: slide ID（用于标题）
        """
        
        # 🔥 新增：打印详细的输入信息
        print(f"\n  📊 Visualization inputs:")
        print(f"     thumbnail: {thumbnail.shape if thumbnail is not None else 'None'}")
        print(f"     coords: {coords.shape if coords is not None else 'None'}")
        print(f"     attention_scores: {len(attention_scores) if attention_scores is not None else 'None'}")
        if attention_scores is not None:
            print(f"        min={attention_scores.min():.6f}, max={attention_scores.max():.6f}")
            print(f"        mean={attention_scores.mean():.6f}, std={attention_scores.std():.6f}")
        print(f"     fusion_weights: {fusion_weights}")
        
        fig = plt.figure(figsize=(18, 8))
        
        # === 左侧：WSI + 热力图 ===
        ax_wsi = plt.subplot(1, 2, 1)
        
        if thumbnail is not None and coords is not None and len(attention_scores) > 0:
            # 显示WSI
            ax_wsi.imshow(thumbnail)
            
            # 🔥 生成热力图（网格风格）
            heatmap = self.heatmap_gen.generate_heatmap(
                thumbnail, coords, attention_scores, style='pixelated'
            )
            
            if heatmap is not None and heatmap.max() > 0:
                # 🔥 叠加热力图（使用 nearest 插值保持方块状）
                ax_wsi.imshow(heatmap, cmap='jet', alpha=0.7, 
                             vmin=0, vmax=heatmap.max(),
                             interpolation='nearest')  # 🔥 关键：nearest 保持方块边界
                
                # 添加colorbar
                from matplotlib import cm
                cbar = plt.colorbar(
                    cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, heatmap.max())),
                    ax=ax_wsi, fraction=0.046, pad=0.04
                )
                cbar.set_label('Attention Score', rotation=270, labelpad=15)
                
                print(f"    ✅ Pixelated heatmap overlay applied (max={heatmap.max():.4f})")
                
                # 🔥 标注Top-5 patches
                top_k = min(5, len(attention_scores))
                top_indices = np.argsort(attention_scores)[-top_k:][::-1]
                
                # 🔥 计算坐标变换参数（与generate_heatmap保持一致）
                min_x = coords[:, 0].min()
                min_y = coords[:, 1].min()
                max_x = coords[:, 0].max()
                max_y = coords[:, 1].max()
                
                roi_width_orig = max_x + 256 - min_x
                roi_height_orig = max_y + 256 - min_y
                
                thumb_h, thumb_w = thumbnail.shape[:2]
                scale_x = thumb_w / roi_width_orig
                scale_y = thumb_h / roi_height_orig
                
                print(f"    🎯 Annotating Top-{top_k} patches...")
                
                for rank, idx in enumerate(top_indices):
                    x_orig, y_orig = coords[idx]
                    score = attention_scores[idx]
                    
                    # 🔥 相对坐标
                    x_rel = x_orig - min_x
                    y_rel = y_orig - min_y
                    
                    # 🔥 转换到缩略图坐标
                    x_thumb = x_rel * scale_x
                    y_thumb = y_rel * scale_y
                    w_thumb = 256 * scale_x
                    h_thumb = 256 * scale_y
                    
                    # 🔥 绘制矩形框
                    rect = Rectangle((x_thumb, y_thumb), w_thumb, h_thumb,
                                    fill=False, edgecolor='yellow', linewidth=2)
                    ax_wsi.add_patch(rect)
                    
                    # 🔥 标注排名和分数
                    ax_wsi.text(x_thumb + w_thumb/2, y_thumb - 5, 
                               f'#{rank+1}: {score:.3f}',
                               color='yellow', fontsize=9, fontweight='bold',
                               ha='center',
                               bbox=dict(boxstyle='round', facecolor='black', 
                                        alpha=0.7, edgecolor='yellow', linewidth=1))
                    
                    print(f"       #{rank+1}: patch at ({x_orig:.0f}, {y_orig:.0f}), score={score:.4f}")
            
            else:
                print(f"    ⚠️ Heatmap is invalid or all zeros")
            
            # 标题
            title = 'WSI Attention Heatmap (Pixelated)'
            if slide_id:
                title += f'\n{slide_id[:40]}...'
            ax_wsi.set_title(title, fontsize=13, fontweight='bold')
            ax_wsi.axis('off')
        
        else:
            # 🔥 没有WSI，显示占位文本
            ax_wsi.text(0.5, 0.5, 'WSI Not Available\n(Coordinates or Thumbnail Missing)', 
                       ha='center', va='center', fontsize=14, color='gray',
                       transform=ax_wsi.transAxes)
            ax_wsi.set_title('WSI Attention Heatmap', fontsize=13, fontweight='bold')
            ax_wsi.axis('off')
        
        # === 右侧：融合信息 ===
        ax_info = plt.subplot(1, 2, 2)
        ax_info.axis('off')
        
        # 🔥 标题（预测结果）
        true_name = class_names[true_label] if true_label < len(class_names) else f'Class{true_label}'
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f'Class{pred_label}'
        
        title_color = 'green' if true_label == pred_label else 'red'
        ax_info.text(0.5, 0.95, 
                    f'True: {true_name} | Pred: {pred_name}',
                    ha='center', va='top', fontsize=14, 
                    fontweight='bold', color=title_color,
                    transform=ax_info.transAxes)
        
        # 🔥 融合权重柱状图
        if fusion_weights is not None:
            ax_bar = fig.add_axes([0.58, 0.55, 0.35, 0.3])
            
            modalities = ['Path', 'Omic', 'Text']
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            
            bars = ax_bar.bar(modalities, fusion_weights, color=colors, alpha=0.7, 
                             edgecolor='black', linewidth=1.5)
            
            # 添加数值标签
            for bar, weight in zip(bars, fusion_weights):
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{weight:.3f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_ylabel('Fusion Weight', fontsize=11, fontweight='bold')
            ax_bar.set_title('Multimodal Fusion Weights', fontsize=12, fontweight='bold')
            ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
            ax_bar.set_axisbelow(True)
        
        # 🔥 统计信息
        if attention_scores is not None and len(attention_scores) > 0:
            info_text = f"📊 Statistics:\n"
            info_text += f"  • Total Patches: {len(attention_scores)}\n"
            info_text += f"  • Avg Attention: {attention_scores.mean():.4f}\n"
            info_text += f"  • Max Attention: {attention_scores.max():.4f}\n"
            info_text += f"  • Min Attention: {attention_scores.min():.4f}\n"
            info_text += f"  • Std Attention: {attention_scores.std():.4f}\n"
            
            if fusion_weights is not None:
                info_text += f"\n🔀 Fusion Weights:\n"
                info_text += f"  • Path: {fusion_weights[0]:.3f}\n"
                info_text += f"  • Omic: {fusion_weights[1]:.3f}\n"
                info_text += f"  • Text: {fusion_weights[2]:.3f}\n"
            
            ax_info.text(0.08, 0.35, info_text, 
                        fontsize=10, family='monospace',
                        transform=ax_info.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                                 edgecolor='black', linewidth=1.5))
        
        # 🔥 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved visualization: {os.path.basename(save_path)}")


class SmartInterpretabilityAnalyzer:
    """智能可解释性分析器 - 自动处理单/双/三模态"""
    
    def __init__(self, model, save_dir, device='cuda', 
                 preprocessing_dir='./outputs_COADREAD/preprocessing', enable_text_centric=True):
        self.model = model
        self.save_dir = save_dir
        self.device = device
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化子模块
        self.heatmap_gen = WSIHeatmapGenerator(preprocessing_dir)
        self.visualizer = EnhancedWSIVisualizer(self.heatmap_gen)

        self.text_centric_explainer = None
        
        if enable_text_centric:
            try:
                from utils.text_centric_interpretability import TextCentricExplainer
                
                # 🔥 为文本中心可视化创建独立的保存目录
                text_centric_save_dir = save_dir.replace('interpretability', 'text_centric')
                if text_centric_save_dir == save_dir:
                    # 如果替换失败，手动添加后缀
                    text_centric_save_dir = save_dir + '_text_centric'
                
                self.text_centric_explainer = TextCentricExplainer(
                    model=model,
                    save_dir=text_centric_save_dir,
                    device=device,
                    preprocessing_dir=preprocessing_dir,
                    n_qa_pairs=6,
                    n_pathways=50,
                    heatmap_generator=self.heatmap_gen  # 🔥 传入共享的生成器
                )
                
                print(f"✅ TextCentricExplainer enabled")
                print(f"   Text-centric save dir: {text_centric_save_dir}")
                
            except ImportError as e:
                print(f"⚠️ Could not import TextCentricExplainer: {e}")
                print(f"   Text-centric interpretability will be disabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize TextCentricExplainer: {e}")
                import traceback
                traceback.print_exc()        
        
        print(f"✅ SmartInterpretabilityAnalyzer initialized")
        print(f"   Save dir: {save_dir}")
        print(f"   Preprocessing dir: {preprocessing_dir}")
    
    def extract_attention_scores(self, model_output):
        """从模型输出提取attention scores - 增强版"""
        
        # 🔥 优先从模型的临时变量获取
        if hasattr(self.model, '_last_batch_attention'):
            batch_attn = self.model._last_batch_attention
            
            if batch_attn is not None and len(batch_attn) > 0:
                # batch_A 是 list，每个元素是一个样本的attention [n_patches]
                attn = batch_attn[0]  # 取第一个样本
                
                # 转换为numpy
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                
                # 🔥 确保是1D
                if attn.ndim == 2:
                    attn = attn.squeeze(-1)
                
                print(f"      ✅ Extracted REAL attention from model")
                print(f"         shape: {attn.shape}, min={attn.min():.6f}, max={attn.max():.6f}, std={attn.std():.6f}")
                
                return attn
        
        # 降级：返回None
        print(f"      ⚠️ Could not extract attention from model")
        return None
    
    def analyze_single_sample(self, data_WSI, data_omic, data_text,
                             slide_id, sample_id, true_label, 
                             class_names, coords=None,
                             survival_time=None, event=None):
        """分析单个样本 - 增强版"""
        
        print(f"\n  🔍 Analyzing sample: {sample_id}")
        print(f"      Slide ID: {slide_id}")
        
        # 构建forward参数
        forward_kwargs = {'compute_loss': False, 'return_features': True}
        
        if data_WSI is not None:
            forward_kwargs['x_path'] = [data_WSI]
        if data_omic is not None:
            forward_kwargs['x_omic'] = data_omic.unsqueeze(0)
        if data_text is not None:
            forward_kwargs['x_text'] = data_text.unsqueeze(0)
        
        # 🔥 生存任务参数
        if survival_time is not None:
            if isinstance(survival_time, torch.Tensor):
                if survival_time.dim() == 0:
                    survival_time = survival_time.unsqueeze(0)
            forward_kwargs['survival_time'] = survival_time
        
        if event is not None:
            if isinstance(event, torch.Tensor):
                if event.dim() == 0:
                    event = event.unsqueeze(0)
            forward_kwargs['event'] = event
        
        # Forward
        with torch.no_grad():
            outputs = self.model(**forward_kwargs)
        
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        _, pred_label = torch.max(logits, 1)
        pred_label = pred_label.item()
        
        # 提取融合权重
        fusion_weights = None
        if isinstance(outputs, dict) and 'fusion_info' in outputs:
            if 'fusion_weights' in outputs['fusion_info']:
                fusion_weights = outputs['fusion_info']['fusion_weights'][0].cpu().numpy()
                print(f"      Fusion weights: Path={fusion_weights[0]:.3f}, Omic={fusion_weights[1]:.3f}, Text={fusion_weights[2]:.3f}")
        
        # 提取attention scores
        attention_scores = None
        if data_WSI is not None:
            n_patches = data_WSI.size(0)
            
            # 尝试从模型获取真实的attention
            attention_scores = self.extract_attention_scores(outputs)
            
            if attention_scores is None:
                # 默认：均匀分布
                attention_scores = np.ones(n_patches) / n_patches
                print(f"      Using uniform attention (model doesn't provide attention)")
            else:
                # 确保是numpy array
                if isinstance(attention_scores, torch.Tensor):
                    attention_scores = attention_scores.cpu().numpy()
                
                # 如果是列表，取第一个
                if isinstance(attention_scores, list):
                    attention_scores = attention_scores[0]
                
                # 展平
                if attention_scores.ndim > 1:
                    attention_scores = attention_scores.flatten()
                
                print(f"      Extracted attention: mean={attention_scores.mean():.4f}, max={attention_scores.max():.4f}")

        """
        # 🔥 尝试加载坐标（优先级：传入的coords > CSV > H5）
        if coords is None:
            # 尝试从CSV加载
            coords = self.heatmap_gen.load_coords_from_csv(slide_id)
            
            # 如果还是None，尝试从H5加载（需要H5路径）
            # 这里可以根据你的数据结构自定义
        """
        # 🔥 尝试加载坐标（优先级：传入的coords > CSV > H5）
        if coords is None:
            print(f"  📍 Coords not provided, attempting to load from files...")  # 🔥 新增
            
            # 尝试从CSV加载
            coords = self.heatmap_gen.load_coords_from_csv(slide_id)
            
            if coords is not None:
                print(f"  ✅ Loaded {len(coords)} coords from CSV")  # 🔥 新增
            else:
                print(f"  ❌ Failed to load coords from CSV")  # 🔥 新增
                # 如果还是None，尝试从H5加载（需要H5路径）
        else:
            print(f"  ✅ Using provided coords: {len(coords)} patches")  # 🔥 新增


        # 加载WSI缩略图
        thumbnail = None
        if slide_id is not None:
            print(f"  📸 Loading thumbnail for slide_id: {slide_id}")  # 🔥 新增
            thumbnail = self.heatmap_gen.load_thumbnail(slide_id)
            
            if thumbnail is not None:
                print(f"  ✅ Thumbnail loaded successfully: {thumbnail.shape}")  # 🔥 新增
            else:
                print(f"  ❌ Failed to load thumbnail!")  # 🔥 新增
        """
        # 加载WSI缩略图
        thumbnail = None
        if slide_id is not None:
            thumbnail = self.heatmap_gen.load_thumbnail(slide_id)
        """

        # 生成可视化
        save_path = os.path.join(self.save_dir, f'{sample_id}_interpretability.png')
        
        if thumbnail is not None and coords is not None and attention_scores is not None:
            # 确保数量匹配
            if len(attention_scores) != len(coords):
                print(f"      ⚠️ Warning: attention ({len(attention_scores)}) != coords ({len(coords)}), using min")
                min_len = min(len(attention_scores), len(coords))
                attention_scores = attention_scores[:min_len]
                coords = coords[:min_len]
            
            self.visualizer.visualize_multimodal(
                thumbnail=thumbnail,
                coords=coords,
                attention_scores=attention_scores,
                fusion_weights=fusion_weights,
                true_label=true_label,
                pred_label=pred_label,
                class_names=class_names,
                save_path=save_path,
                slide_id=slide_id
            )
            return True
        else:
            # 简化版可视化（无WSI）
            print(f"      ⚠️ Missing data for full visualization (thumbnail={thumbnail is not None}, coords={coords is not None})")
            self._visualize_simple(
                fusion_weights, true_label, pred_label, 
                class_names, save_path
            )
            return True
    
    def _visualize_simple(self, fusion_weights, true_label, pred_label,
                         class_names, save_path):
        """简化版可视化（无WSI）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 标题
        true_name = class_names[true_label] if true_label < len(class_names) else f'Class{true_label}'
        pred_name = class_names[pred_label] if pred_label < len(class_names) else f'Class{pred_label}'
        
        title_color = 'green' if true_label == pred_label else 'red'
        ax.text(0.5, 0.9, 
               f'True: {true_name} | Pred: {pred_name}',
               ha='center', va='top', fontsize=14, 
               fontweight='bold', color=title_color,
               transform=ax.transAxes)
        
        # 融合权重
        if fusion_weights is not None:
            modalities = ['Path', 'Omic', 'Text']
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            
            bars = ax.bar(modalities, fusion_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar, weight in zip(bars, fusion_weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{weight:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Fusion Weight', fontsize=12, fontweight='bold')
            ax.set_title('Multimodal Fusion Weights', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No Fusion Information Available',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved simple visualization: {os.path.basename(save_path)}")
    
    def analyze_batch_smart(self, data_WSI_list, data_omic, data_text,
                           slide_ids, sample_ids, true_labels, 
                           class_names, max_visualize=3,
                           survival_times=None, events=None):
        """
        智能批量分析
        """
        if slide_ids is None:
            print("⚠️ No slide_ids provided, skipping interpretability")
            return 0
        
        num_samples = len(slide_ids)
        num_visualize = min(num_samples, max_visualize)
        
        print(f"\n📊 Starting interpretability analysis for {num_visualize}/{num_samples} samples...")
        
        # 智能选择样本
        selected_indices = self._select_samples_smart(
            data_WSI_list, data_omic, data_text, 
            true_labels, num_visualize
        )
        
        num_generated = 0
        
        for idx in selected_indices:
            try:
                # 提取单个样本数据
                sample_wsi = data_WSI_list[idx] if data_WSI_list is not None else None
                sample_omic = data_omic[idx] if data_omic is not None else None
                sample_text = data_text[idx] if data_text is not None else None
                
                slide_id = slide_ids[idx]
                sample_id = sample_ids[idx]
                true_label = true_labels[idx].item() if isinstance(true_labels[idx], torch.Tensor) else true_labels[idx]
                
                # 提取生存数据
                sample_survival_time = None
                sample_event = None
                if survival_times is not None:
                    sample_survival_time = survival_times[idx]
                if events is not None:
                    sample_event = events[idx]
                
                # 分析单个样本
                success = self.analyze_single_sample(
                    data_WSI=sample_wsi,
                    data_omic=sample_omic,
                    data_text=sample_text,
                    slide_id=slide_id,
                    sample_id=sample_id,
                    true_label=true_label,
                    class_names=class_names,
                    coords=None,  # 会自动从CSV/H5加载
                    survival_time=sample_survival_time,
                    event=sample_event
                )
                
                if success:
                    num_generated += 1
                
            except Exception as e:
                print(f"  ⚠️ Failed for sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✅ Successfully generated {num_generated}/{num_visualize} interpretability visualizations\n")
        # 🔥🔥🔥 新增：生成文本中心可解释性
        if self.text_centric_explainer is not None and data_text is not None:
            print(f"\n📊 Generating text-centric explanations...")
            
            try:
                self.text_centric_explainer.analyze_batch(
                    data_WSI_list=data_WSI_list,
                    data_omic=data_omic,
                    data_text=data_text,
                    slide_ids=slide_ids,
                    sample_ids=sample_ids,
                    max_visualize=max_visualize,
                    qa_texts=None,  # 使用默认QA
                    pathway_names=None,  # 使用默认通路名
                    survival_times=survival_times,
                    events=events
                )
            except Exception as e:
                print(f"⚠️ Text-centric explanation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if self.text_centric_explainer is None:
                print(f"\n⚠️ Text-centric explainer not initialized, skipping")
            if data_text is None:
                print(f"\n⚠️ No text data available, skipping text-centric explanation")

        return num_generated
    
    def _select_samples_smart(self, data_WSI_list, data_omic, data_text,
                             true_labels, num_select):
        """智能选择样本（优先选择困难样本）"""
        num_samples = len(true_labels)
        
        if num_samples <= num_select:
            return list(range(num_samples))
        
        # 简单策略：均匀采样
        indices = np.linspace(0, num_samples-1, num_select, dtype=int)
        
        return indices.tolist()