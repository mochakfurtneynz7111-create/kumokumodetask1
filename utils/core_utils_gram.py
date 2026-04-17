"""
GRAM训练工具 - 集成可解释性功能 🔥
==========================================
✅ classification: 多分类
✅ multi_label: 多标签分类
✅ survival: 生存分析
🔥 interpretability: 训练过程可视化
"""

from utils.core_utils import *
import os
from models.gram_porpoise import GRAMPorpoiseMMF, compute_total_loss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import numpy as np
from utils.loss_func import NLLSurvLoss

# 🔥 在文件顶部添加
try:
    from utils.interpretability_module import (
        WSIHeatmapGenerator,
        EnhancedWSIVisualizer,
        SmartInterpretabilityAnalyzer
    )
    INTERPRETABILITY_AVAILABLE = True
    print("✅ Interpretability module loaded successfully!")
except ImportError as e:
    INTERPRETABILITY_AVAILABLE = False
    print(f"⚠️ Interpretability module not available: {e}")
# ========== 早停和监控器 ==========
class AUCEarlyStopping:
    """基于 AUC 的早停（多标签任务）"""
    def __init__(self, warmup=5, patience=20, stop_epoch=30, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = 0.0

    def __call__(self, epoch, val_auc, model, ckpt_name='checkpoint.pt'):
        score = val_auc
        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'AUCEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_auc, model, ckpt_name):
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_auc_max = val_auc
        
class AccuracyEarlyStopping:
    """⭐ 原有：基于准确率的早停（分类任务）"""
    def __init__(self, warmup=5, patience=20, stop_epoch=30, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.0

    def __call__(self, epoch, val_acc, model, ckpt_name='checkpoint.pt'):
        score = val_acc

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'AccuracyEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:  
            self.best_score = score
            self.save_checkpoint(val_acc, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, ckpt_name):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_acc_max = val_acc

class CIndexEarlyStopping:
    """✅ 新增：基于 C-index 的早停（生存分析任务）"""
    def __init__(self, warmup=5, patience=20, stop_epoch=30, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_cindex_max = 0.0

    def __call__(self, epoch, val_cindex, model, ckpt_name='checkpoint.pt'):
        score = val_cindex

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'CIndexEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        if self.verbose:
            print(f'Validation C-index increased ({self.val_cindex_max:.6f} --> {val_cindex:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_cindex_max = val_cindex


class Monitor_Acc:
    """⭐ 原有：分类任务监控器"""
    def __init__(self):
        self.best_score = None

    def __call__(self, val_acc, model, ckpt_name: str = 'checkpoint.pt'):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)


class Monitor_MultiLabel:
    """✅ 新增：多标签任务监控器"""
    def __init__(self):
        self.best_score = None

    def __call__(self, val_auc, model, ckpt_name: str = 'checkpoint.pt'):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)


class Monitor_CIndex:
    """✅ 新增：生存分析任务监控器"""
    def __init__(self):
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):
        score = val_cindex
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)


# ========== 训练循环 ==========

def parse_batch_data(batch_data, task_type='classification'):
    """
    统一解析batch数据，支持PT模式和H5模式，以及单/双/三模态
    
    Args:
        batch_data: 来自dataloader的batch
        task_type: 'classification', 'multi_label', 'survival'
    
    Returns:
        dict with keys: data_WSI_list, data_omic, data_text, label, 
                       event_time (survival), censor (survival),
                       coords_list, slide_ids
        如果解析失败返回None
    """
    if not batch_data or len(batch_data) == 0:
        return None
    
    result = {
        'data_WSI_list': None,
        'data_omic': None,
        'data_text': None,
        'label': None,
        'coords_list': None,
        'slide_ids': None,
        'event_time': None,
        'censor': None
    }
    
    batch_len = len(batch_data)
    # 🔍 诊断打印（text-only调试用，可之后删除）
    if batch_len not in [6, 7, 8]:
        shapes = []
        for x in batch_data:
            if hasattr(x, 'shape'): shapes.append(str(x.shape))
            elif isinstance(x, list): shapes.append(f'list[{len(x)}]')
            else: shapes.append(type(x).__name__)
        # print(f'  🔍 parse_batch_data: batch_len={batch_len}, shapes={shapes}')
    try:
        if task_type in ['classification', 'multi_label']:
            if batch_len == 2:
                # 🔥 单模态：path 或 omic
                data, result['label'] = batch_data
                
                # 判断是哪种模态
                if isinstance(data, list):
                    # 列表格式 -> WSI
                    result['data_WSI_list'] = data
                    
                elif isinstance(data, torch.Tensor):
                    if data.dim() == 3:
                        # [batch, n_patches, dim] -> WSI (列表格式)
                        result['data_WSI_list'] = [data[i] for i in range(data.size(0))]
                        
                    elif data.dim() == 2:
                        # [batch, feature_dim] 或 [n_patches, feature_dim]
                        # 需要根据第一个维度大小判断
                        if data.size(0) == 1:
                            # [1, feature_dim] - 单样本
                            if data.size(1) > 1000:
                                # 高维特征 -> 可能是WSI的单个patch或omic
                                # 进一步检查：如果label也是[1]，可能是batch_size=1的omic
                                if isinstance(result['label'], torch.Tensor) and result['label'].numel() == 1:
                                    # 单样本，判断为omic
                                    result['data_omic'] = data
                                else:
                                    result['data_WSI_list'] = [data]
                            else:
                                # 低维特征 -> Omic
                                result['data_omic'] = data
                                
                        elif data.size(0) > 1:
                            # [batch>1, feature_dim] 或 [n_patches, feature_dim]
                            # 需要和label的大小对比
                            if isinstance(result['label'], torch.Tensor):
                                if result['label'].numel() == data.size(0):
                                    # label数量 = data第一维 -> 是batch的omic数据
                                    result['data_omic'] = data
                                else:
                                    # label数量 != data第一维 -> 是单样本的多个patches
                                    result['data_WSI_list'] = [data]
                            else:
                                # 无法判断，根据维度大小
                                if data.size(1) < 500:
                                    # 特征维度较小 -> omic
                                    result['data_omic'] = data
                                else:
                                    # 特征维度较大 -> WSI patches
                                    result['data_WSI_list'] = [data]
                    
                    elif data.dim() == 1:
                        # [feature_dim] -> Omic (单样本)
                        result['data_omic'] = data.unsqueeze(0)
                    
                    else:
                        print(f"Warning: Unexpected data dimension: {data.dim()}")
                        return None
                else:
                    print(f"Warning: Unexpected data type: {type(data)}")
                    return None
                    
            elif batch_len == 3:
                # 双模态：pathomic (PT模式)
                result['data_WSI_list'], result['data_omic'], result['label'] = batch_data
                
            elif batch_len == 4:
                # 三模态：pathomictext (PT模式)
                result['data_WSI_list'], result['data_omic'], result['data_text'], result['label'] = batch_data
                
            elif batch_len == 5:
                # H5模式，没有text
                (result['data_WSI_list'], result['data_omic'], result['label'], 
                 result['coords_list'], result['slide_ids']) = batch_data
                 
            elif batch_len == 6:
                # H5模式，有text
                (result['data_WSI_list'], result['data_omic'], result['data_text'], 
                 result['label'], result['coords_list'], result['slide_ids']) = batch_data
                 
            else:
                print(f"Warning: Unexpected batch length {batch_len} for {task_type}")
                return None
                
        elif task_type == 'survival':
            if batch_len == 3:
                # 🔥 可能是单模态 survival (但这种情况很少见)
                # 需要进一步判断
                print(f"Warning: batch_len=3 for survival task is ambiguous")
                return None
                
            elif batch_len == 4:
                # 🔥 单模态 survival: (data, label, event_time, censor)
                # 但这种格式不太常见，通常至少有omic
                print(f"Warning: batch_len=4 might be single modality survival")
                # 尝试解析
                data, result['label'], result['event_time'], result['censor'] = batch_data
                
                # 判断是哪种模态
                if isinstance(data, list):
                    result['data_WSI_list'] = data
                elif isinstance(data, torch.Tensor):
                    if data.dim() >= 2 and data.size(-1) > 1000:
                        result['data_WSI_list'] = [data] if data.dim() == 2 else [data[i] for i in range(data.size(0))]
                    else:
                        result['data_omic'] = data
                        
            elif batch_len == 5:
                # 双模态：pathomic survival (PT模式，没有text)
                (result['data_WSI_list'], result['data_omic'], result['label'], 
                 result['event_time'], result['censor']) = batch_data
                 
            elif batch_len == 6:
                # 三模态：pathomictext survival (PT模式，有text)
                (result['data_WSI_list'], result['data_omic'], result['data_text'], 
                 result['label'], result['event_time'], result['censor']) = batch_data
                 
            elif batch_len == 7:
                # H5模式，没有text
                (result['data_WSI_list'], result['data_omic'], result['label'], 
                 result['event_time'], result['censor'], 
                 result['coords_list'], result['slide_ids']) = batch_data
                 
            elif batch_len == 8:
                # H5模式，有text
                (result['data_WSI_list'], result['data_omic'], result['data_text'], 
                 result['label'], result['event_time'], result['censor'],
                 result['coords_list'], result['slide_ids']) = batch_data
                 
            else:
                print(f"Warning: Unexpected batch length {batch_len} for {task_type}")
                return None
                
        else:
            print(f"Warning: Unknown task type {task_type}")
            return None
        
        # 🔥 最终验证：确保至少有一种模态数据
        has_data = (
            result['data_WSI_list'] is not None or 
            result['data_omic'] is not None or 
            result['data_text'] is not None
        )
        
        if not has_data:
            print(f"Warning: No valid modality data found in batch")
            print(f"  batch_len: {batch_len}")
            print(f"  task_type: {task_type}")
            return None
            
    except Exception as e:
        print(f"Error parsing batch data: {e}")
        print(f"  batch_len: {batch_len}")
        print(f"  task_type: {task_type}")
        import traceback
        traceback.print_exc()
        return None
    
    return result

def train_loop_classification_gram(epoch, model, loader, optimizer, n_classes, 
                                   writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., 
                                   gc=16, contrastive_weight=0.1):
    """⭐ 分类任务的GRAM训练循环 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    valid_batches = 0
    
    print('\n')
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='classification')
        
        if parsed is None:
            print(f'Warning: Could not parse batch {batch_idx}, skipping...')
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        coords_list = parsed['coords_list']
        slide_ids = parsed['slide_ids']
        
        # 🔥 检查数据有效性
        try:
            # 检查至少有一种模态
            has_wsi = data_WSI_list is not None and (
                (isinstance(data_WSI_list, list) and len(data_WSI_list) > 0) or
                (not isinstance(data_WSI_list, list) and data_WSI_list.numel() > 0)
            )
            has_omic = data_omic is not None and data_omic.numel() > 0
            has_text = data_text is not None and data_text.numel() > 0
            
            if not has_wsi and not has_omic and not has_text:
                print(f'Warning: Batch {batch_idx} has no valid data, skipping...')
                continue
            
            # 🔥 只处理存在的模态
            forward_kwargs = {
                'compute_loss': True,
                'labels': label  # 🔥 添加labels用于GRAM融合
            }
            
            # 转移到GPU
            label = label.to(device)
            
            if has_omic:
                data_omic = data_omic.to(device).float()
                forward_kwargs['x_omic'] = data_omic
            
            if has_text:
                data_text = data_text.to(device).float()
                forward_kwargs['x_text'] = data_text
            
            if has_wsi:
                # 处理WSI数据
                if isinstance(data_WSI_list, list):
                    valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                    if len(valid_wsi) == 0:
                        print(f'Warning: Batch {batch_idx} has no valid WSI data, skipping...')
                        continue
                    data_WSI_list = valid_wsi
                elif data_WSI_list.numel() == 0:
                    print(f'Warning: Batch {batch_idx} has empty WSI data, skipping...')
                    continue
                
                # 转换为列表格式
                if not isinstance(data_WSI_list, list):
                    if data_WSI_list.dim() == 3:
                        data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                    else:
                        data_WSI_list = [data_WSI_list.to(device).float()]
                else:
                    data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
                
                if len(data_WSI_list) == 0:
                    print(f'Warning: No valid WSI data after processing batch {batch_idx}, skipping...')
                    continue
                
                forward_kwargs['x_path'] = data_WSI_list
            
        except Exception as e:
            print(f'Warning: Error checking batch {batch_idx}: {e}, skipping...')
            import traceback
            traceback.print_exc()
            continue
        
        # 🔥 标签处理
        if label.dim() > 1 and label.size(-1) == 1:
            label = label.view(-1)
        elif label.dim() > 1 and label.size(0) == 1:
            label = label.view(-1)
        
        # 前向传播
        try:
            outputs = model(**forward_kwargs)
            
            loss, loss_dict = compute_total_loss(
                outputs, label,
                task_type='classification',
                contrastive_weight=contrastive_weight,
                model=model
            )
            
            if reg_fn is not None:
                loss += reg_fn(model) * lambda_reg
            
            loss_value = loss.item()
            train_loss += loss_value
            valid_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f'batch {batch_idx}, loss: {loss_value:.4f}, ' + 
                      ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()]))
            
            loss = loss / gc
            loss.backward()
            
            if (batch_idx + 1) % gc == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        except Exception as e:
            print(f'Warning: Error in forward/backward for batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # 检查是否有有效的batch
    if valid_batches == 0:
        print(f'ERROR: No valid batches in epoch {epoch}! Please check your data.')
        raise RuntimeError(f"No valid training data in epoch {epoch}")
    
    train_loss /= valid_batches
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, valid_batches: {valid_batches}')


def train_loop_multilabel_gram(epoch, model, loader, optimizer, n_labels, 
                               writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., 
                               gc=16, contrastive_weight=0.1):
    """✅ 多标签分类的GRAM训练循环 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    valid_batches = 0
    
    print('\n')
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='multi_label')
        
        if parsed is None:
            print(f'Warning: Could not parse batch {batch_idx}, skipping...')
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        coords_list = parsed['coords_list']
        slide_ids = parsed['slide_ids']
        
        # 检查数据有效性
        try:
            if isinstance(data_WSI_list, list):
                valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                if len(valid_wsi) == 0:
                    print(f'Warning: Batch {batch_idx} has no valid WSI data, skipping...')
                    continue
                data_WSI_list = valid_wsi
            elif data_WSI_list is None or data_WSI_list.numel() == 0:
                print(f'Warning: Batch {batch_idx} has empty WSI data, skipping...')
                continue
            
            if data_omic is None or data_omic.numel() == 0:
                print(f'Warning: Batch {batch_idx} has empty omic data, skipping...')
                continue
            
            if label is None or label.numel() == 0:
                print(f'Warning: Batch {batch_idx} has empty labels, skipping...')
                continue
            
        except Exception as e:
            print(f'Warning: Error checking batch {batch_idx}: {e}, skipping...')
            continue
        
        # 转移到GPU
        try:
            data_omic = data_omic.to(device).float()
            label = label.to(device).float()  # 多标签用float
            
            if data_text is not None:
                data_text = data_text.to(device).float()
            
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            if len(data_WSI_list) == 0:
                print(f'Warning: No valid WSI data after processing batch {batch_idx}, skipping...')
                continue
            
        except Exception as e:
            print(f'Warning: Error moving batch {batch_idx} to device: {e}, skipping...')
            continue
        
        forward_kwargs = {
            'compute_loss': True,
            'x_path': data_WSI_list,
            'x_omic': data_omic
        }
        if data_text is not None:
            forward_kwargs['x_text'] = data_text
        
        try:
            outputs = model(**forward_kwargs)
            
            loss, loss_dict = compute_total_loss(
                outputs, label,
                task_type='multi_label',
                contrastive_weight=contrastive_weight
            )
            
            if reg_fn is not None:
                loss += reg_fn(model) * lambda_reg
            
            loss_value = loss.item()
            train_loss += loss_value
            valid_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f'batch {batch_idx}, loss: {loss_value:.4f}, ' + 
                      ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()]))
            
            loss = loss / gc
            loss.backward()
            
            if (batch_idx + 1) % gc == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        except Exception as e:
            print(f'Warning: Error in forward/backward for batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    if valid_batches == 0:
        print(f'ERROR: No valid batches in epoch {epoch}!')
        raise RuntimeError(f"No valid training data in epoch {epoch}")
    
    train_loss /= valid_batches
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, valid_batches: {valid_batches}')


def train_loop_survival_gram(epoch, model, loader, optimizer, n_classes,
                             writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., 
                             gc=16, contrastive_weight=0.1):
    """✅ 生存分析的GRAM训练循环 - 支持单/双/三模态 + 数值稳定性增强"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    valid_batches = 0
    
    # 🔥 添加梯度统计
    grad_norms = []
    
    print('\n')
    for batch_idx, batch_data in enumerate(loader):
        parsed = parse_batch_data(batch_data, task_type='survival')
        
        if parsed is None:
            print(f'Warning: Could not parse batch {batch_idx}, skipping...')
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        y_disc = parsed['label']
        event_time = parsed['event_time']
        censor = parsed['censor']
        
        # 🔥 检查至少有一种模态数据
        # 先检测 zeros 占位符（text-only 时 wsi/omic 为 zeros(B,1,?) 或 zeros(B,1)）
        def _is_placeholder(t):
            if t is None: return True
            if isinstance(t, torch.Tensor):
                return t.shape[-1] <= 1 and t.numel() > 0
            if isinstance(t, list):
                return all(_is_placeholder(x) for x in t)
            return False

        has_wsi = data_WSI_list is not None and (
            (isinstance(data_WSI_list, list) and len(data_WSI_list) > 0) or
            (not isinstance(data_WSI_list, list) and data_WSI_list.numel() > 0)
        ) and not _is_placeholder(data_WSI_list)

        has_omic = (data_omic is not None and data_omic.numel() > 0
                    and not _is_placeholder(data_omic))

        has_text = data_text is not None and data_text.numel() > 0

        # 如果是占位符，置 None，不传给模型
        if not has_wsi: data_WSI_list = None
        if not has_omic: data_omic = None

        if not has_wsi and not has_omic and not has_text:
            print(f'Warning: Batch {batch_idx} has no valid data, skipping...')
            continue
        
        # 转移到GPU
        try:
            # 🔥 只处理存在的模态
            if has_omic:
                data_omic = data_omic.to(device).float()
                # 🔥 检查输入是否有NaN
                if torch.isnan(data_omic).any():
                    print(f'Warning: NaN in omic data at batch {batch_idx}, skipping...')
                    continue
            
            y_disc = y_disc.to(device)
            event_time = event_time.to(device).float()
            censor = censor.to(device)
            
            # 🔥 检查生存时间和事件
            if torch.isnan(event_time).any() or torch.isnan(censor.float()).any():
                print(f'Warning: NaN in survival data at batch {batch_idx}, skipping...')
                continue
            
            if data_text is not None:
                data_text = data_text.to(device).float()
                if torch.isnan(data_text).any():
                    print(f'Warning: NaN in text data at batch {batch_idx}, skipping...')
                    continue
            
            # 🔥 处理WSI数据（如果存在）
            if has_wsi:
                if not isinstance(data_WSI_list, list):
                    if data_WSI_list.dim() == 3:
                        data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                    else:
                        data_WSI_list = [data_WSI_list.to(device).float()]
                else:
                    # 🔥 修复：先过滤None，再检查numel
                    data_WSI_list = [x.to(device).float() for x in data_WSI_list if x is not None and x.numel() > 0]
                
                # 🔥 检查WSI数据
                valid_wsi = []
                for i, wsi in enumerate(data_WSI_list):
                    if torch.isnan(wsi).any():
                        print(f'Warning: NaN in WSI data at batch {batch_idx}, sample {i}')
                        continue
                    valid_wsi.append(wsi)
                
                if len(valid_wsi) == 0:
                    has_wsi = False
                    print(f'Warning: No valid WSI data after NaN check at batch {batch_idx}, skipping...')
                    continue
                else:
                    data_WSI_list = valid_wsi
            
        except Exception as e:
            print(f'Warning: Error moving batch {batch_idx} to device: {e}, skipping...')
            import traceback
            traceback.print_exc()
            continue
        
        # 🔥 构建forward参数（只包含存在的模态）
        forward_kwargs = {
            'compute_loss': True,
            'survival_time': event_time,
            'event': censor
        }
        
        if has_wsi:
            forward_kwargs['x_path'] = data_WSI_list
        if has_omic:
            forward_kwargs['x_omic'] = data_omic
        if data_text is not None:
            forward_kwargs['x_text'] = data_text
            
        # ← 在这里加这一行：
        # print(f"  🔍 DEBUG forward_kwargs keys: {list(forward_kwargs.keys())}, data_text shape: {data_text.shape if data_text is not None else None}")
        
        # 🔥 前向传播
        try:
            outputs = model(**forward_kwargs)
            
            # 🔥 检查输出
            if isinstance(outputs, dict):
                logits = outputs.get('logits', None)
                if logits is None:
                    print(f'Warning: No logits in outputs at batch {batch_idx}, skipping...')
                    continue
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f'Warning: NaN/Inf in logits at batch {batch_idx}, skipping...')
                    continue
            else:
                logits = outputs
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f'Warning: NaN/Inf in logits at batch {batch_idx}, skipping...')
                    continue
            
            # 🔥 计算损失
            loss, loss_dict = compute_total_loss(
                outputs, 
                labels=y_disc,
                survival_time=event_time,
                event=censor,
                task_type='survival',
                contrastive_weight=contrastive_weight,
                survival_loss_fn=loss_fn,
                model=model  # 🔥 传入model用于访问正则化
            )
            
            # 🔥 检查loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'⚠️ Warning: NaN/Inf loss detected in batch {batch_idx}')
                print(f'   Loss dict: {loss_dict}')
                print(f'   Skipping this batch...')
                continue
            
            # 🔥 添加L1/L2正则化（如果有）
            if reg_fn is not None:
                reg_loss = reg_fn(model) * lambda_reg
                if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                    print(f'Warning: NaN/Inf in reg_loss at batch {batch_idx}, skipping regularization')
                else:
                    loss += reg_loss
            
            loss_value = loss.item()
            
            # 🔥 再次检查loss_value
            if not np.isfinite(loss_value):
                print(f'Warning: Non-finite loss value {loss_value} at batch {batch_idx}, skipping...')
                continue
            
            train_loss += loss_value
            valid_batches += 1
            
            # 打印信息
            if (batch_idx + 1) % 50 == 0:
                print(f'batch {batch_idx}, loss: {loss_value:.4f}, ' + 
                      ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()]))
            
            # 🔥 反向传播前归一化loss
            loss = loss / gc
            
            # 🔥 清零梯度（在backward之前）
            if (batch_idx) % gc == 0:
                optimizer.zero_grad()
            
            loss.backward()
            
            # 🔥🔥🔥 梯度裁剪（关键！）
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(total_norm.item())
            
            # 🔥 检查梯度
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f'Warning: NaN/Inf gradient in {name} at batch {batch_idx}')
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f'Warning: Skipping optimizer step due to NaN/Inf gradients at batch {batch_idx}')
                optimizer.zero_grad()
                continue
            
            # 🔥 优化器步进
            if (batch_idx + 1) % gc == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        except RuntimeError as e:
            print(f'RuntimeError in forward/backward for batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            
            # 🔥 清理梯度
            optimizer.zero_grad()
            
            # 🔥 尝试清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue
            
        except Exception as e:
            print(f'Warning: Error in forward/backward for batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            
            # 🔥 清理梯度
            optimizer.zero_grad()
            
            continue
    
    # 🔥 检查是否有有效的batch
    if valid_batches == 0:
        print(f'ERROR: No valid batches in epoch {epoch}! Please check your data.')
        print(f'   Total batches attempted: {batch_idx + 1}')
        raise RuntimeError(f"No valid training data in epoch {epoch}")
    
    # 🔥 计算平均损失
    train_loss /= valid_batches
    
    # 🔥 打印梯度统计
    if len(grad_norms) > 0:
        avg_grad_norm = np.mean(grad_norms)
        max_grad_norm = np.max(grad_norms)
        print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, valid_batches: {valid_batches}/{batch_idx+1}')
        print(f'   Gradient norm - avg: {avg_grad_norm:.4f}, max: {max_grad_norm:.4f}')
    else:
        print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, valid_batches: {valid_batches}/{batch_idx+1}')
        print(f'   Warning: No gradient norms recorded')
    
    # 🔥 检查训练损失
    if not np.isfinite(train_loss):
        print(f'ERROR: Training loss is not finite: {train_loss}')
        raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
        
# ========== 验证循环（带可解释性）==========

def validate_classification_gram(cur, epoch, model, loader, n_classes, 
                                early_stopping=None, monitor_acc=None, 
                                writer=None, loss_fn=None, reg_fn=None, 
                                lambda_reg=0., results_dir=None, metric_stopping=None,
                                interpretability_analyzer=None, args=None):
    """⭐ 分类任务的验证循环 + 可解释性 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_preds = []
    all_labels = []
    valid_batch_count = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='classification')
        
        if parsed is None:
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        coords_list = parsed['coords_list']
        slide_ids = parsed['slide_ids']
        
        try:
            # 检查至少有一种模态
            has_wsi = data_WSI_list is not None and (
                (isinstance(data_WSI_list, list) and len(data_WSI_list) > 0) or
                (not isinstance(data_WSI_list, list) and data_WSI_list.numel() > 0)
            )
            has_omic = data_omic is not None and data_omic.numel() > 0
            has_text = data_text is not None and data_text.numel() > 0
            
            if not has_wsi and not has_omic and not has_text:
                continue
            
            if label is None or label.numel() == 0:
                continue
            
            # 构建forward参数
            forward_kwargs = {
                'compute_loss': True,
                'labels': label
            }
            
            # 转移到GPU
            label = label.to(device)
            
            if has_omic:
                data_omic = data_omic.to(device).float()
                forward_kwargs['x_omic'] = data_omic
            
            if has_text:
                data_text = data_text.to(device).float()
                forward_kwargs['x_text'] = data_text
            
            if has_wsi:
                if isinstance(data_WSI_list, list):
                    valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                    if len(valid_wsi) == 0:
                        continue
                    data_WSI_list = valid_wsi
                elif data_WSI_list.numel() == 0:
                    continue
                
                if not isinstance(data_WSI_list, list):
                    if data_WSI_list.dim() == 3:
                        data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                    else:
                        data_WSI_list = [data_WSI_list.to(device).float()]
                else:
                    data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
                
                if len(data_WSI_list) == 0:
                    continue
                
                forward_kwargs['x_path'] = data_WSI_list
            
            with torch.no_grad():
                outputs = model(**forward_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if label.dim() > 1 and label.size(-1) == 1:
                label = label.view(-1)
            elif label.dim() > 1 and label.size(0) == 1:
                label = label.view(-1)
            
            if logits.size(0) != label.size(0):
                continue
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            valid_batch_count += 1
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # 🔥 可解释性分析（只在H5模式且slide_ids不为None时）
            if (interpretability_analyzer is not None and 
                batch_idx == 0 and 
                hasattr(args, 'interpretability_freq') and
                epoch % args.interpretability_freq == 0 and
                slide_ids is not None):
                
                try:
                    print(f"\n📊 Generating interpretability (Epoch {epoch})...")
                    
                    class_names = getattr(args, 'class_names', 
                                         [f'Class{i}' for i in range(n_classes)])
                    
                    # 使用智能分析器
                    num_visualized = interpretability_analyzer.analyze_batch_smart(
                        data_WSI_list=data_WSI_list if has_wsi else None,
                        data_omic=data_omic if has_omic else None,
                        data_text=data_text if has_text else None,
                        slide_ids=slide_ids,
                        sample_ids=[f'fold{cur}_e{epoch}_s{i}' for i in range(len(label))],
                        true_labels=label,
                        class_names=class_names,
                        max_visualize=getattr(args, 'max_visualize_per_epoch', 3)
                    )
                    
                    print(f"✅ Interpretability saved!")
                    
                except Exception as e:
                    print(f"⚠️ Interpretability generation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if valid_batch_count == 0:
        print("Warning: No valid batches!")
        return True
    
    val_loss /= valid_batch_count
    
    if len(all_labels) == 0:
        return True
    
    val_acc = accuracy_score(all_labels, all_preds)
    
    print(f'Epoch: {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
    
    if monitor_acc:
        monitor_acc(val_acc, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
    
    if metric_stopping:
        metric_stopping(epoch, val_acc, model, 
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_maxacc_checkpoint.pt"))
        if metric_stopping.early_stop:
            return True
    
    if early_stopping:
        early_stopping(epoch, val_loss, model, 
                      ckpt_name=os.path.join(results_dir, f"s_{cur}_minloss_checkpoint.pt"))
        if early_stopping.early_stop:
            return True
    
    return False


def validate_multilabel_gram(cur, epoch, model, loader, n_labels,
                            early_stopping=None, monitor_multilabel=None,
                            writer=None, loss_fn=None, reg_fn=None,
                            lambda_reg=0., results_dir=None, 
                            metric_stopping=None,  # 🔥 添加这个参数
                            interpretability_analyzer=None, args=None):
    """✅ 多标签分类的验证循环 + 可解释性 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_preds = []
    all_labels = []
    valid_batch_count = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='multi_label')
        
        if parsed is None:
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        coords_list = parsed['coords_list']
        slide_ids = parsed['slide_ids']
        
        try:
            if isinstance(data_WSI_list, list):
                valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                if len(valid_wsi) == 0:
                    continue
                data_WSI_list = valid_wsi
            elif data_WSI_list is None or data_WSI_list.numel() == 0:
                continue
            
            if data_omic is None or data_omic.numel() == 0:
                continue
            
            if label is None or label.numel() == 0:
                continue
            
            data_omic = data_omic.to(device).float()
            label = label.to(device).float()
            
            if data_text is not None:
                data_text = data_text.to(device).float()
            
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            if len(data_WSI_list) == 0:
                continue
            
            forward_kwargs = {
                'compute_loss': True,
                'x_path': data_WSI_list,
                'x_omic': data_omic
            }
            if data_text is not None:
                forward_kwargs['x_text'] = data_text
            
            with torch.no_grad():
                outputs = model(**forward_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if logits.size(0) != label.size(0):
                continue
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            valid_batch_count += 1
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            
            # 🔥 可解释性
            if (interpretability_analyzer is not None and 
                batch_idx == 0 and 
                hasattr(args, 'interpretability_freq') and
                epoch % args.interpretability_freq == 0 and
                slide_ids is not None):
                
                try:
                    print(f"\n📊 Generating interpretability (Epoch {epoch})...")
                    label_names = getattr(args, 'label_names', 
                                         [f'Label{i}' for i in range(n_labels)])
                    
                    num_visualized = interpretability_analyzer.analyze_batch_smart(
                        data_WSI_list=data_WSI_list,
                        data_omic=data_omic,
                        data_text=data_text,
                        slide_ids=slide_ids,
                        sample_ids=[f'fold{cur}_e{epoch}_s{i}' for i in range(len(label))],
                        true_labels=None,  # 多标签不好直接可视化
                        class_names=label_names,
                        max_visualize=2
                    )
                    print(f"✅ Interpretability saved")
                except Exception as e:
                    print(f"⚠️ Interpretability failed: {e}")
                
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            continue
    
    if valid_batch_count == 0 or len(all_labels) == 0:
        print("Warning: No valid batches!")
        return True
    
    val_loss /= valid_batch_count
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    aucs = []
    for i in range(n_labels):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except:
            aucs.append(0.5)
    
    mean_auc = np.mean(aucs)
    
    print(f'Epoch: {epoch}, val_loss: {val_loss:.4f}, mean_auc: {mean_auc:.4f}')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/mean_auc', mean_auc, epoch)
    
    if monitor_multilabel:
        monitor_multilabel(mean_auc, model, 
                          ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
    
    # 🔥 新增：基于 AUC 的早停
    if metric_stopping:
        metric_stopping(epoch, mean_auc, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_maxauc_checkpoint.pt"))
        if metric_stopping.early_stop:
            return True
    
    # 原有的基于 loss 的早停
    if early_stopping:
        early_stopping(epoch, val_loss, model,
                      ckpt_name=os.path.join(results_dir, f"s_{cur}_minloss_checkpoint.pt"))
        if early_stopping.early_stop:
            return True
    
    return False

def validate_survival_gram(cur, epoch, model, loader, n_classes,
                          early_stopping=None, monitor_cindex=None,
                          writer=None, loss_fn=None, reg_fn=None,
                          lambda_reg=0., results_dir=None, metric_stopping=None,
                          interpretability_analyzer=None, args=None):
    """✅ 生存分析的验证循环 + 可解释性 + 融合权重监控 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    valid_batch_count = 0
    
    # 🔥 新增：用于记录融合信息
    fusion_weights_list = []
    volume_list = []
    alignment_score_list = []
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='survival')
        
        if parsed is None:
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        event_time = parsed['event_time']
        censorship = parsed['censor']
        coords_list = parsed['coords_list']
        slide_ids = parsed['slide_ids']
        
        try:
            if isinstance(data_WSI_list, list):
                valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                if len(valid_wsi) == 0:
                    continue
                data_WSI_list = valid_wsi
            elif data_WSI_list is None or data_WSI_list.numel() == 0:
                continue
            
            if data_omic is None or data_omic.numel() == 0:
                continue
            
            data_omic = data_omic.to(device).float()
            label = label.to(device)
            event_time = event_time.to(device).float()
            censorship = censorship.to(device)
            
            if data_text is not None:
                data_text = data_text.to(device).float()
            
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            if len(data_WSI_list) == 0:
                continue
            
            forward_kwargs = {
                'compute_loss': True,
                'x_path': data_WSI_list,
                'x_omic': data_omic,
                'survival_time': event_time,
                'event': censorship
            }
            if data_text is not None:
                forward_kwargs['x_text'] = data_text
            
            with torch.no_grad():
                outputs = model(**forward_kwargs)
                h = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # 🔥 记录融合信息
                if isinstance(outputs, dict) and 'fusion_info' in outputs:
                    info = outputs['fusion_info']
                    
                    if 'fusion_weights' in info:
                        weights = info['fusion_weights'].cpu().numpy()
                        fusion_weights_list.append(weights)
                    
                    if 'volume' in info:
                        vol = info['volume'].cpu().numpy()
                        volume_list.append(vol)
                    
                    if 'alignment_score' in info:
                        align_score = info['alignment_score'].cpu().numpy()
                        alignment_score_list.append(align_score)
            
            if isinstance(loss_fn, NLLSurvLoss):
                hazards = torch.sigmoid(h)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            else:
                risk = h.detach().cpu().numpy()
            
            loss, _ = compute_total_loss(
                outputs, 
                labels=label,
                survival_time=event_time,
                event=censorship,
                task_type='survival',
                contrastive_weight=0.0,
                survival_loss_fn=loss_fn
            )
            
            val_loss += loss.item()
            valid_batch_count += 1
            
            all_risk_scores.extend(risk.tolist() if risk.ndim > 0 else [risk.item()])
            all_censorships.extend(censorship.cpu().numpy().tolist())
            all_event_times.extend(event_time.cpu().numpy().tolist())
            
            # 🔥 可解释性（保持原逻辑）
            # 🔥 在validate_survival_gram函数中
            # 🔥 可解释性（保持原逻辑）
            if (interpretability_analyzer is not None and 
                batch_idx == 0 and 
                hasattr(args, 'interpretability_freq') and
                epoch % args.interpretability_freq == 0 and
                slide_ids is not None):
                
                try:
                    print(f"\n📊 Generating interpretability (Epoch {epoch})...")
                    
                    class_names = [f'Time{i}' for i in range(n_classes)]
                    
                    # 🔥🔥🔥 方案1: 融合权重可解释性（WSI热力图 + 融合信息）
                    print("  📊 Generating fusion weights visualization...")
                    num_visualized = interpretability_analyzer.analyze_batch_smart(
                        data_WSI_list=data_WSI_list,
                        data_omic=data_omic,
                        data_text=data_text,
                        slide_ids=slide_ids,
                        sample_ids=[f'fold{cur}_e{epoch}_s{i}' for i in range(len(label))],
                        true_labels=label,
                        class_names=class_names,
                        max_visualize=getattr(args, 'max_visualize_per_epoch', 2),
                        survival_times=event_time,
                        events=censorship
                    )
                    
                    # 🔥🔥🔥 方案2: Text-centric 可解释性（如果启用且存在）
                    if (interpretability_analyzer.text_centric_explainer is not None and 
                        getattr(args, 'enable_text_centric', False)):
                        
                        print("  📊 Generating text-centric explanation...")
                        
                        # 准备QA和通路名称
                        qa_texts = None
                        pathway_names = None
                        
                        if hasattr(args, 'qa_text_file') and args.qa_text_file and os.path.exists(args.qa_text_file):
                            with open(args.qa_text_file, 'r') as f:
                                qa_texts = [line.strip() for line in f.readlines()][:getattr(args, 'n_qa_pairs', 6)]
                        else:
                            qa_texts = [
                                "What is the histological grade or differentiation of the tumor? ",
                                "What is the extent of tumor-infiltrating lymphocytes (TILs) within or around the tumor? ",
                                "What is the local tumor invasion pattern and which adjacent structures are involved? ",
                                "Is there tumor necrosis present, and if so, what is its estimated extent and pattern? ",
                                "Is there evidence of lymphovascular invasion (LVI) by the tumor? ",
                                "What is the surgical resection margin status? "
                            ]
                        
                        if hasattr(args, 'pathway_names_file') and args.pathway_names_file and os.path.exists(args.pathway_names_file):
                            with open(args.pathway_names_file, 'r') as f:
                                pathway_names = [line.strip() for line in f.readlines()][:getattr(args, 'n_pathways', 50)]
                        else:
                            pathway_names = [f"Pathway_{i+1}" for i in range(getattr(args, 'n_pathways', 50))]
                        
                        # 🔥 使用 interpretability_analyzer 中的 text_centric_explainer
                        interpretability_analyzer.text_centric_explainer.analyze_batch(
                            data_WSI_list=data_WSI_list,
                            data_omic=data_omic,
                            data_text=data_text,
                            slide_ids=slide_ids,
                            sample_ids=[f'fold{cur}_e{epoch}_s{i}' for i in range(len(label))],
                            max_visualize=getattr(args, 'max_visualize_per_epoch', 2),
                            qa_texts=qa_texts,
                            pathway_names=pathway_names,
                            survival_times=event_time,
                            events=censorship
                        )
                    
                    print(f"✅ Interpretability saved")
                    
                except Exception as e:
                    print(f"⚠️ Interpretability failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            continue
    
    if valid_batch_count == 0:
        print("Warning: No valid batches!")
        return True
    
    val_loss /= valid_batch_count
    
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)
    
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), 
        all_event_times, 
        all_risk_scores, 
        tied_tol=1e-08
    )[0]
    
    # 🔥 详细的融合分析
    print(f'\n{"="*70}')
    print(f'📊 Validation Results (Epoch {epoch}, Fold {cur})')
    print(f'{"="*70}')
    print(f'Loss: {val_loss:.4f} | C-Index: {c_index:.4f}')
    
    if len(fusion_weights_list) > 0:
        all_weights = np.vstack(fusion_weights_list)
        
        path_mean = all_weights[:, 0].mean()
        path_std = all_weights[:, 0].std()
        omic_mean = all_weights[:, 1].mean()
        omic_std = all_weights[:, 1].std()
        text_mean = all_weights[:, 2].mean()
        text_std = all_weights[:, 2].std()
        
        print(f'\n--- Fusion Weights ---')
        print(f'  Path:  {path_mean:.3f} ± {path_std:.3f} (min: {all_weights[:, 0].min():.3f}, max: {all_weights[:, 0].max():.3f})')
        print(f'  Omic:  {omic_mean:.3f} ± {omic_std:.3f} (min: {all_weights[:, 1].min():.3f}, max: {all_weights[:, 1].max():.3f})')
        print(f'  Text:  {text_mean:.3f} ± {text_std:.3f} (min: {all_weights[:, 2].min():.3f}, max: {all_weights[:, 2].max():.3f})')
        
        if len(volume_list) > 0:
            volumes = np.concatenate(volume_list)
            print(f'\n--- Volume Statistics ---')
            print(f'  Mean: {volumes.mean():.4f} ± {volumes.std():.4f}')
            print(f'  Range: [{volumes.min():.4f}, {volumes.max():.4f}]')
            print(f'  Median: {np.median(volumes):.4f}')
        
        if len(alignment_score_list) > 0:
            scores = np.concatenate(alignment_score_list)
            print(f'\n--- Alignment Score ---')
            print(f'  Mean: {scores.mean():.4f} ± {scores.std():.4f}')
            print(f'  Range: [{scores.min():.4f}, {scores.max():.4f}]')
        
        # 🔥 诊断分析
        print(f'\n--- Fusion Diagnosis ---')
        if 0.45 <= path_mean <= 0.70:
            print(f'  ✅ Fusion using prior knowledge (Path weight: {path_mean:.3f} in [0.45, 0.70])')
        elif path_mean > 0.80:
            print(f'  ⚠️  WARNING: Fusion collapsed to Path-only! (Path weight: {path_mean:.3f} > 0.80)')
            print(f'      → Possible cause: Omic/Text data quality is poor')
            print(f'      → Suggestion: Increase alpha to 0.8 or check data preprocessing')
        elif path_mean < 0.35:
            print(f'  ⚠️  WARNING: Path weight too low! (Path weight: {path_mean:.3f} < 0.35)')
            print(f'      → Possible cause: Path features are weak')
            print(f'      → Suggestion: Check WSI feature extraction')
        else:
            print(f'  ⚡ Path weight: {path_mean:.3f} (slightly off target [0.45, 0.70])')
        
        # 检查Omic贡献
        if omic_mean > 0.20:
            print(f'  ✅ Omic contributing meaningfully (weight: {omic_mean:.3f} > 0.20)')
        elif omic_mean > 0.10:
            print(f'  ⚡ Omic contribution moderate (weight: {omic_mean:.3f} in [0.10, 0.20])')
        else:
            print(f'  ❌ Omic contribution minimal (weight: {omic_mean:.3f} < 0.10)')
        
        # 检查Volume范围
        if len(volume_list) > 0:
            vol_mean = volumes.mean()
            if 0.2 <= vol_mean <= 0.6:
                print(f'  ✅ Volume in healthy range (mean: {vol_mean:.4f} in [0.2, 0.6])')
            elif vol_mean < 0.2:
                print(f'  ⚠️  Volume too low (mean: {vol_mean:.4f} < 0.2) - over-alignment!')
            else:
                print(f'  ⚠️  Volume too high (mean: {vol_mean:.4f} > 0.6) - under-alignment!')
    
    else:
        print(f'\n⚠️  No fusion information available (single modality or fusion disabled)')
    
    print(f'{"="*70}\n')
    
    # Tensorboard记录
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)
        
        if len(fusion_weights_list) > 0:
            writer.add_scalar('val/fusion_weight_path', path_mean, epoch)
            writer.add_scalar('val/fusion_weight_omic', omic_mean, epoch)
            writer.add_scalar('val/fusion_weight_text', text_mean, epoch)
            
            if len(volume_list) > 0:
                writer.add_scalar('val/volume_mean', volumes.mean(), epoch)
            
            if len(alignment_score_list) > 0:
                writer.add_scalar('val/alignment_score', scores.mean(), epoch)
    
    # Monitor和Early Stopping
    if monitor_cindex:
        monitor_cindex(c_index, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
    
    if metric_stopping:
        metric_stopping(epoch, c_index, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_maxcindex_checkpoint.pt"))
        if metric_stopping.early_stop:
            return True
    
    if early_stopping:
        early_stopping(epoch, val_loss, model,
                      ckpt_name=os.path.join(results_dir, f"s_{cur}_minloss_checkpoint.pt"))
        if early_stopping.early_stop:
            return True
    
    return False

# ========== 总结函数（保持原样）==========

def summary_classification_gram(model, loader, n_classes):
    """分类任务总结 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    sample_idx = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='classification')
        
        if parsed is None:
            print(f'Warning: Could not parse batch {batch_idx} in summary, skipping...')
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        coords_list = parsed['coords_list']
        batch_slide_ids = parsed['slide_ids']
        
        try:
            # 检查至少有一种模态
            has_wsi = data_WSI_list is not None and (
                (isinstance(data_WSI_list, list) and len(data_WSI_list) > 0) or
                (not isinstance(data_WSI_list, list) and data_WSI_list.numel() > 0)
            )
            has_omic = data_omic is not None and data_omic.numel() > 0
            has_text = data_text is not None and data_text.numel() > 0
            
            if not has_wsi and not has_omic and not has_text:
                continue
            
            if label is None or label.numel() == 0:
                continue
            
            # 构建forward参数
            forward_kwargs = {
                'compute_loss': False
            }
            
            # 转移到GPU
            label = label.to(device)
            
            if has_omic:
                data_omic = data_omic.to(device).float()
                forward_kwargs['x_omic'] = data_omic
            
            if has_text:
                data_text = data_text.to(device).float()
                forward_kwargs['x_text'] = data_text
            
            if has_wsi:
                if isinstance(data_WSI_list, list):
                    valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                    if len(valid_wsi) == 0:
                        continue
                    data_WSI_list = valid_wsi
                elif data_WSI_list.numel() == 0:
                    continue
                
                if not isinstance(data_WSI_list, list):
                    if data_WSI_list.dim() == 3:
                        data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                    else:
                        data_WSI_list = [data_WSI_list.to(device).float()]
                else:
                    data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
                
                if len(data_WSI_list) == 0:
                    continue
                
                forward_kwargs['x_path'] = data_WSI_list
            
        except Exception as e:
            print(f'Warning: Error processing batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue
        
        # Forward
        with torch.no_grad():
            outputs = model(**forward_kwargs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            batch_size = logits.size(0)
            preds_np = preds.cpu().numpy()
            labels_np = label.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            for i in range(batch_size):
                pred_val = preds_np[i]
                label_val = labels_np[i]
                prob_val = probs_np[i]
                
                all_preds.append(pred_val)
                all_labels.append(label_val)
                all_probs.append(prob_val)
                
                # 🔥 优先使用H5模式的slide_ids
                if batch_slide_ids is not None and len(batch_slide_ids) > i:
                    slide_id = batch_slide_ids[i]
                elif sample_idx < len(slide_ids):
                    slide_id = slide_ids.iloc[sample_idx]
                else:
                    slide_id = f"sample_{sample_idx}"
                
                patient_results[slide_id] = {
                    'slide_id': slide_id,
                    'Y': int(label_val),
                    'Y_hat': int(pred_val),
                    'p': prob_val
                }
                
                sample_idx += 1
    
    # 检查是否有有效结果
    if len(all_labels) == 0:
        print("⚠️ WARNING: No valid samples in summary!")
        return {}, 0.0, 0.0, 0.0
    
    acc = accuracy_score(all_labels, all_preds)
    
    if n_classes == 2:
        try:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        except:
            auc = 0.0
    else:
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
        except:
            auc = 0.0
    
    try:
        f1 = f1_score(all_labels, all_preds, average='macro')
    except:
        f1 = 0.0
    
    print(f"📊 Summary: {len(all_labels)} samples processed")
    
    return patient_results, acc, auc, f1


def summary_multilabel_gram(model, loader, n_labels):
    """多标签任务总结 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    sample_idx = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='multi_label')
        
        if parsed is None:
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        batch_slide_ids = parsed['slide_ids']
        
        try:
            if isinstance(data_WSI_list, list):
                valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                if len(valid_wsi) == 0:
                    continue
                data_WSI_list = valid_wsi
            elif data_WSI_list is None or data_WSI_list.numel() == 0:
                continue
            
            if data_omic is None or data_omic.numel() == 0:
                continue
            
            if label is None or label.numel() == 0:
                continue
            
            data_omic = data_omic.to(device).float()
            label = label.to(device).float()
            
            if data_text is not None:
                data_text = data_text.to(device).float()
            
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            if len(data_WSI_list) == 0:
                continue
            
        except Exception as e:
            print(f'Warning: Error processing batch {batch_idx}: {e}')
            continue
        
        with torch.no_grad():
            forward_kwargs = {
                'compute_loss': False,
                'x_path': data_WSI_list,
                'x_omic': data_omic
            }
            if data_text is not None:
                forward_kwargs['x_text'] = data_text
            
            outputs = model(**forward_kwargs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            probs = torch.sigmoid(logits)
            
            batch_size = logits.size(0)
            probs_np = probs.cpu().numpy()
            labels_np = label.cpu().numpy()
            
            for i in range(batch_size):
                prob_val = probs_np[i]
                label_val = labels_np[i]
                
                all_preds.append(prob_val)
                all_labels.append(label_val)
                
                if batch_slide_ids is not None and len(batch_slide_ids) > i:
                    slide_id = batch_slide_ids[i]
                elif sample_idx < len(slide_ids):
                    slide_id = slide_ids.iloc[sample_idx]
                else:
                    slide_id = f"sample_{sample_idx}"
                
                patient_results[slide_id] = {
                    'slide_id': slide_id,
                    'Y': label_val,
                    'Y_hat': (prob_val > 0.5).astype(int),
                    'p': prob_val
                }
                
                sample_idx += 1
    
    if len(all_labels) == 0:
        print("⚠️ WARNING: No valid samples in summary!")
        return {}, 0.0, 0.0, [], []
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    aucs = []
    aps = []
    for i in range(n_labels):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
            aps.append(ap)
        except:
            aucs.append(0.5)
            aps.append(0.5)
    
    mean_auc = np.mean(aucs)
    mean_ap = np.mean(aps)
    
    print(f"📊 Summary: {len(all_labels)} samples processed")
    
    return patient_results, mean_auc, mean_ap, aucs, aps


def summary_survival_gram(model, loader, n_classes, args=None, cur=0):
    """生存分析任务总结 - 支持PT和H5模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    sample_idx = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 🔥 使用统一的解析函数
        parsed = parse_batch_data(batch_data, task_type='survival')
        
        if parsed is None:
            continue
        
        data_WSI_list = parsed['data_WSI_list']
        data_omic = parsed['data_omic']
        data_text = parsed['data_text']
        label = parsed['label']
        event_time = parsed['event_time']
        censorship = parsed['censor']
        batch_slide_ids = parsed['slide_ids']
        
        try:
            if isinstance(data_WSI_list, list):
                valid_wsi = [x for x in data_WSI_list if x is not None and x.numel() > 0]
                if len(valid_wsi) == 0:
                    continue
                data_WSI_list = valid_wsi
            elif data_WSI_list is None or data_WSI_list.numel() == 0:
                continue
            
            if data_omic is None or data_omic.numel() == 0:
                continue
            
            data_omic = data_omic.to(device).float()
            label = label.to(device)
            event_time = event_time.to(device).float()
            censorship = censorship.to(device)
            
            if data_text is not None:
                data_text = data_text.to(device).float()
            
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            if len(data_WSI_list) == 0:
                continue
            
        except Exception as e:
            print(f'Warning: Error processing batch {batch_idx}: {e}')
            continue
        
        with torch.no_grad():
            forward_kwargs = {
                'compute_loss': False,
                'x_path': data_WSI_list,
                'x_omic': data_omic,
                'survival_time': event_time,
                'event': censorship
            }
            if data_text is not None:
                forward_kwargs['x_text'] = data_text
            
            outputs = model(**forward_kwargs)
            h = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if h.shape[1] > 1:
                hazards = torch.sigmoid(h)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            else:
                risk = h.detach().cpu().numpy().squeeze()
            
            batch_size = h.size(0)
            risk_scores = risk if risk.ndim == 1 else risk.squeeze()
            censorship_vals = censorship.cpu().numpy()
            event_time_vals = event_time.cpu().numpy()
            
            for i in range(batch_size):
                risk_val = risk_scores[i] if batch_size > 1 else risk_scores.item()
                censorship_val = censorship_vals[i]
                time_val = event_time_vals[i]
                
                all_risk_scores.append(risk_val)
                all_censorships.append(censorship_val)
                all_event_times.append(time_val)
                
                if batch_slide_ids is not None and len(batch_slide_ids) > i:
                    slide_id = batch_slide_ids[i]
                elif sample_idx < len(slide_ids):
                    slide_id = slide_ids.iloc[sample_idx]
                else:
                    slide_id = f"sample_{sample_idx}"
                
                patient_results[slide_id] = {
                    'slide_id': slide_id,
                    'censorship': float(censorship_val),
                    'survival_time': float(time_val),
                    'risk': float(risk_val)
                }
                
                sample_idx += 1
    
    if len(all_risk_scores) == 0:
        print("⚠️ WARNING: No valid samples in summary!")
        return {}, 0.0
    
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)
    
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), 
        all_event_times, 
        all_risk_scores, 
        tied_tol=1e-08
    )[0]
    
    print(f"📊 Summary: {len(all_risk_scores)} samples processed")
    # 新增：保存 KM 所需数据
    import pandas as pd
    
    df_km = pd.DataFrame({
        'risk_score':  all_risk_scores,
        'event_time':  all_event_times,
        'censorship':  all_censorships,  # 你的代码里 1=删失, 0=事件发生
    })
    
    median_risk = df_km['risk_score'].median()
    df_km['risk_group'] = (df_km['risk_score'] >= median_risk).map({True: 'High', False: 'Low'})
    
    save_dir = args.results_dir if args is not None else '.'
    df_km.to_csv(os.path.join(save_dir, f'km_data_fold{cur}.csv'), index=False)
    
    return patient_results, c_index

# ========== 主训练函数（集成可解释性）==========

def train(datasets: tuple, cur: int, args: Namespace):
    """⭐ 主训练函数 - 集成可解释性 🔥"""
    print(f'\nTraining Fold {cur} with GRAM (Task: {args.task_type})!')
    global text_explainer  # 🔥 声明为全局变量

    # 🔥🔥🔥 1. 先创建结果目录和writer
    print('\nInit Results Directory...', end=' ')
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)
    print('Done!')
    
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], 
               os.path.join(args.results_dir, f'splits_{cur}.csv'))
    print('Done!')
    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")

    # 损失函数
    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            from utils.core_utils import CrossEntropySurvLoss
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            from utils.core_utils import NLLSurvLoss
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        else:
            raise NotImplementedError
    elif args.task_type == 'multi_label':
        if args.bag_loss == 'bce':
            if hasattr(args, 'pos_weight') and args.pos_weight is not None:
                pos_weight = torch.tensor([args.pos_weight] * args.n_labels).float()
                if torch.cuda.is_available():
                    pos_weight = pos_weight.cuda()
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
    else:
        if args.bag_loss == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    if args.reg_type == 'omic':
        from utils.core_utils import l1_reg_omic
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        from utils.core_utils import l1_reg_modules
        reg_fn = l1_reg_modules
    else:
        reg_fn = None
    print('Done!')

    # 模型初始化
    print('\nInit GRAM Model...', end=' ')
    
    if args.task_type == 'multi_label':
        output_dim = args.n_labels
    else:
        output_dim = args.n_classes
    
    model_dict = {
        'omic_input_dim': args.omic_input_dim,
        'text_input_dim': getattr(args, 'text_input_dim', 768),
        'path_input_dim': args.path_input_dim,
        'fusion': args.fusion,
        'n_classes': output_dim if args.task_type != 'multi_label' else 4,
        'n_labels': output_dim if args.task_type == 'multi_label' else 5,
        'contra_dim': getattr(args, 'contra_dim', 256),
        'contra_temp': getattr(args, 'contra_temp', 0.07),
        'use_contrastive': getattr(args, 'use_gram_contrastive', False),
        'dropout': 0.25 if args.drop_out else 0.0,
        'task_type': args.task_type,
        'path_hidden_dim': getattr(args, 'path_hidden_dim', 512),
        'path_attention_dim': getattr(args, 'path_attention_dim', 256),
        'encoder_hidden_dim': getattr(args, 'encoder_hidden_dim', 768),
        'distance_type': getattr(args, 'distance_type', 'volume'),
        'contrastive_weight': getattr(args, 'contrastive_weight', 0.1),
        'use_gram_fusion': getattr(args, 'use_gram_fusion', False),
        # 🔥🔥🔥 新增：可解释性参数
        'enable_explainability': getattr(args, 'enable_text_centric', False),
        'n_qa_pairs': getattr(args, 'n_qa_pairs', 6),
        'n_pathways': getattr(args, 'n_pathways', 50),
        'pathway_gene_mapping': None,  # 可以后续添加
    }
    
    model = GRAMPorpoiseMMF(**model_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    
    # 🔥🔥🔥 2. 初始化 TensorBoard Writer（在模型之后）
    print('\nInit TensorBoard Writer...', end=' ')
    if args.log_data:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(writer_dir, flush_secs=15)
            print('Done!')
        except ImportError:
            print('TensorboardX not available, logging disabled')
            writer = None
    else:
        writer = None
        print('Disabled')
    
    # 🔥🔥🔥 3. 初始化可解释性（在模型和writer之后）
    # 🔥 现在初始化可解释性分析器（模型已创建）
    interpretability_analyzer = None
    if INTERPRETABILITY_AVAILABLE and getattr(args, 'enable_interpretability', False):
        try:
            print(f"\n{'='*70}")
            print(f"🔍 Initializing interpretability modules...")
            print(f"{'='*70}")
            
            interpretability_save_dir = os.path.join(
                args.results_dir, 
                f'interpretability_fold{cur}'
            )
            
            # 🔥 创建 SmartInterpretabilityAnalyzer
            # 它会自动初始化 WSIHeatmapGenerator 和 TextCentricExplainer
            interpretability_analyzer = SmartInterpretabilityAnalyzer(
                model=model,
                save_dir=interpretability_save_dir,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                preprocessing_dir=getattr(args, 'preprocessing_dir', './outputs_LUAD/preprocessing'),
                enable_text_centric=getattr(args, 'enable_text_centric', True)  # 🔥 可选参数
            )
            
            print(f"\n✅ Interpretability successfully initialized!")
            print(f"   Main save dir: {interpretability_save_dir}")
            print(f"   Text-centric enabled: {interpretability_analyzer.text_centric_explainer is not None}")
            print(f"   Frequency: every {getattr(args, 'interpretability_freq', 5)} epochs")
            print(f"   Max visualize: {getattr(args, 'max_visualize_per_epoch', 2)} samples")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n⚠️ Failed to initialize interpretability: {e}")
            import traceback
            traceback.print_exc()
            interpretability_analyzer = None
    
    else:
        print(f"\n⚠️ Interpretability disabled")
        print(f"   enable_interpretability: {getattr(args, 'enable_interpretability', False)}")
        print(f"   module available: {INTERPRETABILITY_AVAILABLE}\n")
    # 优化器
    print('\nInit optimizer...', end=' ')
    from utils.core_utils import get_optim
    optimizer = get_optim(model, args)
    print('Done!')

    # 数据加载器
    print('\nInit Loaders...', end=' ')
    from utils.core_utils import get_split_loader
    
    if args.task_type == 'multi_label':
        train_weighted = False
    else:
        train_weighted = args.weighted_sample

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, 
        weighted=train_weighted, mode=args.mode, batch_size=args.batch_size, task_type=args.task_type)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, 
        batch_size=args.batch_size, task_type=args.task_type)
    
    print('Done!')

    # 早停和监控器
    print('\nSetup EarlyStopping...', end=' ')
    
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=30, verbose=True)
    else:
        early_stopping = None
    
    if args.task_type == 'survival':
        monitor = Monitor_CIndex()
        if args.metric_early_stopping:
            metric_stopping = CIndexEarlyStopping(warmup=5, patience=20, stop_epoch=30, verbose=True)
        else:
            metric_stopping = None
    elif args.task_type == 'multi_label':
        monitor = Monitor_MultiLabel()
        if args.metric_early_stopping:
            metric_stopping = AUCEarlyStopping(warmup=5, patience=20, stop_epoch=30, verbose=True)
        else:
            metric_stopping = None
    else:
        monitor = Monitor_Acc()
        if args.metric_early_stopping:
            metric_stopping = AccuracyEarlyStopping(warmup=5, patience=20, stop_epoch=30, verbose=True)
        else:
            metric_stopping = None
    
    print('Done!')

    # 🔥 训练循环（带可解释性）
    contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    
    for epoch in range(args.max_epochs):
        if args.task_type == 'classification':
            train_loop_classification_gram(epoch, model, train_loader, optimizer, args.n_classes, 
                                         writer, loss_fn, reg_fn, args.lambda_reg, args.gc, contrastive_weight)
            stop = validate_classification_gram(cur, epoch, model, val_loader, args.n_classes, 
                                              early_stopping, monitor, writer, loss_fn, reg_fn, 
                                              args.lambda_reg, args.results_dir, metric_stopping,
                                              interpretability_analyzer, args)  # 🔥 传入
        elif args.task_type == 'multi_label':
            train_loop_multilabel_gram(epoch, model, train_loader, optimizer, args.n_labels,
                                      writer, loss_fn, reg_fn, args.lambda_reg, args.gc, contrastive_weight)
            stop = validate_multilabel_gram(cur, epoch, model, val_loader, args.n_labels,
                                          early_stopping, monitor, writer, loss_fn, reg_fn,
                                          args.lambda_reg, args.results_dir, 
                                          metric_stopping,  # 🔥 添加这个参数
                                          interpretability_analyzer, args)
        elif args.task_type == 'survival':
            train_loop_survival_gram(epoch, model, train_loader, optimizer, args.n_classes,
                                    writer, loss_fn, reg_fn, args.lambda_reg, args.gc, contrastive_weight)
            stop = validate_survival_gram(cur, epoch, model, val_loader, args.n_classes,
                                        early_stopping, monitor, writer, loss_fn, reg_fn,
                                        args.lambda_reg, args.results_dir, metric_stopping,
                                        interpretability_analyzer, args)  # 🔥 传入
        
        if stop:
            print("Early stopping triggered!")
            break

    # 关闭 writer
    if writer:
        writer.close()
        
    # 最终评估
    print('\nLoading best model for final evaluation...')
    
    if args.task_type == 'survival':
        if args.metric_early_stopping:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_maxcindex_checkpoint.pt")
        else:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    elif args.task_type == 'classification':
        if args.metric_early_stopping:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_maxacc_checkpoint.pt")
        else:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    if args.task_type == 'classification':
        results_dict, acc, auc, f1 = summary_classification_gram(model, val_loader, args.n_classes)
        print(f'Val Acc: {acc:.4f}, Val AUC: {auc:.4f}, Val F1: {f1:.4f}')
        if writer:
            writer.close()
        return results_dict, acc, auc, f1
        
    # 🔥 多标签任务：加载最佳模型
    elif args.task_type == 'multi_label':
        if args.metric_early_stopping:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_maxauc_checkpoint.pt")
        else:
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        
        print(f'Loading best model from: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path))
        
        results_dict, mean_auc, mean_ap, aucs, aps = summary_multilabel_gram(model, val_loader, args.n_labels)
        print(f'Val Mean AUC: {mean_auc:.4f}, Val Mean AP: {mean_ap:.4f}')
        if writer:
            writer.close()
        return results_dict, mean_auc, mean_ap, aucs, aps
        
    elif args.task_type == 'survival':
        results_dict, c_index = summary_survival_gram(model, val_loader, args.n_classes, args=args, cur=cur)
        print(f'Val C-Index: {c_index:.4f}')
        if writer:
            writer.close()
        return results_dict, c_index
