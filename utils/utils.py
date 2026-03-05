import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    print("Batch items:", batch)
    print("Labels:", [item[1] for item in batch])
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_MIL(batch):
    # 假设不同模态下标签的位置可能不同，需统一处理
    if len(batch[0]) == 3:  # 例如 mode='pathomic' 或 'omic'
        inputs = [item[0] for item in batch]  # 病理特征或占位符
        other_features = [item[1] for item in batch]  # 基因组特征
        labels = torch.LongTensor([item[2].item() for item in batch])  # 标签在 item[2]
        return torch.cat(inputs, dim=0), torch.cat(other_features, dim=0), labels
    elif len(batch[0]) == 4:  # 例如 mode='pathomictext'
        inputs = [item[0] for item in batch]
        other_features = [item[1] for item in batch]
        text_features = [item[2] for item in batch]
        labels = torch.LongTensor([item[3].item() for item in batch])  # 标签在 item[3]
        return torch.cat(inputs, dim=0), torch.cat(other_features, dim=0), torch.cat(text_features, dim=0), labels
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")
       
def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


"""
"""
# 在文件末尾添加以下函数
                
def collate_MIL_classification_h5(batch, mode='pathomic'):
    """
    H5模式的collate函数 - 处理带坐标的数据
    返回格式: [img_list, omic_batch, text_batch, label_batch, coords_list, slide_ids]
    """
    if not batch:
        print("Warning: Empty batch in collate_MIL_classification_h5")
        if mode == 'pathomictext':
            return [[], torch.empty(0, 2013), torch.empty(0, 768), 
                   torch.empty(0, dtype=torch.long), [], []]
        else:
            return [[], torch.empty(0, 2013), 
                   torch.empty(0, dtype=torch.long), [], []]
    
    # 🔥 检查是否是字典格式（H5模式）
    is_h5_format = isinstance(batch[0], dict)
    
    if is_h5_format:
        # 🔥 H5格式：从字典中提取数据
        img_list = []       # 保持为列表，每个元素是一个样本的patches
        omic_list = []
        text_list = []
        label_list = []
        coords_list = []    # 🔥 新增：坐标列表
        slide_ids = []      # 🔥 新增：slide ID列表
        
        for i, item in enumerate(batch):
            try:
                # 提取数据
                img_list.append(item['path'])
                omic_list.append(item['omic'].squeeze(0) if item['omic'].dim() > 1 else item['omic'])
                label_list.append(item['label'].item() if hasattr(item['label'], 'item') else int(item['label']))
                
                # 🔥 提取coords和slide_id
                if item['coords'] is not None:
                    coords_list.append(item['coords'])
                else:
                    coords_list.append(None)
                
                slide_ids.append(item['slide_id'])
                
                # 如果有text
                if 'text' in item and item['text'] is not None:
                    text_list.append(item['text'].squeeze(0) if item['text'].dim() > 1 else item['text'])
                    
            except Exception as e:
                print(f"Error processing H5 sample {i}: {e}")
                continue
        
        if not omic_list or not label_list:
            print("Warning: No valid H5 samples after processing")
            if mode == 'pathomictext':
                return [[], torch.empty(0, 2013), torch.empty(0, 768), 
                       torch.empty(0, dtype=torch.long), [], []]
            else:
                return [[], torch.empty(0, 2013), 
                       torch.empty(0, dtype=torch.long), [], []]
        
        try:
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.LongTensor(label_list)
            
            if mode == 'pathomictext' and text_list:
                text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
                return [img_list, omic_batch, text_batch, label_batch, coords_list, slide_ids]
            else:
                return [img_list, omic_batch, label_batch, coords_list, slide_ids]
                
        except Exception as e:
            print(f"Error creating H5 batch tensors: {e}")
            if mode == 'pathomictext':
                return [[], torch.empty(0, 2013), torch.empty(0, 768), 
                       torch.empty(0, dtype=torch.long), [], []]
            else:
                return [[], torch.empty(0, 2013), 
                       torch.empty(0, dtype=torch.long), [], []]
    
    else:
        # 🔥 PT格式：使用原有逻辑（tuple格式）
        img = []
        omic = []
        text = []
        label = []
        
        for item in batch:
            img.append(item[0])
            omic.append(item[1])
            if mode == 'pathomictext':
                text.append(item[2])
                label.append(int(item[3]))
            else:
                label.append(int(item[2]))
        
        img = torch.cat(img, dim=0)
        omic = torch.cat(omic, dim=0).type(torch.FloatTensor)
        label = torch.LongTensor(label)
        
        if mode == 'pathomictext':
            text = torch.cat(text, dim=0).type(torch.FloatTensor)
            return [img, omic, text, label]
        else:
            return [img, omic, label]


def collate_MIL_survival_h5(batch, mode='pathomic'):
    """
    H5模式的生存分析collate函数
    """
    if not batch:
        return []
    
    # 检查是否是H5格式
    is_h5_format = isinstance(batch[0], dict)
    
    if is_h5_format:
        # H5格式
        img_list = []
        omic_list = []
        text_list = []
        label_list = []
        event_time_list = []
        censor_list = []
        coords_list = []
        slide_ids = []
        
        for i, item in enumerate(batch):
            try:
                img_list.append(item['path'])  # 🔥 保持为列表
                omic_list.append(item['omic'].squeeze(0) if item['omic'].dim() > 1 else item['omic'])
                label_list.append(int(item['label']))
                event_time_list.append(item['event_time'])
                censor_list.append(item['censorship'])
                
                if item['coords'] is not None:
                    coords_list.append(item['coords'])
                else:
                    coords_list.append(None)
                
                slide_ids.append(item['slide_id'])
                
                if 'text' in item and item['text'] is not None:
                    text_list.append(item['text'].squeeze(0) if item['text'].dim() > 1 else item['text'])
                    
            except Exception as e:
                print(f"Error processing H5 survival sample {i}: {e}")
                continue
        
        if not omic_list:
            return []
        
        try:
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.LongTensor(label_list)
            event_time_batch = torch.FloatTensor(event_time_list)
            censor_batch = torch.FloatTensor(censor_list)
            
            if mode == 'pathomictext' and text_list:
                text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
                # 🔥 返回 img_list（列表）
                return [img_list, omic_batch, text_batch, label_batch, 
                       event_time_batch, censor_batch, coords_list, slide_ids]
            else:
                return [img_list, omic_batch, label_batch, 
                       event_time_batch, censor_batch, coords_list, slide_ids]
                
        except Exception as e:
            print(f"Error creating H5 survival batch: {e}")
            return []
    
    else:
        # 🔥 PT格式回退 - 也要保持 img_list 为列表
        img_list = []
        omic_list = []
        text_list = []
        label_list = []
        event_time_list = []
        censor_list = []

        for item in batch:
            img_list.append(item[0])  # 🔥 保持为列表，不要cat
            
            omic_features = item[1]
            if omic_features.dim() > 1:
                omic_features = omic_features.squeeze(0)
            omic_list.append(omic_features)
            
            if mode == 'pathomictext':
                text_features = item[2]
                if text_features.dim() > 1:
                    text_features = text_features.squeeze(0)
                text_list.append(text_features)
                label_list.append(int(item[3]))
                event_time_list.append(item[4])
                censor_list.append(item[5])
            else:
                label_list.append(int(item[2]))
                event_time_list.append(item[3])
                censor_list.append(item[4])

        if not omic_list:
            return []

        try:
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.LongTensor(label_list)
            event_time_batch = torch.FloatTensor(event_time_list)
            censor_batch = torch.FloatTensor(censor_list)

            if mode == 'pathomictext':
                text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
                return [img_list, omic_batch, text_batch, label_batch, event_time_batch, censor_batch]
            else:
                return [img_list, omic_batch, label_batch, event_time_batch, censor_batch]
                
        except Exception as e:
            print(f"Error creating survival batch tensors: {e}")
            return []


def collate_MIL_multilabel_h5(batch, mode='pathomic'):
    """
    H5模式的多标签collate函数
    """
    if not batch:
        return []
    
    is_h5_format = isinstance(batch[0], dict)
    
    if is_h5_format:
        # H5格式
        img_list = []
        omic_list = []
        text_list = []
        label_list = []
        coords_list = []
        slide_ids = []
        
        for i, item in enumerate(batch):
            try:
                img_list.append(item['path'])  # 🔥 保持为列表
                omic_list.append(item['omic'].squeeze(0) if item['omic'].dim() > 1 else item['omic'])
                label_list.append(item['label'])
                
                if item['coords'] is not None:
                    coords_list.append(item['coords'])
                else:
                    coords_list.append(None)
                
                slide_ids.append(item['slide_id'])
                
                if 'text' in item and item['text'] is not None:
                    text_list.append(item['text'].squeeze(0) if item['text'].dim() > 1 else item['text'])
                    
            except Exception as e:
                print(f"Error processing H5 multilabel sample {i}: {e}")
                continue
        
        if not omic_list:
            return []
        
        try:
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.stack(label_list, dim=0)
            
            if mode == 'pathomictext' and text_list:
                text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
                # 🔥 返回 img_list（列表），不是合并后的tensor
                return [img_list, omic_batch, text_batch, label_batch, coords_list, slide_ids]
            else:
                return [img_list, omic_batch, label_batch, coords_list, slide_ids]
                
        except Exception as e:
            print(f"Error creating H5 multilabel batch: {e}")
            return []
    
    else:
        # 🔥 PT格式回退 - 也要保持 img_list 为列表
        img_list = []
        omic_list = []
        text_list = []
        label_list = []
        
        for item in batch:
            img_list.append(item[0])  # 🔥 保持为列表，不要cat
            omic_features = item[1]
            if omic_features.dim() > 1:
                omic_features = omic_features.squeeze(0)
            omic_list.append(omic_features)
            
            if mode == 'pathomictext':
                text_features = item[2]
                if text_features.dim() > 1:
                    text_features = text_features.squeeze(0)
                text_list.append(text_features)
                label_list.append(item[3])
            else:
                label_list.append(item[2])
        
        omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
        label_batch = torch.stack(label_list, dim=0)
        
        if mode == 'pathomictext':
            text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
            return [img_list, omic_batch, text_batch, label_batch]
        else:
            return [img_list, omic_batch, label_batch]

"""
"""
def collate_MIL_classification(batch, mode='pathomic'):
    """
    统一的分类collate函数 - 支持所有模态
    
    数据格式:
    - mode='path': (wsi_features, label) -> 2个元素
    - mode='omic': (omic_features, label) -> 2个元素  
    - mode='pathomic': (wsi_features, omic_features, label) -> 3个元素
    - mode='pathomictext': (wsi_features, omic_features, text_features, label) -> 4个元素
    """
    
    if not batch:
        print("Warning: Empty batch in collate_MIL_classification")
        return []
    
    # 🔥 根据第一个样本的元素数量判断模式
    batch_len = len(batch[0])
    
    try:
        if batch_len == 2:
            # 🔥 mode='path' 或 mode='omic'
            data_list = [item[0] for item in batch]
            label_list = [item[1] for item in batch]
            
            # 判断是path还是omic
            first_item = data_list[0]
            if first_item.dim() == 2 and first_item.size(0) > 100:
                # ✅ WSI features: [n_patches, feature_dim]
                # 保持为列表，因为每个样本的patch数量不同
                label_batch = torch.cat(label_list, dim=0)
                return [data_list, label_batch]
                
            else:
                # ✅ Omic features: [1, omic_dim] or [omic_dim]
                # Omic特征可以stack，因为维度相同
                omic_list = []
                for item in data_list:
                    if item.dim() > 1:
                        omic_list.append(item.squeeze(0))
                    else:
                        omic_list.append(item)
                
                # ✅ 只对omic特征使用stack（它们维度相同）
                omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
                label_batch = torch.cat(label_list, dim=0)
                return [omic_batch, label_batch]
                   
        elif batch_len == 3:
            # 🔥 mode='pathomic'
            img_list = []
            omic_list = []
            label_list = []
            
            for item in batch:
                img_list.append(item[0])  # 保持为列表
                
                omic_features = item[1]
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                omic_list.append(omic_features)
                
                label_list.append(item[2])
            
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.cat(label_list, dim=0)
            
            return [img_list, omic_batch, label_batch]
        
        elif batch_len == 4:
            # 🔥 mode='pathomictext'
            img_list = []
            omic_list = []
            text_list = []
            label_list = []
            
            for item in batch:
                img_list.append(item[0])  # 保持为列表
                
                omic_features = item[1]
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                omic_list.append(omic_features)
                
                text_features = item[2]
                if text_features.dim() > 1:
                    text_features = text_features.squeeze(0)
                text_list.append(text_features)
                
                label_list.append(item[3])
            
            omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
            text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
            label_batch = torch.cat(label_list, dim=0)
            
            return [img_list, omic_batch, text_batch, label_batch]
        
        else:
            raise ValueError(f"Unexpected batch format with {batch_len} elements")
            
    except Exception as e:
        print(f"Error in collate_MIL_classification: {e}")
        print(f"  Batch length: {batch_len}")
        print(f"  Mode: {mode}")
        import traceback
        traceback.print_exc()
        return []

# 同样修复其他collate函数
def collate_MIL_survival(batch, mode='pathomic'):
    """
    修复后的生存分析collate函数
    """
    if not batch:
        return []
    
    img_list = []
    omic_list = []
    text_list = []
    label_list = []
    event_time_list = []
    censor_list = []

    for i, item in enumerate(batch):
        try:
            img_list.append(item[0])
            
            if mode == 'pathomictext':
                omic_features = item[1]
                text_features = item[2]
                label = item[3]
                event_time = item[4]
                censor = item[5]
                
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                if text_features.dim() > 1:
                    text_features = text_features.squeeze(0)
                
                omic_list.append(omic_features)
                text_list.append(text_features)
            else:
                omic_features = item[1]
                label = item[2]
                event_time = item[3]
                censor = item[4]
                
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                
                omic_list.append(omic_features)

            label_list.append(int(label))
            event_time_list.append(event_time)
            censor_list.append(censor)
            
        except Exception as e:
            print(f"Error processing survival sample {i}: {e}")
            continue

    if not omic_list:
        if mode == 'pathomictext':
            return [[], torch.empty(0, 2013), torch.empty(0, 768), 
                   torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0)]
        else:
            return [[], torch.empty(0, 2013), torch.empty(0, dtype=torch.long), 
                   torch.empty(0), torch.empty(0)]

    try:
        omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
        label_batch = torch.LongTensor(label_list)
        event_time_batch = torch.FloatTensor(event_time_list)
        censor_batch = torch.FloatTensor(censor_list)

        if mode == 'pathomictext':
            text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
            return [img_list, omic_batch, text_batch, label_batch, event_time_batch, censor_batch]
        else:
            return [img_list, omic_batch, label_batch, event_time_batch, censor_batch]
            
    except Exception as e:
        print(f"Error creating survival batch tensors: {e}")
        if mode == 'pathomictext':
            return [[], torch.empty(0, 2013), torch.empty(0, 768), 
                   torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0)]
        else:
            return [[], torch.empty(0, 2013), torch.empty(0, dtype=torch.long), 
                   torch.empty(0), torch.empty(0)]


def collate_MIL_multilabel(batch, mode='pathomic'):
    """
    修复后的多标签collate函数
    """
    if not batch:
        return []
    
    img_list = []
    omic_list = []
    text_list = []
    label_list = []
    
    for i, item in enumerate(batch):
        try:
            img_list.append(item[0])
            
            if mode == 'pathomictext':
                omic_features = item[1]
                text_features = item[2]
                label = item[3]  # 多标签张量
                
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                if text_features.dim() > 1:
                    text_features = text_features.squeeze(0)
                
                omic_list.append(omic_features)
                text_list.append(text_features)
            else:
                omic_features = item[1]
                label = item[2]  # 多标签张量
                
                if omic_features.dim() > 1:
                    omic_features = omic_features.squeeze(0)
                
                omic_list.append(omic_features)

            label_list.append(label)  # 保持为张量
            
        except Exception as e:
            print(f"Error processing multilabel sample {i}: {e}")
            continue

    if not omic_list:
        if mode == 'pathomictext':
            return [[], torch.empty(0, 2013), torch.empty(0, 768), torch.empty(0, 5)]
        else:
            return [[], torch.empty(0, 2013), torch.empty(0, 5)]

    try:
        omic_batch = torch.stack(omic_list, dim=0).type(torch.FloatTensor)
        label_batch = torch.stack(label_list, dim=0)  # 多标签用stack

        if mode == 'pathomictext':
            text_batch = torch.stack(text_list, dim=0).type(torch.FloatTensor)
            return [img_list, omic_batch, text_batch, label_batch]
        else:
            return [img_list, omic_batch, label_batch]
            
    except Exception as e:
        print(f"Error creating multilabel batch tensors: {e}")
        if mode == 'pathomictext':
            return [[], torch.empty(0, 2013), torch.empty(0, 768), torch.empty(0, 5)]
        else:
            return [[], torch.empty(0, 2013), torch.empty(0, 5)]
"""
def collate_MIL_multilabel(batch, mode='pathomic'):
    img = []
    omic = []
    text = []
    label = []
    
    for item in batch:
        img.append(item[0])
        omic.append(item[1])
        if mode == 'pathomictext':
            text.append(item[2])
            label.append(item[3])  # 多标签：保持tensor，不用int()
        else:
            label.append(item[2])   # 多标签：保持tensor，不用int()
    
    img = torch.cat(img, dim=0)
    omic = torch.cat(omic, dim=0).type(torch.FloatTensor)
    
    if mode == 'pathomictext':
        text = torch.cat(text, dim=0).type(torch.FloatTensor)  # 与分类任务相同的处理
        label = torch.stack(label, dim=0)  # 多标签用stack而不是LongTensor
        return [img, omic, text, label]
    else:
        label = torch.stack(label, dim=0)  # 多标签用stack而不是LongTensor
        return [img, omic, label]

def collate_MIL_survival(batch, mode='pathomic'):
    img = []
    omic = []
    text = []
    label = []
    event_time = []
    censor = []

    for item in batch:
        img.append(item[0])
        omic.append(item[1])
        if mode == 'pathomictext':
            text.append(item[2])
            label.append(int(item[3]))
            event_time.append(item[4])
            censor.append(item[5])
        else:
            label.append(int(item[2]))
            event_time.append(item[3])
            censor.append(item[4])

    img = torch.cat(img, dim = 0)
    omic = torch.cat(omic, dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor(label)
    event_time = torch.FloatTensor(event_time)
    censor = torch.FloatTensor(censor)

    if mode == 'pathomictext':
        text = torch.cat(text, dim = 0).type(torch.FloatTensor)
        return [img, omic, text, label, event_time, censor]
    else:
        return [img, omic, label, event_time, censor]

def collate_MIL_classification(batch, mode='pathomic'):
    img = []
    omic = []
    text = []
    label = []
    
    for item in batch:
        img.append(item[0])
        omic.append(item[1])
        if mode == 'pathomictext':
            text.append(item[2])
            label.append(int(item[3]))
        else:
            label.append(int(item[2]))
    
    img = torch.cat(img, dim=0)
    omic = torch.cat(omic, dim=0).type(torch.FloatTensor)
    label = torch.LongTensor(label)
    
    if mode == 'pathomictext':
        text = torch.cat(text, dim=0).type(torch.FloatTensor)
        return [img, omic, text, label]
    else:
        return [img, omic, label]
"""


def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

def collate_MIL_survival_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    event_time = np.array([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]

def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, 
                    mode='coattn', batch_size=1, task_type='survival'):
    """
    获取数据加载器 - 支持H5模式
    """
    # 🔥 检查是否使用H5
    use_h5 = hasattr(split_dataset, 'use_h5') and split_dataset.use_h5
    
    # 🔥 根据任务类型和H5模式选择collate函数
    if use_h5:
        # H5模式：使用H5版本的collate
        if task_type == 'classification':
            collate = lambda batch: collate_MIL_classification_h5(batch, mode=mode)
        elif task_type == 'multi_label':
            collate = lambda batch: collate_MIL_multilabel_h5(batch, mode=mode)
        elif task_type == 'survival':
            if mode == 'coattn':
                collate = collate_MIL_survival_sig  # 特殊模式保持原样
            else:
                collate = lambda batch: collate_MIL_survival_h5(batch, mode=mode)
        else:
            collate = lambda batch: collate_MIL_survival_h5(batch, mode=mode)
    else:
        # PT模式：使用原有collate
        if task_type == 'classification':
            collate = lambda batch: collate_MIL_classification(batch, mode=mode)
        elif mode == 'coattn':
            collate = collate_MIL_survival_sig
        elif mode == 'cluster':
            collate = collate_MIL_survival_cluster
        elif task_type == 'multi_label':
            collate = lambda batch: collate_MIL_multilabel(batch, mode=mode)
        else:
            collate = lambda batch: collate_MIL_survival(batch, mode=mode)
    
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, 
                                  sampler=WeightedRandomSampler(weights, len(weights)), 
                                  collate_fn=collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, 
                                  sampler=RandomSampler(split_dataset), 
                                  collate_fn=collate, **kwargs)
        else:
            val_batch_size = max(1, batch_size) if batch_size > 0 else 1
            loader = DataLoader(split_dataset, batch_size=val_batch_size, 
                              sampler=SequentialSampler(split_dataset), 
                              collate_fn=collate, **kwargs)
    else:
        ids = np.random.choice(np.arange(len(split_dataset)), 
                              int(len(split_dataset)*0.1), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, 
                          sampler=SubsetSequentialSampler(ids), 
                          collate_fn=collate, **kwargs)
    
    return loader


def get_split_loader_before(split_dataset, training=False, testing=False, weighted=False, mode='coattn', batch_size=1, task_type='survival'):
    # 根据任务类型选择合适的 collate 函数
    if task_type == 'classification':
        collate = lambda batch: collate_MIL_classification(batch, mode=mode)
    elif mode == 'coattn':
        collate = collate_MIL_survival_sig
    elif mode == 'cluster':
        collate = collate_MIL_survival_cluster
    elif task_type == 'multi_label':
        collate = lambda batch: collate_MIL_multilabel(batch, mode=mode)
    else:
        collate = lambda batch: collate_MIL_survival(batch, mode=mode)
    
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            # 验证时使用与训练相同的batch_size，或者确保batch_size至少为1
            val_batch_size = max(1, batch_size) if batch_size > 0 else 1
            loader = DataLoader(split_dataset, batch_size=val_batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )
    return loader
    

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    pdb.set_trace()
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

"""
def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)
"""
def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = []
    for c in range(len(dataset.slide_cls_ids)):
        count = len(dataset.slide_cls_ids[c])
        # 添加零样本检查
        if count == 0:
            weight = 0.0  # 设为0避免除零错误
            print(f"Warning: Class {c} has 0 samples! Weight set to 0.")
        else:
            weight = N / count
        weight_per_class.append(weight)
    
    weight_per_sample = []
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight_per_sample.append(weight_per_class[y])
        
    return torch.DoubleTensor(weight_per_sample)
    
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = 0 if T_cont \in (-inf, 0), Y = 1 if T_cont \in [0, a_1),  Y = 2 if T_cont in [a_1, a_2), ..., Y = k if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = 0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=0) = 0
# h(0) = 0 ---> do not need to model
# S(0) = P(Y > 0 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 1,2,...,k
corresponding Y = 1, ..., k. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
# def neg_likelihood_loss(hazards, Y, c):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#   # without padding, S(1) = S[0], h(1) = h[0]
#   S_padded = torch.cat([torch.ones_like(c), S], 1) #S(0) = 1, all patients are alive from (-inf, 0) by definition
#   # after padding, S(0) = S[0], S(1) = S[1], etc, h(1) = h[0]
#   #h[y] = h(1)
#   #S[1] = S(1)
#   neg_l = - c * torch.log(torch.gather(S_padded, 1, Y)) - (1 - c) * (torch.log(torch.gather(S_padded, 1, Y-1)) + torch.log(hazards[:, Y-1]))
#   neg_l = neg_l.mean()
#   return neg_l


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

# def nll_loss(hazards, Y, c, S=None, alpha=0.4, eps=1e-8):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   if S is None:
#       S = 1 - torch.cumsum(hazards, dim=1) # surival is cumulative product of 1 - hazards
#   uncensored_loss = -(1 - c) * (torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#   censored_loss = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps))
#   loss = censored_loss + uncensored_loss
#   loss = loss.mean()
#   return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss_dep(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    #reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg

def l1_reg_omic(model, reg_type=None):
    l1_reg = 0

    if hasattr(model, 'fc_omic'):
        l1_reg += l1_reg_all(model.fc_omic)
    else:
        l1_reg += l1_reg_all(model)

    return l1_reg

def get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)
    """
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'datasets_csv'
    param_code = ''

    ### Model Type
    if args.model_type == 'porpoise_mmf':
      param_code += 'PorpoiseMMF'
    elif args.model_type == 'gram_porpoise_mmf':
        param_code += 'gram_porpoise_mmf'  # 或者设置为 'gram_porpoise_mmf'
    elif args.model_type == 'porpoise_amil':
      param_code += 'PorpoiseAMIL'
    elif args.model_type == 'max_net' or args.model_type == 'snn':
      param_code += 'SNN'
    elif args.model_type == 'amil':
      param_code += 'AMIL'
    elif args.model_type == 'deepset':
      param_code += 'DS'
    elif args.model_type == 'mi_fcn':
      param_code += 'MIFCN'
    elif args.model_type == 'mcat':
      param_code += 'MCAT'
    else:
      raise NotImplementedError

    ### Loss Function
    param_code += '_%s' % args.bag_loss
    if args.bag_loss in ['nll_surv']:
        param_code += '_a%s' % str(args.alpha_surv)

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
      param_code += '_%sreg%s' % (args.reg_type, format(args.lambda_reg, '.0e'))

    if args.dropinput:
      param_code += '_drop%s' % str(int(args.dropinput*100))

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
      param_code += '_gc%s' % str(args.gc)

    ### Applying Which Features
    if args.apply_sigfeats:
      param_code += '_sig'
      dataset_path += '_sig'
    elif args.apply_mutsig:
      param_code += '_mutsig'
      dataset_path += '_mutsig'

    ### Fusion Operation
    if args.fusion != "None":
      param_code += '_' + args.fusion

    ### Updating
    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args