import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import math
from os.path import join
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable

class MediumDimTransformerFusion(nn.Module):
    """
    对齐到中等维度1024的Transformer融合
    """
    def __init__(self, dim1=512, dim2=2013, dim3=768, d_model=1024, dropout_rate=0.25):  # 修改dim2默认值为2013
        super().__init__()
        
        self.d_model = d_model
        
        # 将各模态投影到统一的1024维
        self.path_proj = nn.Linear(dim1, d_model)  # 512 → 1024
        self.omic_proj = nn.Linear(dim2, d_model)  # 动态维度 → 1024  
        self.text_proj = nn.Linear(dim3, d_model)  # 768 → 1024
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 1024
            nhead=8,  
            dim_feedforward=d_model * 2,  # 2048
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出维度: 1024 * 3 = 3072
        self.output_dim = d_model * 3
        
    def forward(self, h_path, h_omic, h_text):
        # 投影到统一维度
        path_emb = self.path_proj(h_path)    # [B, 1024]
        omic_emb = self.omic_proj(h_omic)    # [B, 1024]
        text_emb = self.text_proj(h_text)    # [B, 1024]
        
        # 构建输入序列
        sequence = torch.stack([path_emb, omic_emb, text_emb], dim=1)  # [B, 3, 1024]
        
        # Transformer处理
        attended_sequence = self.transformer(sequence)  # [B, 3, 1024]
        
        # 展平输出
        output = attended_sequence.flatten(start_dim=1)  # [B, 3072]
        
        return output

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.index = -1

    def forward(self, x):
        self.index += 1
        return x + self.pe[self.index:x.size(0)+self.index]


class TransformerFusion(nn.Module):
    """
    修复batch_size=1问题的三模态融合模块
    使用LayerNorm替代BatchNorm以支持任意batch_size
    """
    def __init__(self, 
                 dim1=256,           # path模态维度 
                 dim2=256,           # omic模态维度
                 dim3=256,           # text模态维度
                 d_model=256,        # transformer隐藏维度
                 nhead=8,            # attention头数
                 num_encoder_layers=2,  # encoder层数
                 dropout_rate=0.25,
                 output_dim=768,     # 输出融合特征维度
                 num_targets=1):     # 预测任务数（用于生成融合向量）
        super(TransformerFusion, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_targets = num_targets
        
        # 定义模态配置（模拟ADRD的src_modalities结构）
        self.src_modalities = {
            'path': {'type': 'numerical', 'shape': [dim1]},
            'omic': {'type': 'numerical', 'shape': [dim2]}, 
            'text': {'type': 'numerical', 'shape': [dim3]}
        }
        
        # 模态特定的embedding层（使用LayerNorm替代BatchNorm，支持batch_size=1）
        self.modules_emb_src = torch.nn.ModuleDict()
        for k, info in self.src_modalities.items():
            self.modules_emb_src[k] = torch.nn.Sequential(
                torch.nn.LayerNorm(info['shape'][0]),  # 替换BatchNorm1d为LayerNorm
                torch.nn.Linear(info['shape'][0], d_model),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            )
        
        # 位置编码
        self.pe = PositionalEncoding(d_model)
        
        # 目标任务的辅助embedding（用于生成融合向量的"查询"）
        self.emb_aux = torch.nn.Parameter(
            torch.zeros(num_targets, 1, d_model),
            requires_grad=True,
        )
        
        # Transformer编码器
        enc = torch.nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 2,
            activation='gelu',
            dropout=dropout_rate,
            batch_first=False  # [seq_len, batch, d_model]
        )
        self.transformer = torch.nn.TransformerEncoder(enc, num_encoder_layers)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, output_dim)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'emb_aux' in name:
                nn.init.normal_(param, 0, 0.02)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, h_path, h_omic, h_text, return_attention=False):
        """
        前向传播
        Args:
            h_path: 路径模态特征 [batch_size, dim1] 
            h_omic: 分子模态特征 [batch_size, dim2]
            h_text: 文本模态特征 [batch_size, dim3]
            return_attention: 是否返回注意力权重
        Returns:
            融合特征向量 [batch_size, output_dim]
        """
        batch_size = h_path.shape[0]
        device = h_path.device
        
        # 1. 模态特征编码
        x_dict = {'path': h_path, 'omic': h_omic, 'text': h_text}
        
        # 生成mask（这里假设所有模态都可用）
        mask_dict = {k: torch.ones(batch_size, dtype=torch.bool, device=device) 
                    for k in self.src_modalities.keys()}
        
        # 通过embedding层编码各模态（现在支持batch_size=1了）
        out_emb = {}
        for k in self.modules_emb_src.keys():
            out_emb[k] = self.modules_emb_src[k](x_dict[k])  # [batch_size, d_model]
        
        # 2. 构建Transformer输入序列
        # 目标embedding
        emb_tgt = self.emb_aux.repeat(1, batch_size, 1)  # [num_targets, batch_size, d_model]
        
        # 源模态embedding
        emb_src = torch.stack([out_emb[k] for k in ['path', 'omic', 'text']], dim=0)  # [3, batch_size, d_model]
        
        # 拼接：[目标, 源模态]
        emb_all = torch.cat((emb_tgt, emb_src), dim=0)  # [num_targets+3, batch_size, d_model]
        
        # 3. 添加位置编码
        self.pe.index = -1  # 重置位置编码索引
        emb_all = self.pe(emb_all)
        
        # 4. Transformer处理
        transformer_output = self.transformer(emb_all)  # [num_targets+3, batch_size, d_model]
        
        # 5. 提取融合特征（取第一个目标位置的输出作为融合向量）
        fusion_vector = transformer_output[0]  # [batch_size, d_model]
        
        # 6. 输出投影
        output = self.output_projection(fusion_vector)  # [batch_size, output_dim]
        
        if return_attention:
            # 这里可以添加attention权重提取逻辑
            return output, None
        
        return output


class GramFusion(nn.Module): 
    def __init__(self, skip=0, gate1=1, gate2=1, gate3=1, dim1=256, dim2=256, dim3=256, 
                 scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=256, dropout_rate=0.25):
        super(GramFusion, self).__init__()
        self.skip = skip
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og = dim1, dim2, dim3
        dim1, dim2, dim3 = dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1_og + dim2_og + dim3_og if skip else 0

        # 为每个模态定义gated units
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        
        # 改进1: 提取更丰富的Gram特征
        self.gram_feature_dim = 9  # 3x3 Gram矩阵的上三角+对角线元素
        
        # 改进2: 更复杂的编码器
        self.encoder1 = nn.Sequential(
            nn.Linear(self.gram_feature_dim, 128), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(256 + skip_dim, mmhid), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, mmhid),
            nn.ReLU()
        )
        
        # 改进3: 可学习的特征重要性权重
        self.feature_weights = nn.Parameter(torch.ones(self.gram_feature_dim))

    def extract_gram_features(self, vec1, vec2, vec3):
        """
        提取更丰富的Gram矩阵特征，而不仅仅是行列式
        """
        batch_size = vec1.shape[0]
        
        # 计算Gram矩阵元素
        v1_v1 = torch.einsum('bi,bi->b', vec1, vec1)
        v2_v2 = torch.einsum('bi,bi->b', vec2, vec2)
        v3_v3 = torch.einsum('bi,bi->b', vec3, vec3)
        v1_v2 = torch.einsum('bi,bi->b', vec1, vec2)
        v1_v3 = torch.einsum('bi,bi->b', vec1, vec3)
        v2_v3 = torch.einsum('bi,bi->b', vec2, vec3)
        
        # 构建完整的Gram矩阵
        G = torch.stack([
            torch.stack([v1_v1, v1_v2, v1_v3], dim=-1),
            torch.stack([v1_v2, v2_v2, v2_v3], dim=-1),
            torch.stack([v1_v3, v2_v3, v3_v3], dim=-1)
        ], dim=-2)
        
        # 提取多种特征
        features = []
        
        # 1. 对角线元素（各模态自相关）
        features.extend([v1_v1, v2_v2, v3_v3])
        
        # 2. 非对角线元素（模态间相关）
        features.extend([v1_v2, v1_v3, v2_v3])
        
        # 3. 行列式（体积信息）
        det = torch.det(G.float())
        features.append(torch.abs(det) + 1e-8)
        
        # 4. 迹（总体大小）
        trace = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)  # 对每个batch计算迹
        features.append(trace)
        
        # 5. 最大特征值近似（主要变化方向）
        # 使用迹作为特征值和的近似
        max_eigenval_approx = torch.max(torch.stack([v1_v1, v2_v2, v3_v3], dim=-1), dim=-1)[0]
        features.append(max_eigenval_approx)
        
        # 归一化处理
        gram_features = torch.stack(features, dim=-1)  # [batch_size, 9]
        
        # 数值稳定性处理
        gram_features = torch.log(gram_features + 1e-8)
        
        return gram_features

    def forward(self, vec1, vec2, vec3):
        # Gated处理保持不变
        concat_all = torch.cat((vec1, vec2, vec3), dim=1)
        
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(concat_all)
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(concat_all)
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(concat_all)
            o3 = self.linear_o3(torch.sigmoid(z3) * h3)
        else:
            h3 = self.linear_h3(vec3)
            o3 = self.linear_o3(h3)

        # 提取丰富的Gram特征
        gram_features = self.extract_gram_features(o1, o2, o3)
        
        # 应用可学习权重
        gram_features = gram_features * self.feature_weights
        
        # 通过改进的编码器
        out = self.post_fusion_dropout(gram_features)
        out = self.encoder1(out)
        
        # Skip connection
        if self.skip:
            out = torch.cat((out, vec1, vec2, vec3), 1)
        
        out = self.encoder2(out)
        return out


class LRBilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, 
                 scale_dim1=1, scale_dim2=1, dropout_rate=0.25,
                rank=16, output_dim=4):
        super(LRBilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.rank = rank
        self.output_dim = output_dim

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        
        self.h1_factor = Parameter(torch.Tensor(self.rank, dim1 + 1, output_dim))
        self.h2_factor = Parameter(torch.Tensor(self.rank, dim2 + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        xavier_normal_(self.h1_factor)
        xavier_normal_(self.h2_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = F.dropout(self.linear_h1(vec1), 0.25)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = F.dropout(self.linear_h2(vec2), 0.25)
            o2 = self.linear_o2(h2)

        ### Fusion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _o1 = torch.cat((torch.ones(o1.shape[0], 1).to(device), o1), dim=1)
        _o2 = torch.cat((torch.ones(o2.shape[0], 1).to(device), o2), dim=1)
        o1_fusion = torch.matmul(_o1, self.h1_factor)
        o2_fusion = torch.matmul(_o2, self.h2_factor)
        fusion_zy = o1_fusion * o2_fusion
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1).to(device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1).to(device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))

def MLP_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.Dropout(p=dropout, inplace=False))

class Attn_Net(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        if dropout:
            self.module.append(nn.Dropout(0.25))
        
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# 新增：批处理注意力网络
class BatchedAttn_Net_Gated(nn.Module):
    """支持批处理的注意力网络"""
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_net = Attn_Net_Gated(L, D, dropout, n_classes)
        self._debug_printed = False
    
    def forward(self, x_batch):
        """
        x_batch: List of tensors, each with shape [n_patches, feature_dim]
        Returns: batched attention features [batch_size, L]
        """
        batch_attention_features = []
        
        for i, x in enumerate(x_batch):
            A, h = self.attention_net(x)
            
            if not self._debug_printed and i == 0:
                print(f"Batch Attention: A shape: {A.shape}, h shape: {h.shape}")
                self._debug_printed = True
            
            # 处理注意力权重
            if A.shape[1] == 1:
                A = A.squeeze(1)
            elif A.shape[0] == 1:
                A = A.squeeze(0)
            else:
                A = A[:, 0]
            
            A = F.softmax(A, dim=0)
            attention_feat = torch.einsum('n,nd->d', A, h)
            batch_attention_features.append(attention_feat)
        
        return torch.stack(batch_attention_features, dim=0)


class PorpoiseMMF(nn.Module):
    def __init__(self, omic_input_dim, text_input_dim=768, path_input_dim=1024, 
                 fusion='concat', dropout=0.25, n_classes=4, n_labels=5,
                 scale_dim1=8, scale_dim2=8, scale_dim3=8, 
                 gate_path=1, gate_omic=1, gate_text=1, skip=True, 
                 dropinput=0.10, use_mlp=False, size_arg="small", 
                 task_mode='classification', transformer_d_model=256, 
                 transformer_nhead=8, transformer_layers=2):
        super(PorpoiseMMF, self).__init__()
        self.fusion = fusion
        self.task_mode = task_mode
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.omic_input_dim = omic_input_dim  # 保存实际维度
        self.text_input_dim = text_input_dim
        self._forward_count = 0
        self._batch_mode_enabled = True  # 启用批处理模式

        # 根据fusion模式设置不同的size_dict
        if self.fusion == 'concat':
            self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        else:
            self.size_dict_path = {"small": [path_input_dim, 512, 512], "big": [1024, 512, 512]}
            
        self.size_dict_omic = {'small': [256, 256]}

        ### 图像模态处理 - 修改为支持批处理
        size = self.size_dict_path[size_arg]
        
        # 原始的预处理层
        self.path_fc = nn.Sequential(
            nn.Dropout(dropinput), 
            nn.Linear(size[0], size[1]), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        
        # 原始的单样本attention网络
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        self.attention_net = nn.Sequential(self.path_fc, attention_net)
        
        # 新增：批处理attention网络
        self.batched_attention_net = BatchedAttn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        # 复制权重
        self.batched_attention_net.attention_net.load_state_dict(attention_net.state_dict())
        
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### 分子模态处理
        if self.fusion == 'concat':
            Block = MLP_Block if use_mlp else SNN_Block
            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        else:
            # 其他模式：保持原始维度（使用恒等变换）
            self.fc_omic = nn.Linear(omic_input_dim, omic_input_dim, bias=False)
            with torch.no_grad():
                self.fc_omic.weight.copy_(torch.eye(omic_input_dim))

        ### 文本模态处理
        if self.fusion == 'concat':
            self.fc_text = nn.Sequential(
                nn.Linear(text_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 其他模式：保持原始维度（使用恒等变换）
            self.fc_text = nn.Linear(text_input_dim, text_input_dim, bias=False)
            with torch.no_grad():
                self.fc_text.weight.copy_(torch.eye(text_input_dim))

        ### 融合层 - 修改concatonly以支持动态维度
        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(256 * 3, size[2]),  # size[2]=256
                nn.ReLU(),
                nn.Linear(size[2], size[2]),
                nn.ReLU()
            )
        elif self.fusion == 'concatonly':
            # 使用实际维度动态计算
            total_dim = 512 + omic_input_dim + text_input_dim
            print(f"ConcatOnly fusion: 512 + {omic_input_dim} + {text_input_dim} = {total_dim}")
            
            self.mm = nn.Sequential(
                nn.Linear(total_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.ReLU()
            )
        elif self.fusion == 'transformer_medium':
            self.mm = MediumDimTransformerFusion(
                dim1=512,
                dim2=omic_input_dim,  # 使用实际维度
                dim3=text_input_dim,
                d_model=1024,
                dropout_rate=dropout
            )
        elif self.fusion == 'bilinear':
            if self.fusion == 'concat':
                dim_size = 256
            else:
                dim_size = 512
            self.mm = BilinearFusion(dim1=dim_size, dim2=dim_size, 
                                   scale_dim1=scale_dim1, gate1=gate_path, 
                                   scale_dim2=scale_dim2, gate2=gate_omic, 
                                   skip=skip, mmhid=dim_size)
        elif self.fusion == 'gram':
            if self.fusion == 'concat':
                dim_size = 256
            else:
                dim_size = 512
            self.mm = GramFusion(dim1=dim_size, dim2=dim_size, dim3=dim_size,
                               scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3,
                               gate1=gate_path, gate2=gate_omic, gate3=gate_text,
                               skip=skip, mmhid=dim_size)
        elif self.fusion == 'lrb':
            if self.fusion == 'concat':
                dim_size = 256
            else:
                dim_size = 512
            self.mm = LRBilinearFusion(dim1=dim_size, dim2=dim_size,
                                     scale_dim1=scale_dim1, gate1=gate_path,
                                     scale_dim2=scale_dim2, gate2=gate_omic)
        elif self.fusion == 'transformer':
            if self.fusion == 'concat':
                dim_size = 256
            else:
                dim_size = 512
            self.mm = TransformerFusion(
                dim1=dim_size, dim2=dim_size, dim3=dim_size,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_encoder_layers=transformer_layers,
                dropout_rate=dropout,
                output_dim=dim_size*3,
                num_targets=1
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")

        ### 分类器输入维度设置 - 修改concatonly的维度
        if self.fusion == 'concat':
            classifier_input_dim = size[2]  # 256
        elif self.fusion == 'transformer_medium':
            classifier_input_dim = 1024 * 3  # 3072
        elif self.fusion == 'concatonly':
            classifier_input_dim = 512  # 使用mm输出的维度
        elif self.fusion == 'transformer':
            if self.fusion == 'concat':
                classifier_input_dim = 256 * 3  # 768
            else:
                classifier_input_dim = 512 * 3  # 1536
        else:
            if self.fusion == 'concat':
                classifier_input_dim = size[2]  # 256
            else:
                classifier_input_dim = 512

        ### 根据任务类型初始化对应的输出头
        if self.task_mode == 'survival':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        elif self.task_mode == 'classification':
            self.classifier = nn.Linear(classifier_input_dim, n_classes)
        elif self.task_mode == 'multi_label':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_labels)
            )
        else:
            raise ValueError(f"Unsupported task mode: {self.task_mode}")

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            self.attention_net = nn.DataParallel(self.attention_net).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)
        self.fc_omic = self.fc_omic.to(device)
        self.fc_text = self.fc_text.to(device)
        self.rho = self.rho.to(device)
        self.mm = self.mm.to(device)
        self.classifier = self.classifier.to(device)
        # 新增：移动批处理attention网络
        self.batched_attention_net = self.batched_attention_net.to(device)

    def forward(self, **kwargs):
        """修改后的前向传播，支持批处理"""
        
        self._forward_count += 1
        should_debug = self._forward_count <= 2
        
        x_path = kwargs['x_path']
        x_omic = kwargs['x_omic']
        x_text = kwargs['x_text']
        
        # 检测是否为批处理输入
        is_batch_input = isinstance(x_path, list) and len(x_path) > 1
        
        if is_batch_input and self._batch_mode_enabled:
            if should_debug:
                print(f"PorpoiseMMF: Using batch mode for {len(x_path)} samples")
            return self._forward_batch(x_path, x_omic, x_text, should_debug)
        else:
            if should_debug:
                print("PorpoiseMMF: Using single sample mode")
            return self._forward_single(x_path, x_omic, x_text)

    def _forward_single(self, x_path, x_omic, x_text):
        """原始的单样本处理逻辑"""
        # 如果x_path是列表，取第一个元素
        if isinstance(x_path, list):
            x_path = x_path[0]
        
        # 图像路径模态
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        # 分子模态
        h_omic = self.fc_omic(x_omic)

        # 文本模态
        if x_text.dim() == 1:
            x_text = x_text.unsqueeze(0)
        h_text = self.fc_text(x_text)

        # 多模态融合
        if self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic, h_text], dim=1))
        elif self.fusion == 'concatonly':
            # 对于单样本，h_path已经是pooled的结果，形状应该是[1, 512]
            concatenated = torch.cat([h_path, h_omic, h_text], dim=1)
            h_mm = self.mm(concatenated)
        elif self.fusion == 'transformer_medium':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion == 'gram':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion == 'transformer':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion in ['bilinear', 'lrb']:
            h_mm = self.mm(h_path, h_omic)
        else:
            raise NotImplementedError

        logits = self.classifier(h_mm)
        return logits

    def _forward_batch(self, x_path_batch, x_omic, x_text, should_debug=False):
        """新增的批处理逻辑"""
        
        # 预处理所有path样本
        processed_batch = []
        for x_path in x_path_batch:
            processed = self.path_fc(x_path)  # [n_patches, 512]
            processed_batch.append(processed)
        
        # 批处理attention
        h_path_attention = self.batched_attention_net(processed_batch)  # [batch_size, 512]
        h_path = self.rho(h_path_attention)  # [batch_size, 256/512]
        
        # Omic和Text特征（已经是批处理格式）
        h_omic = self.fc_omic(x_omic)
        
        if x_text.dim() == 1:
            x_text = x_text.unsqueeze(0)
        h_text = self.fc_text(x_text)
        
        if should_debug:
            print(f"Batch features:")
            print(f"   h_path: {h_path.shape}")
            print(f"   h_omic: {h_omic.shape}")
            print(f"   h_text: {h_text.shape}")
        
        # 融合
        if self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic, h_text], dim=1))
        elif self.fusion == 'concatonly':
            # 对于批处理，使用attention输出和原始特征
            concatenated = torch.cat([h_path_attention, x_omic, x_text], dim=1)
            if should_debug:
                print(f"ConcatOnly batch: {concatenated.shape}")
            h_mm = self.mm(concatenated)
        elif self.fusion == 'transformer_medium':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion == 'gram':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion == 'transformer':
            h_mm = self.mm(h_path, h_omic, h_text)
        else:
            # 对于不支持的融合模式，fallback到逐个处理
            return self._forward_batch_fallback(x_path_batch, x_omic, x_text)
        
        if should_debug:
            print(f"Fusion output: {h_mm.shape}")
        
        logits = self.classifier(h_mm)
        return logits

    def _forward_batch_fallback(self, x_path_batch, x_omic, x_text):
        """批处理fallback：逐个处理样本"""
        batch_logits = []
        batch_size = x_omic.size(0)
        
        for i in range(len(x_path_batch)):
            # 确保不超出batch范围
            omic_idx = min(i, batch_size - 1)
            text_idx = min(i, x_text.size(0) - 1)
            
            logits = self._forward_single(
                x_path_batch[i],
                x_omic[omic_idx:omic_idx+1],
                x_text[text_idx:text_idx+1]
            )
            batch_logits.append(logits)
        
        return torch.cat(batch_logits, dim=0)

    def get_shared_features(self, **kwargs):
        """
        获取共享的多模态特征表示，可用于特征提取或迁移学习
        """
        # 图像路径模态
        x_path = kwargs['x_path']
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        # 分子模态
        x_omic = kwargs['x_omic']
        h_omic = self.fc_omic(x_omic)

        # 文本模态
        x_text = kwargs['x_text']
        if x_text.dim() == 1:
            x_text = x_text.unsqueeze(0)
        h_text = self.fc_text(x_text)

        # 多模态融合
        if self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic, h_text], dim=1))
        elif self.fusion == 'concatonly':
            concatenated = torch.cat([h_path, h_omic, h_text], dim=1)
            h_mm = self.mm(concatenated)
        elif self.fusion == 'gram':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion == 'transformer':
            h_mm = self.mm(h_path, h_omic, h_text)
        elif self.fusion in ['bilinear', 'lrb']:
            h_mm = self.mm(h_path, h_omic)
        else:
            raise NotImplementedError

        return h_mm

    def captum(self, h, X, text=None):
        """保持原有的captum方法不变，用于可解释性分析"""
        A, h = self.attention_net(h)
        A = A.squeeze(dim=2)
        A = F.softmax(A, dim=1)
        M = torch.bmm(A.unsqueeze(dim=1), h).squeeze(dim=1)
        M = self.rho(M)
        O = self.fc_omic(X)
        features = [M, O]
        
        if text is not None:
            h_text = self.fc_text(text.view(M.shape[0], -1))
            features.append(h_text)
        
        if self.fusion == 'concat':
            MM = self.mm(torch.cat(features, dim=1))
        elif self.fusion == 'concatonly':
            concatenated = torch.cat(features, dim=1)
            MM = self.mm(concatenated)
        elif self.fusion == 'gram' and len(features) == 3:
            MM = self.mm(features[0], features[1], features[2])
        elif self.fusion == 'transformer' and len(features) == 3:
            MM = self.mm(features[0], features[1], features[2])
        else:
            MM = self.mm(M, O)
        
        if self.task_mode == 'survival':
            logits = self.classifier(MM)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S, dim=1)
            return risk
        else:
            return self.classifier(MM)


# 向后兼容的PorpoiseAMIL
class PorpoiseAMIL(nn.Module):
    def __init__(self, omic_input_dim, model_size_omic="small", n_classes=4, dropout=0.25):
        super(PorpoiseAMIL, self).__init__()
        # 使用PorpoiseMMF，但配置为单模态
        self.porpoise_mmf = PorpoiseMMF(
            omic_input_dim=omic_input_dim,
            text_input_dim=768,
            path_input_dim=1024,
            fusion='concat',
            dropout=dropout,
            n_classes=n_classes,
            n_labels=5,
            task_mode='classification',
            use_mlp=True
        )

    def relocate(self):
        self.porpoise_mmf.relocate()

    def forward(self, **kwargs):
        return self.porpoise_mmf(**kwargs)

    def get_slide_features(self, **kwargs):
        return self.porpoise_mmf.get_shared_features(**kwargs)