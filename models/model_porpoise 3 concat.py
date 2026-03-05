"""
model_porpoise旧版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable

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


class PorpoiseMMF(nn.Module):
    def __init__(self, 
        omic_input_dim,
        text_input_dim=768,
        path_input_dim=1024,
        fusion='concat', 
        dropout=0.25,
        n_classes=4,
        n_labels=5,     # 修改参数名，与core_utils保持一致
        scale_dim1=8, 
        scale_dim2=8, 
        scale_dim3=8,
        gate_path=1, 
        gate_omic=1, 
        gate_text=1,
        skip=True, 
        dropinput=0.10,
        use_mlp=False,
        size_arg="small",
        task_mode='classification',
        ):
        super(PorpoiseMMF, self).__init__()
        self.fusion = fusion
        self.task_mode = task_mode
        self.n_classes = n_classes
        self.n_labels = n_labels  # 修改属性名
        
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.text_input_dim = text_input_dim

        ### 图像模态（Deep Sets Attention）
        size = self.size_dict_path[size_arg]
        fc = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### 分子模态
        Block = MLP_Block if use_mlp else SNN_Block
        hidden = self.size_dict_omic['small']
        fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)

        ### 文本模态（映射到256维）
        self.fc_text = nn.Sequential(
            nn.Linear(text_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        ### 融合层
        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(256 * 3, size[2]),
                nn.ReLU(),
                nn.Linear(size[2], size[2]),
                nn.ReLU()
            )
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path,
                                     scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256)
        elif self.fusion == 'gram':
            self.mm = GramFusion(dim1=256, dim2=256, dim3=256, 
                                scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3,
                                gate1=gate_path, gate2=gate_omic, gate3=gate_text, 
                                skip=skip, mmhid=256)
        elif self.fusion == 'lrb':
            self.mm = LRBilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1,
                                       gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")

        ### 根据任务类型初始化对应的输出头
        if self.task_mode == 'survival':
            # 生存分析任务 - 输出风险评分
            self.classifier = nn.Sequential(
                nn.Linear(size[2], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)  # 单个风险评分
            )
        elif self.task_mode == 'classification':
            # 多分类任务
            self.classifier = nn.Linear(size[2], n_classes)
        elif self.task_mode == 'multi_label':
            # 多标签分类任务（如五基因突变预测）
            self.classifier = nn.Sequential(
                nn.Linear(size[2], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_labels)  # 每个标签一个输出
            )
        else:
            raise ValueError(f"Unsupported task mode: {self.task_mode}")

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            self.attention_net = nn.DataParallel(self.attention_net).to('cuda:0')
        self.fc_omic = self.fc_omic.to(device)
        self.fc_text = self.fc_text.to(device)
        self.rho = self.rho.to(device)
        self.mm = self.mm.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        # 提取多模态特征（共享部分）
        # 图像路径模态
        x_path = kwargs['x_path']
        print("x_path:",x_path.shape)
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)  # → [B, 256]

        # 分子模态
        x_omic = kwargs['x_omic']
        print("x_omic:",x_omic.shape)
        h_omic = self.fc_omic(x_omic)  # → [B, 256]

        # 文本模态
        x_text = kwargs['x_text']
        print("x_text:",x_text.shape)
        x_text = x_text.unsqueeze(0)  # [768] → [1, 768]
        h_text = self.fc_text(x_text)  # → [B, 256]

        # 多模态融合
        if self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic, h_text], dim=1))  # → [B, 256]
        elif self.fusion == 'gram':
            h_mm = self.mm(h_path, h_omic, h_text)  # → [B, 256]
        elif self.fusion in ['bilinear', 'lrb']:
            h_mm = self.mm(h_path, h_omic)  # Text暂未加入 bilinear 支持
        else:
            raise NotImplementedError

        # 根据任务类型输出相应结果
        logits = self.classifier(h_mm)
        
        if self.task_mode == 'survival':
            # 生存分析 - 返回风险评分
            return logits  # [B, 1]
        elif self.task_mode == 'classification':
            # 多分类 - 返回类别logits
            assert logits.shape[1] == self.n_classes
            return logits  # [B, n_classes]
        elif self.task_mode == 'multi_label':
            # 多标签分类 - 返回每个标签的logits
            assert logits.shape[1] == self.n_labels
            return logits  # [B, n_labels]

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
        x_text = x_text.unsqueeze(0)
        h_text = self.fc_text(x_text)

        # 多模态融合
        if self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic, h_text], dim=1))
        elif self.fusion == 'gram':
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
        elif self.fusion == 'gram' and len(features) == 3:
            MM = self.mm(features[0], features[1], features[2])
        else:
            MM = self.mm(M, O)

        if self.task_mode == 'survival':
            # 生存分析的风险计算
            logits = self.classifier(MM)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S, dim=1)
            return risk
        else:
            # 其他任务直接返回logits
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