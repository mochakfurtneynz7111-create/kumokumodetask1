from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
 
import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth

import warnings

warnings.filterwarnings("ignore")  # 忽略所有警告

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/classification_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_classes = 4, ignore=[],
        patient_strat=False, label_col = 'label', filter_dict = {}, eps=1e-6):
        
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.n_classes = n_classes

        slide_data = pd.read_csv(csv_path, low_memory=False)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        if 'Unnamed: 0' in slide_data.columns:
            slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        # 验证标签列存在
        if not label_col:
            label_col = 'label'
        else:
            assert label_col in slide_data.columns, f"Label column '{label_col}' not found in data"
        self.label_col = label_col

        if "IDC" in slide_data.get('oncotree_code', []): # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 对于分类任务，直接使用标签列的值
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        
        # 确保标签是整数且在合理范围内
        assert slide_data[label_col].min() >= 0, "Labels should be non-negative"
        assert slide_data[label_col].max() < n_classes, f"Labels should be less than n_classes ({n_classes})"
        
        # 直接使用标签值，不需要像生存分析那样进行分箱
        patients_df['label'] = patients_df[label_col].astype(int)

        # 构建患者字典
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        # 为分类任务创建简单的标签字典
        label_dict = {}
        for i in range(n_classes):
            label_dict[i] = i
        
        self.label_dict = label_dict

        # 确保标签正确映射
        for i in slide_data.index:
            label_val = slide_data.loc[i, 'label']
            slide_data.at[i, 'label'] = label_dict[label_val]

        self.num_classes = n_classes
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        # 重新排列列顺序
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        
        # 定义元数据列（根据分类任务调整）
        # 修改metadata定义部分
        metadata = ['Unnamed: 0', 'case_id', 'slide_id', 'site', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship']
        self.metadata = [col for col in metadata if col in slide_data.columns]

        print("self.metadata:", self.metadata)
        
        # 检查基因组特征列
        genomic_cols = slide_data.drop(self.metadata, axis=1).columns
        for col in genomic_cols:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(f"Non-genomic column found: {col}")

        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        """准备每个类别的样本索引"""
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_classes, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        return data

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Classification_Split(df_slice, metadata=self.metadata, mode=self.mode, 
                                               signatures=self.signatures, data_dir=self.data_dir, 
                                               label_col=self.label_col, patient_dict=self.patient_dict, 
                                               num_classes=self.num_classes)
        else:
            split = None
        
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            print(f"csv_path:{csv_path}")

            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            print(f"Train Split Length: {len(train_split)}")
            test_split = None

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Classification_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.LongTensor([self.slide_data['label'][idx]])  # 分类标签使用LongTensor
        slide_ids = self.patient_dict[case_id]
    
        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':  ## 🔥 仅WSI - 修复版
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    # 🔥 修改：只返回2个元素 (path_features, label)
                    return (path_features, label)
    
                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label)
    
                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    # 🔥 修改：只返回2个元素 (omic, label)
                    return (genomic_features.unsqueeze(dim=0), label)
    
                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), label)
    
                elif self.mode == 'pathomictext':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    text_features = torch.tensor(self.text_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), text_features.unsqueeze(dim=0), label)
                elif self.mode == 'pathotext':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    # omic位置用零向量占位，保持和pathomictext相同的4元素格式
                    omic_dim = self.genomic_features.shape[1] if not self.genomic_features.empty else 1
                    dummy_omic = torch.zeros(omic_dim)
                    text_features = torch.tensor(self.text_features.iloc[idx].values, dtype=torch.float32)
                    return (path_features, dummy_omic.unsqueeze(dim=0), text_features.unsqueeze(dim=0), label)
    
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label


# Generic_Classification_Split类 - 完整版本（只有这个类有变化，其他保持不变）
class Generic_Classification_Split(Generic_MIL_Classification_Dataset):
    def __init__(self, slide_data, metadata, mode, 
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=4):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> 改进的特征提取方式
        # 获取所有列名
        all_cols = self.slide_data.columns.tolist()
        
        # 分类不同类型的特征列
        metadata_cols = [col for col in all_cols if not ('_cnv' in col or '_mut' in col or '_rnaseq' in col or col.startswith('embedding_'))]
        cnv_cols = [col for col in all_cols if col.endswith('_cnv')]
        mut_cols = [col for col in all_cols if col.endswith('_mut')]
        rna_cols = [col for col in all_cols if col.endswith('_rnaseq')]
        embedding_cols = [col for col in all_cols if col.startswith('embedding_')]
        
        print(f"Feature distribution:")
        print(f"  CNV features: {len(cnv_cols)}")
        print(f"  MUT features: {len(mut_cols)}")
        print(f"  RNA features: {len(rna_cols)}")
        print(f"  Embedding features: {len(embedding_cols)}")
        print(f"  Metadata columns: {len(metadata_cols)}")
        
        # 组合分子特征 (CNV + MUT + RNA) - 分类任务保留所有特征
        molecular_cols = cnv_cols + mut_cols + rna_cols
        
        # 提取分子特征 (去除标签列如果存在)
        if self.label_col and self.label_col in molecular_cols:
            molecular_cols.remove(self.label_col)
            
        self.genomic_features = self.slide_data[molecular_cols] if molecular_cols else pd.DataFrame()
        
        # 提取文本嵌入特征
        self.text_features = self.slide_data[embedding_cols] if embedding_cols else pd.DataFrame()
        
        # 存储特征列信息，用于调试和分析
        self.feature_info = {
            'cnv_cols': cnv_cols,
            'mut_cols': mut_cols,  # 分类任务保留突变特征
            'rna_cols': rna_cols,
            'embedding_cols': embedding_cols,
            'molecular_cols': molecular_cols,
            'metadata_cols': metadata_cols
        }
        
        print(f"Final genomic features shape: {self.genomic_features.shape}")
        print(f"Final text features shape: {self.text_features.shape}")
            
        self.signatures = signatures
        
        if mode == 'cluster':
            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
                self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler
    def get_scaler(self):
        scalers = {}
        if not self.genomic_features.empty:
            scalers['omic'] = StandardScaler().fit(self.genomic_features)
        if not self.text_features.empty:
            scalers['text'] = StandardScaler().fit(self.text_features)
        return scalers
    
    def apply_scaler(self, scalers: dict=None):
        if scalers is None:
            return
            
        if 'omic' in scalers and not self.genomic_features.empty:
            transformed_omic = pd.DataFrame(scalers['omic'].transform(self.genomic_features))
            transformed_omic.columns = self.genomic_features.columns
            self.genomic_features = transformed_omic
            print("dataset_classification:self.genomic_features:", self.genomic_features.shape)

        if 'text' in scalers and not self.text_features.empty:
            transformed_text = pd.DataFrame(scalers['text'].transform(self.text_features))
            transformed_text.columns = self.text_features.columns
            self.text_features = transformed_text
            print("dataset_classification:self.text_features:", self.text_features.shape)

    def set_split_id(self, split_id):
        self.split_id = split_id
    
    def get_feature_info(self):
        """获取特征分布信息"""
        return self.feature_info
