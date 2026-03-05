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

warnings.filterwarnings("ignore")

class Generic_WSI_MultiLabel_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/multilabel_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_labels = 5, ignore=[],
        patient_strat=False, label_cols = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5'], 
        filter_dict = {}, eps=1e-6):
        
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.n_labels = n_labels
        self.label_cols = label_cols
        
        # 🔥 添加多标签任务所需的属性
        self.num_classes = n_labels  # ✅ 对于多标签，num_classes = n_labels
        self.task_type = 'multi_label'  # ✅ 显式标记任务类型

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

        # 验证标签列存在且数量正确
        assert all(col in slide_data.columns for col in label_cols), f"Some label columns {label_cols} not found in data"
        assert len(label_cols) == n_labels, f"Number of label columns ({len(label_cols)}) must match n_labels ({n_labels})"
        
        # 验证标签值都是0或1（二进制）
        for col in label_cols:
            unique_vals = slide_data[col].unique()
            assert set(unique_vals).issubset({0, 1, 0.0, 1.0}), f"Column {col} contains non-binary values: {unique_vals}"
            slide_data[col] = slide_data[col].astype(int)

        if "IDC" in slide_data.get('oncotree_code', []):
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 对于多标签任务，为每个患者创建多标签向量
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        
        for col in label_cols:
            patients_df[col] = patients_df[col].astype(int)

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

        # 为多标签创建标签字典（每个标签一个字典）
        self.label_dicts = {}
        for i, col in enumerate(label_cols):
            self.label_dicts[col] = {0: 0, 1: 1}
        
        # 创建组合标签统计
        self.label_combinations = {}
        for idx in slide_data.index:
            combo = tuple(slide_data.loc[idx, label_cols].values)
            if combo not in self.label_combinations:
                self.label_combinations[combo] = 0
            self.label_combinations[combo] += 1

        # 患者数据包含所有标签
        patients_df = slide_data.drop_duplicates(['case_id'])
        patient_labels = patients_df[label_cols].values
        self.patient_data = {
            'case_id': patients_df['case_id'].values, 
            'labels': patient_labels
        }

        # 重新排列列顺序
        other_cols = [col for col in slide_data.columns if col not in label_cols]
        new_cols = label_cols + other_cols
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        
        # 定义元数据列
        metadata = ['Unnamed: 0', 'case_id', 'slide_id', 'site', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship']
        self.metadata = [col for col in metadata if col in slide_data.columns]

        print("self.metadata:", self.metadata)
        
        # 检查基因组特征列
        genomic_cols = slide_data.drop(self.metadata + label_cols, axis=1).columns
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
        """为每个标签准备正负样本索引"""
        self.label_cls_ids = {}
        for i, col in enumerate(self.label_cols):
            self.label_cls_ids[col] = {
                'positive': np.where(self.patient_data['labels'][:, i] == 1)[0],
                'negative': np.where(self.patient_data['labels'][:, i] == 0)[0]
            }
        
        self.slide_label_cls_ids = {}
        for i, col in enumerate(self.label_cols):
            self.slide_label_cls_ids[col] = {
                'positive': np.where(self.slide_data[col] == 1)[0],
                'negative': np.where(self.slide_data[col] == 0)[0]
            }

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            labels = self.slide_data.loc[locations[0], self.label_cols].values
            patient_labels.append(labels)
        
        self.patient_data = {
            'case_id': patients, 
            'labels': np.array(patient_labels)
        }

    @staticmethod
    def df_prep(data, n_labels, ignore, label_cols):
        data.reset_index(drop=True, inplace=True)
        return data

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label columns: {}".format(self.label_cols))
        print("number of labels: {}".format(self.n_labels))
        
        # 统计每个标签的分布
        for col in self.label_cols:
            counts = self.slide_data[col].value_counts()
            print(f"Label '{col}' distribution: Negative={counts.get(0, 0)}, Positive={counts.get(1, 0)}")
        
        # 统计标签组合
        print("\nLabel combinations (top 10):")
        sorted_combos = sorted(self.label_combinations.items(), key=lambda x: x[1], reverse=True)
        for combo, count in sorted_combos[:10]:
            print(f"  {combo}: {count} patients")
        
        # 患者级别统计
        for i, col in enumerate(self.label_cols):
            pos_count = len(self.label_cls_ids[col]['positive'])
            neg_count = len(self.label_cls_ids[col]['negative'])
            print(f'Patient-LVL; {col} - Positive: {pos_count}, Negative: {neg_count}')
            
            slide_pos_count = len(self.slide_label_cls_ids[col]['positive'])
            slide_neg_count = len(self.slide_label_cls_ids[col]['negative'])
            print(f'Slide-LVL; {col} - Positive: {slide_pos_count}, Negative: {slide_neg_count}')

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_MultiLabel_Split(df_slice, metadata=self.metadata, mode=self.mode, 
                                           signatures=self.signatures, data_dir=self.data_dir, 
                                           label_cols=self.label_cols, patient_dict=self.patient_dict, 
                                           n_labels=self.n_labels)
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

    def getlabels(self, ids):
        return self.slide_data[self.label_cols].iloc[ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_MultiLabel_Dataset(Generic_WSI_MultiLabel_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_MultiLabel_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        
        # 🔥 确保继承了父类的属性
        # 这些属性应该已经在父类中设置，这里只是确认
        if not hasattr(self, 'num_classes'):
            self.num_classes = self.n_labels
        if not hasattr(self, 'task_type'):
            self.task_type = 'multi_label'

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        # 🔥 多标签使用FloatTensor
        labels = torch.FloatTensor(self.slide_data[self.label_cols].iloc[idx].values)
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            # 🔥 PT模式
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), labels)

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
                    return (path_features, cluster_ids, genomic_features, labels)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features.unsqueeze(dim=0), labels)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), labels)

                elif self.mode == 'pathomictext':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.FloatTensor(self.genomic_features.iloc[idx].values)
                    text_features = torch.FloatTensor(self.text_features.iloc[idx].values)
                    return (path_features, genomic_features.unsqueeze(0), text_features, labels)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, labels
        
        else:
            # 🔥 H5模式 - 添加完整的H5支持
            if self.data_dir:
                if self.mode == 'pathomictext':
                    # 加载H5文件中的WSI特征
                    path_features = []
                    coords_list = []
                    for slide_id in slide_ids:
                        h5_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id.rstrip('.svs')))
                        
                        try:
                            with h5py.File(h5_path, 'r') as f:
                                features = torch.from_numpy(f['tile_embeds'][:]).float()
                                coords = torch.from_numpy(f['coords'][:]).long()
                                path_features.append(features)
                                coords_list.append(coords)
                        except Exception as e:
                            print(f"Error loading H5 file {h5_path}: {e}")
                            # 使用零填充作为后备
                            path_features.append(torch.zeros(1, 1024))
                            coords_list.append(torch.zeros(1, 2))
                    
                    path_features = torch.cat(path_features, dim=0)
                    coords = torch.cat(coords_list, dim=0) if coords_list else torch.zeros(1, 2)
                    
                    # 加载组学和文本特征
                    genomic_features = torch.FloatTensor(self.genomic_features.iloc[idx].values)
                    text_features = torch.FloatTensor(self.text_features.iloc[idx].values)
                    
                    # 🔥 返回格式：(path, omic, text, labels, coords, slide_ids)
                    return (path_features, genomic_features.unsqueeze(0), text_features, 
                            labels, coords, slide_ids)
                
                elif self.mode == 'pathomic':
                    path_features = []
                    coords_list = []
                    for slide_id in slide_ids:
                        h5_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id.rstrip('.svs')))
                        
                        try:
                            with h5py.File(h5_path, 'r') as f:
                                features = torch.from_numpy(f['tile_embeds'][:]).float()
                                coords = torch.from_numpy(f['coords'][:]).long()
                                path_features.append(features)
                                coords_list.append(coords)
                        except Exception as e:
                            print(f"Error loading H5 file {h5_path}: {e}")
                            path_features.append(torch.zeros(1, 1024))
                            coords_list.append(torch.zeros(1, 2))
                    
                    path_features = torch.cat(path_features, dim=0)
                    coords = torch.cat(coords_list, dim=0) if coords_list else torch.zeros(1, 2)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    
                    return (path_features, genomic_features.unsqueeze(dim=0), 
                            labels, coords, slide_ids)
                
                else:
                    raise NotImplementedError(f'H5 mode not implemented for mode: {self.mode}')
            else:
                return slide_ids, labels


class Generic_MultiLabel_Split(Generic_MIL_MultiLabel_Dataset):
    def __init__(self, slide_data, metadata, mode, 
        signatures=None, data_dir=None, label_cols=None, patient_dict=None, n_labels=5):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.n_labels = n_labels
        self.label_cols = label_cols or ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']
        self.patient_dict = patient_dict
        
        # 🔥 添加必要的属性
        self.num_classes = n_labels
        self.task_type = 'multi_label'
        
        # 为每个标签创建正负样本索引
        self.slide_label_cls_ids = {}
        for col in self.label_cols:
            self.slide_label_cls_ids[col] = {
                'positive': np.where(self.slide_data[col] == 1)[0],
                'negative': np.where(self.slide_data[col] == 0)[0]
            }

        ### --> 改进的特征提取方式
        all_cols = self.slide_data.columns.tolist()
        
        metadata_cols = [col for col in all_cols if not ('_cnv' in col or '_mut' in col or '_rnaseq' in col or col.startswith('embedding_'))]
        cnv_cols = [col for col in all_cols if col.endswith('_cnv')]
        mut_cols = [col for col in all_cols if col.endswith('_mut')]
        rna_cols = [col for col in all_cols if col.endswith('_rnaseq')]
        embedding_cols = [col for col in all_cols if col.startswith('embedding_')]
        
        print(f"Feature distribution:")
        print(f"  CNV features: {len(cnv_cols)}")
        print(f"  MUT features: {len(mut_cols)} (excluded for gene mutation prediction)")
        print(f"  RNA features: {len(rna_cols)}")
        print(f"  Embedding features: {len(embedding_cols)}")
        print(f"  Metadata columns: {len(metadata_cols)}")
        
        # 🔥 多标签分类任务排除突变特征
        print("Excluding mutation features for gene mutation prediction task")
        molecular_cols = cnv_cols + rna_cols
        
        for label_col in self.label_cols:
            if label_col in molecular_cols:
                molecular_cols.remove(label_col)
                print(f"Removed label column {label_col} from molecular features")
        
        self.genomic_features = self.slide_data[molecular_cols] if molecular_cols else pd.DataFrame()
        self.text_features = self.slide_data[embedding_cols] if embedding_cols else pd.DataFrame()
        
        self.feature_info = {
            'cnv_cols': cnv_cols,
            'mut_cols': [],
            'rna_cols': rna_cols,
            'embedding_cols': embedding_cols,
            'molecular_cols': molecular_cols,
            'metadata_cols': metadata_cols,
            'label_cols': self.label_cols
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
            print("Genomic features after scaling:", self.genomic_features.shape)

        if 'text' in scalers and not self.text_features.empty:
            transformed_text = pd.DataFrame(scalers['text'].transform(self.text_features))
            transformed_text.columns = self.text_features.columns
            self.text_features = transformed_text
            print("Text features after scaling:", self.text_features.shape)

    def set_split_id(self, split_id):
        self.split_id = split_id
    
    def get_label_distribution(self):
        """获取每个标签的分布情况"""
        distribution = {}
        for col in self.label_cols:
            pos_count = len(self.slide_label_cls_ids[col]['positive'])
            neg_count = len(self.slide_label_cls_ids[col]['negative'])
            distribution[col] = {
                'positive': pos_count,
                'negative': neg_count,
                'total': pos_count + neg_count,
                'pos_ratio': pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
            }
        return distribution
    
    def get_feature_info(self):
        """获取特征分布信息"""
        return self.feature_info