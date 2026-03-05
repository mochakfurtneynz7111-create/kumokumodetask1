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
class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        print("csv_path:!!!",csv_path)
        # 修改后（安全删除）
        if 'Unnamed: 0' in slide_data.columns:
            slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        # slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

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

        """
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]
        废弃代码！
        """

        # 设置bins和classes
        self.n_bins = len(q_bins) - 1
        self.num_classes = self.n_bins  # 4个bins，不是8个
        self.bins = q_bins
        # ✅ 添加这一行：创建简化的label_dict
        self.label_dict = {i: i for i in range(self.num_classes)}# 结果: {0: 0, 1: 1, 2: 2, 3: 3}
        print(f"\n{'='*60}")
        print(f"✓ Survival 任务配置:")
        print(f"  n_bins (时间区间数): {self.n_bins}")
        print(f"  num_classes (模型输出类别数): {self.num_classes}")
        print(f"  bins: {self.bins}")
        print(f"{'='*60}\n")
        
        # ✅ 创建disc_label列（就是bin值）
        slide_data['disc_label'] = slide_data['label']
        
        # 获取患者级别数据
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {
            'case_id': patients_df['case_id'].values, 
            'label': patients_df['disc_label'].values
        }
        
        """
        self.bins = q_bins
        #self.num_classes=len(self.label_dict)
        self.n_bins = len(q_bins) - 1
        self.num_classes = self.n_bins  # Should be 4, not 8
        print(f"\n{'='*60}")
        print(f"✓ Survival 任务配置:")
        print(f"  n_bins (时间区间数): {self.n_bins}")
        print(f"  num_classes (模型输出类别数): {self.num_classes}")
        print(f"  bins: {q_bins}")
        print(f"{'='*60}\n")
        patients_df = slide_data.drop_duplicates(['case_id'])
        #self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}
        
        self.patient_data = {'case_id':patients_df['case_id'].values, 
                     'label':patients_df['disc_label'].values}  # 使用disc_label
        """
        #new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        # metadata = ['disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id', 'age', 'site', 'survival_months', 'censorship', 'is_female', 'oncotree_code', 'train']
        # metadata = ['disc_label', 'case_id', 'slide_id', 'label', 'site', 'is_female','oncotree_code', 'age', 'survival_months', 'censorship', 'train', 'NDUFS5_cnv']
        metadata = ['disc_label', 'case_id', 'slide_id', 'label', 'site', 'is_female','oncotree_code', 'age', 'survival_months', 'censorship']
        self.metadata = slide_data.columns[:10]
        
        for col in slide_data.drop(self.metadata, axis=1).columns:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(col)
        #pdb.set_trace()

        print()
        print("self.metadata:", self.metadata)
        print("pd.Index(metadata):", pd.Index(metadata))
        
        assert self.metadata.equals(pd.Index(metadata))
        self.mode = mode
        self.cls_ids_prep()

        ### ICCV discrepancies
        # For BLCA, TPTEP1_rnaseq was accidentally appended to the metadata
        #pdb.set_trace()

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
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        r"""
        
        """
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""
        
        """

        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

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
            
            # 🔥 传递 text_npy_path
            if hasattr(self.slide_data, 'text_npy_path'):
                df_slice.text_npy_path = self.slide_data.text_npy_path
            
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, 
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
            # print(f"Train Slides: {train_slides}")  # 打印 train_slides 的内容
            # print(f"Length of Train Slides: {len(train_slides)}") # 打印 train_slides 的长度

            # 🔥 添加：设置文本npy路径（在创建split之前）
            #Text NPY not found: ./splits/5foldcv/tcga_luad/splits_0_text_embeddings_qa_level.npy, will use CSV embeddings
            text_npy_path = ('/root/autodl-tmp/PORPOISE/data_text_features/LUAD_text_embeddings_qa_level.npy')
            if os.path.exists(text_npy_path):
                self.slide_data.text_npy_path = text_npy_path
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"✅ Text NPY found: {text_npy_path}")
            else:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                self.slide_data.text_npy_path = None
                print(f"⚠️ Text NPY not found: {text_npy_path}, will use CSV embeddings")

            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            print(f"Train Split Length: {len(train_split)}") # 打印 train_split 的长度
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        self.text_embeddings_raw = None  # 🔥 添加这一行

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])        
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]      
        
        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        # 🔥 新增：H5 模式
        if self.use_h5:
            # H5 格式：返回 dict
            result = {
                'path': None,
                'omic': None,
                'text': None,
                'label': label,
                'event_time': event_time.item(),
                'censorship': c.item(),
                'coords': None,
                'slide_id': slide_ids[0] if len(slide_ids) > 0 else 'unknown'
            }
            
            if self.data_dir:
                # 加载 path features 和 coords
                if self.mode in ['path', 'pathomic', 'pathomictext']:
                    path_features_list = []
                    coords_list = []
                    
                    for slide_id in slide_ids:
                        slide_id_clean = slide_id.rstrip('.svs') 
                        h5_path = os.path.join(data_dir, 'h5_files', f'{slide_id_clean}.h5')
                        
                        if os.path.exists(h5_path):
                            try:
                                with h5py.File(h5_path, 'r') as f:
                                    features = f['tile_embeds'][:]
                                    coords = f['coords'][:]
                                    path_features_list.append(torch.from_numpy(features))
                                    coords_list.append(coords)
                            except Exception as e:
                                print(f"Error loading H5 {h5_path}: {e}")
                                continue
                    
                    if path_features_list:
                        result['path'] = torch.cat(path_features_list, dim=0)
                        result['coords'] = np.vstack(coords_list) if coords_list else None
                
                # 加载 omic features
                if self.mode in ['omic', 'pathomic', 'pathomictext']:
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    result['omic'] = genomic_features.unsqueeze(0)
                
                # 加载 text features
                if self.mode == 'pathomictext':
                    text_features = torch.tensor(self.text_embeddings_raw[idx]).float()  # [6, 768]
                    result['text'] = text_features
            
            return result

        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path': ## 仅WSI
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

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
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    # print("genomic_features",genomic_features.shape) 1024 1024
                    return (torch.zeros((1,1)), genomic_features.unsqueeze(dim=0), label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    # print("------------------------------------------------------,path_features.shape:",path_features.shape)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    # print("------------------------------------------------------,genomic_features.shape:",genomic_features.shape)
                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c)

                
                elif self.mode == 'pathomictext':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    # text_features= torch.tensor(self.text_features.iloc[idx])
                    text_features = torch.tensor(self.text_embeddings_raw[idx]).float()  # [6, 768]
                    return (path_features, genomic_features.unsqueeze(dim=0), text_features, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, 
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
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

        """
        embedding_cols = [col for col in all_cols if col.startswith('embedding_')]
        # 提取文本嵌入特征
        self.text_features = self.slide_data[embedding_cols] if embedding_cols else pd.DataFrame()
        """
        # 🔥 修改：保持 [N, 6, 768] 的 3D 结构
        self.text_npy_path = getattr(slide_data, 'text_npy_path', None)
        self.text_embeddings_raw = None  # 存储原始 3D 数据
        text_npy_path = getattr(slide_data, 'text_npy_path', None)
        
        if text_npy_path and os.path.exists(text_npy_path):
            self.text_embeddings_raw = np.load(text_npy_path)  # [N, 6, 768]
            print(f"✅ Loaded QA-level text embeddings: {self.text_embeddings_raw.shape}")
            
            # 展平用于 StandardScaler
            N = self.text_embeddings_raw.shape[0]
            text_flat = self.text_embeddings_raw.reshape(N, -1)  # [N, 4608]
            text_data = {f'qa_dim_{i}': text_flat[:, i] for i in range(4608)}
            self.text_features = pd.DataFrame(text_data)
            embedding_cols = list(text_data.keys())
        else:
            raise FileNotFoundError(f"Text NPY not found: {text_npy_path}")
        
        print(f"Feature distribution:")
        print(f"  CNV features: {len(cnv_cols)}")
        print(f"  MUT features: {len(mut_cols)}")
        print(f"  RNA features: {len(rna_cols)}")
        print(f"  Embedding features: {len(embedding_cols)}")
        print(f"  Metadata columns: {len(metadata_cols)}")
        
        # 组合分子特征 (CNV + MUT + RNA) - 生存分析任务保留所有特征
        molecular_cols = cnv_cols + mut_cols + rna_cols
        
        # 提取分子特征
        self.genomic_features = self.slide_data[molecular_cols] if molecular_cols else pd.DataFrame()
        
        # 存储特征列信息，用于调试和分析
        self.feature_info = {
            'cnv_cols': cnv_cols,
            'mut_cols': mut_cols,  # 生存分析任务保留突变特征
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
            print("self.genomic_features:", self.genomic_features.shape)
        if not self.text_features.empty:
            scalers['text'] = StandardScaler().fit(self.text_features)
            print("self.text_features:", self.text_features.shape)
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
    
    def get_feature_info(self):
        """获取特征分布信息"""
        return self.feature_info