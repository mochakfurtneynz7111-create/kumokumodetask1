import numpy as np
import torch_geometric

import argparse
import pdb
import os 
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset 
from datasets.dataset_classification import Generic_WSI_Classification_Dataset, Generic_MIL_Classification_Dataset
from datasets.dataset_multilabel import Generic_WSI_MultiLabel_Dataset, Generic_MIL_MultiLabel_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

# main函数
def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # 根据任务类型初始化不同的评估指标存储
    if args.task_type == 'survival':
        latest_val_cindex = []
    elif args.task_type == 'classification':
        latest_val_acc = []
        latest_val_auc = []
        latest_val_f1 = []
    elif args.task_type == 'multi_label':
        latest_val_auc = []
        latest_val_ap = []
        all_per_label_aucs = []  # 🔥 添加这一行
        all_per_label_aps = []   # 🔥 添加这一行
    
    folds = np.arange(start, end)

    ### Start 5-Fold CV Evaluation.
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path) and (not args.overwrite):
            print("Skipping Split %d" % i)
            continue

        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, 
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )
        
        # 🔥 关键修改1: 如果使用H5文件,启用H5加载
        if args.use_h5:
            print(f"\n{'='*70}")
            print(f"✅ H5 mode enabled - loading features and coordinates from H5 files")
            print(f"{'='*70}\n")
            train_dataset.load_from_h5(True)
            val_dataset.load_from_h5(True)
        
        train_dataset.set_split_id(split_id=i)
        val_dataset.set_split_id(split_id=i)
        
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)
        
        # 打印特征信息
        if hasattr(train_dataset, 'get_feature_info'):
            feature_info = train_dataset.get_feature_info()
            print(f"Feature composition for fold {i}:")
            print(f"  CNV features: {len(feature_info['cnv_cols'])}")
            print(f"  MUT features: {len(feature_info['mut_cols'])}")
            print(f"  RNA features: {len(feature_info['rna_cols'])}")
            print(f"  Embedding features: {len(feature_info['embedding_cols'])}")
        
        ### Specify the input dimension size if using genomic features.
        if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)
        elif 'coattn' in args.mode:
            args.omic_sizes = train_dataset.omic_sizes
            print('Genomic Dimensions', args.omic_sizes)
        else:
            args.omic_input_dim = 0

        # 🔥 单模态：恢复被 mode 判断覆写的 dim
        if getattr(args, 'single_modality', None) == 'path':
            args.omic_input_dim = 0
            args.text_input_dim = 0
        elif getattr(args, 'single_modality', None) == 'omic':
            args.path_input_dim = 0
            args.text_input_dim = 0
        elif getattr(args, 'single_modality', None) == 'text':
            args.path_input_dim = 0
            args.omic_input_dim = 0

        ### 选择训练函数
        if args.model_type == 'gram_porpoise_mmf':
            # 🔥 使用GRAM版本 (支持可解释性)
            from utils.core_utils_gram import train as train_gram
            
            # 如果使用CPath-Omni融合，建议调整超参数
            if hasattr(args, 'fusion') and args.fusion.startswith('cpathomni'):
                print(f"\n{'='*70}")
                print(f"⚠️  Detected CPath-Omni fusion: {args.fusion}")
                print(f"📌 Recommended adjustments:")
                print(f"   - Learning rate: 1e-4 to 2e-4 (current: {args.lr})")
                print(f"   - Consider dropout: 0.1-0.2")
                print(f"{'='*70}\n")
            
            train_func = train_gram
        else:
            # 使用原有版本
            from utils.core_utils import train
            train_func = train

        ### Run Train-Val on Different Tasks.
        if args.task_type == 'survival':
            val_latest, cindex_latest = train_func(datasets, i, args)
            latest_val_cindex.append(cindex_latest)
        elif args.task_type == 'classification':
            val_latest, acc_latest, auc_latest, f1_latest = train_func(datasets, i, args)
            latest_val_acc.append(acc_latest)
            latest_val_auc.append(auc_latest)
            latest_val_f1.append(f1_latest)
        elif args.task_type == 'multi_label':
            val_latest, mean_auc, mean_ap, aucs, aps = train_func(datasets, i, args)
            latest_val_auc.append(mean_auc)
            latest_val_ap.append(mean_ap)
            all_per_label_aucs.append(aucs)
            all_per_label_aps.append(aps)

        ### Write Results for Each Split to PKL
        save_pkl(results_pkl_path, val_latest)
        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    ### Finish 5-Fold CV Evaluation.
    if args.task_type == 'survival':
        if len(latest_val_cindex) == 0:
            latest_val_cindex = [np.nan] * len(folds)
        print(len(folds), len(latest_val_cindex))
        results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    elif args.task_type == 'classification':
        if len(latest_val_acc) == 0:
            latest_val_acc = [np.nan] * len(folds)
            latest_val_auc = [np.nan] * len(folds)
            latest_val_f1 = [np.nan] * len(folds)
        print(len(folds), len(latest_val_acc))
        results_latest_df = pd.DataFrame({
            'folds': folds, 
            'val_acc': latest_val_acc,
            'val_auc': latest_val_auc,
            'val_f1': latest_val_f1
        })
    elif args.task_type == 'multi_label':
        if len(latest_val_auc) == 0:
            latest_val_auc = [np.nan] * len(folds)
            latest_val_ap = [np.nan] * len(folds)  # ✅
        
        print(len(folds), len(latest_val_auc))
        
        results_latest_df = pd.DataFrame({
            'folds': folds,
            'val_mean_auc': latest_val_auc,
            'val_mean_ap': latest_val_ap  # ✅
        })
        
        # 添加per-label结果
        if len(all_per_label_aucs) > 0 and hasattr(args, 'gene_names'):
            all_per_label_aucs_array = np.array(all_per_label_aucs)
            all_per_label_aps_array = np.array(all_per_label_aps)
            
            for j, gene_name in enumerate(args.gene_names):
                results_latest_df[f'{gene_name}_auc'] = all_per_label_aucs_array[:, j]
                results_latest_df[f'{gene_name}_ap'] = all_per_label_aps_array[:, j]
        
        # 打印汇总
        print(f"\n{'='*70}")
        print("Multi-Label CV Summary:")
        print(f"Mean AUC: {np.mean(latest_val_auc):.4f} ± {np.std(latest_val_auc):.4f}")
        print(f"Mean AP:  {np.mean(latest_val_ap):.4f} ± {np.std(latest_val_ap):.4f}")
        print(f"{'='*70}\n")

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis, Classification, and Multi-Label Classification on TCGA Data.')

### 🔥 单模态支持
parser.add_argument('--single_modality', type=str, 
                   choices=[None, 'path', 'omic', 'text'], 
                   default=None,
                   help='Use single modality only (path/omic/text). If None, use all available modalities.')

### Task Type Parameter
parser.add_argument('--task_type', type=str, choices=['survival', 'classification', 'multi_label'], 
                   default='survival', help='Type of task: survival analysis, classification, or multi-label classification (default: survival)')
parser.add_argument('--extract_features', action='store_true', default=False, 
                   help='Extract and save fusion features after training')

### 🔥 H5 File Support (新增)
parser.add_argument('--use_h5', action='store_true', default=False,
                   help='Use H5 files for loading features with coordinates (enables interpretability)')
parser.add_argument('--preprocessing_dir', type=str, 
                   default='./outputs_LUAD/preprocessing',
                   help='Path to preprocessing directory containing output/thumbnails for interpretability')

### 添加新的命令行参数 (PROV MUSK UNI H5/PT路径选择，新增)
parser.add_argument('--feature_prefix', type=str, default=None,
                   help='Prefix for feature processing method (PROV, MUSK, UNI, etc.)')
parser.add_argument('--feature_dir', type=str, default=None,
                   help='Directly specify feature directory name (overrides auto-construction)')


### 🔥 Interpretability Parameters
parser.add_argument('--enable_interpretability', action='store_true', default=False,
                   help='Enable interpretability visualization during training')
parser.add_argument('--interpretability_freq', type=int, default=5,
                   help='Generate interpretability visualizations every N epochs (default: 5)')
parser.add_argument('--max_visualize_per_epoch', type=int, default=3,
                   help='Maximum number of samples to visualize per epoch (default: 3)')
parser.add_argument('--class_names', nargs='+', default=None,
                   help='Class names for visualization (e.g., --class_names LUAD LUSC HNSC STAD)')
parser.add_argument('--label_names', nargs='+', default=None,
                   help='Label names for multi-label tasks (e.g., --label_names TP53 KRAS PIK3CA)')


### 🔥 Explainability Parameters
parser.add_argument('--enable_explainability', action='store_true', default=False,
                   help='Enable model explainability module (for text-centric interpretation)')
parser.add_argument('--enable_text_centric', action='store_true', default=False,
                   help='Enable text-centric interpretability (QA as semantic anchors)')
parser.add_argument('--n_qa_pairs', type=int, default=6,
                   help='Number of QA pairs (default: 6)')
parser.add_argument('--n_pathways', type=int, default=50,
                   help='Number of pathways for omic explainability (default: 50)')
parser.add_argument('--pathway_gene_mapping', type=str, default=None,
                   help='Path to pathway-gene mapping file (optional, .pt format)')
parser.add_argument('--qa_text_file', type=str, default=None,
                   help='Path to file containing QA texts (one per line)')
parser.add_argument('--pathway_names_file', type=str, default='./pathway_names.txt',
                   help='Path to file containing pathway names (one per line)')
parser.add_argument('--save_explanation_results', action='store_true', default=False,
                   help='Save explanation results to pickle files')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='dataroot', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--dataset_path',    type=str, default='./dataset_csv', help='Path to dataset CSV files')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results_new', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_blca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type', type=str, 
                   choices=['mcat', 'porpoise_mmf', 'porpoise_amil', 'snn', 'deepset', 'amil', 'mi_fcn', 'gram_porpoise_mmf'], 
                   default='mcat', 
                   help='Type of model (Default: mcat). Use gram_porpoise_mmf for GRAM-enhanced multimodal model')

parser.add_argument('--mode',            type=str, choices=['text','omic', 'path', 'pathomic', 'pathomic_fast', 'cluster', 'coattn', 'pathomictext'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear','gram','transformer','concatonly','transformer_medium', 'cross_attention', 'cpathomni_sequential','highdim_fusion'], default='bilinear', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

# 任务相关的类别/标签数
parser.add_argument('--n_classes', type=int, default=8, help='Number of classes for classification task')
parser.add_argument('--n_labels', type=int, default=5, help='Number of labels for multi-label classification task (e.g., 5 genes)')
parser.add_argument('--gene_names', nargs='+', default=['EGFR_mut', 'FAT1_mut', 'KRAS_mut', 'LRP1B_mut', 'TP53_mut'], 
                   help='Gene names for multi-label prediction') #EGFR_mut	FAT1_mut	KRAS_mut	LRP1B_mut	TP53_mut


### GRAM-Specific Parameters
parser.add_argument('--use_gram_contrastive', action='store_true', default=False,
                   help='Enable GRAM contrastive learning (volume-based alignment)')
parser.add_argument('--contrastive_weight', type=float, default=0.5,
                   help='Weight for GRAM contrastive loss relative to task loss (default: 0.5)')
parser.add_argument('--contra_dim', type=int, default=512,
                   help='Dimension for GRAM contrastive learning projections (default: 256)')
parser.add_argument('--contra_temp', type=float, default=0.07,
                   help='Temperature parameter for GRAM volume computation (default: 0.07)')
parser.add_argument('--use_gram_fusion', action='store_true', default=False,
                   help='Enable GRAM-Fusion for adaptive multimodal fusion')

### PORPOISE
parser.add_argument('--apply_mutsig', action='store_true', default=False)
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--gate_text', action='store_true', default=False, help='Gate for text modality in multi-modal fusion')
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--scale_dim3', type=int, default=8, help='Scaling dimension for third modality (text)')
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1536)
parser.add_argument('--text_input_dim', type=int, default=768, help='Input dimension for text features')
parser.add_argument('--use_mlp', action='store_true', default=False)

### Optimizer Parameters + Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=1, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=500, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=0.0001, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'focal', 'bce'], 
                   default='nll_surv', help='slide-level loss function (default: nll_surv for survival, ce for classification, bce for multi-label)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--metric_early_stopping', action='store_true', default=False, 
                   help='Enable metric-based early stopping (accuracy for classification, C-index for survival)')

### Multi-Label specific parameters
parser.add_argument('--multilabel_threshold', type=float, default=0.5, help='Threshold for multi-label predictions (default: 0.5)')
parser.add_argument('--pos_weight', type=float, default=None, help='Positive class weight for imbalanced multi-label datasets')

### Contrastive Learning
parser.add_argument('--lambda_contra', type=float, default=1e-5, help='Weight for contrastive loss')

### CLAM-Specific Parameters
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--testing', 	 	 action='store_true', default=False, help='debugging tool')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)

# 根据任务类型设置任务名称
if args.task_type == 'survival':
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
elif args.task_type == 'classification':
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_classification'
elif args.task_type == 'multi_label':
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_multilabel'

print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1536
settings = {'num_splits': args.k, 
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'task_type': args.task_type,
			'max_epochs': args.max_epochs, 
			'results_dir': args.results_dir, 
			'lr': args.lr,
			'experiment': args.exp_code,
			'reg': args.reg,
			'label_frac': args.label_frac,
			'bag_loss': args.bag_loss,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size_wsi': args.model_size_wsi,
			'model_size_omic': args.model_size_omic,
			"use_drop_out": args.drop_out,
			'weighted_sample': args.weighted_sample,
			'gc': args.gc,
			'opt': args.opt,
			'n_classes': args.n_classes,
			'n_labels': args.n_labels}

# 🔥 添加可解释性设置
settings.update({
    'enable_explainability': args.enable_explainability,
    'enable_text_centric': args.enable_text_centric,
    'n_qa_pairs': args.n_qa_pairs,
    'n_pathways': args.n_pathways,
    'pathway_names_file': args.pathway_names_file,
})

# 🔥 关键修改2: H5支持提示
if args.use_h5:
    print(f"\n{'='*70}")
    print(f"🔍 H5 Mode Configuration:")
    print(f"  Using H5 files: {args.use_h5}")
    print(f"  H5 files should contain:")
    print(f"    - 'features': patch features")
    print(f"    - 'coords': patch coordinates")
    if args.enable_interpretability:
        print(f"  Interpretability enabled: coordinates will be used for WSI heatmaps")
        print(f"  Preprocessing dir: {args.preprocessing_dir}")
    print(f"{'='*70}\n")


### 修改数据集加载部分的逻辑
def build_feature_directory(args):
    """智能构建特征目录路径"""
    
    # 如果直接指定了特征目录，直接使用
    if args.feature_dir:
        return args.feature_dir
    
    # 提取癌症类型（从 split_dir）
    # split_dir 可能是 'tcga_coadread' 或 'tcga_coadread_survival'
    if args.split_dir.startswith('tcga_'):
        cancer_parts = args.split_dir.split('_')
        # 取前两部分作为癌症类型（如 'tcga_coadread'）
        cancer_type = '_'.join(cancer_parts[:2])
    else:
        cancer_type = args.split_dir
    
    # 构建特征目录
    if args.feature_prefix:
        # 使用指定的前缀
        feature_dir = f'{args.feature_prefix}_{cancer_type}_20x_features'
    else:
        # 默认情况（兼容旧代码）
        feature_dir = f'{cancer_type}_20x_features'
    
    return feature_dir

### 3. 在数据集加载前调用
feature_dir = build_feature_directory(args)
print(f"📁 Using feature directory: {feature_dir}")


# 数据集加载部分
print('\nLoad Dataset')

# 根据任务类型加载不同的数据集
if args.task_type == 'survival':
	study = '_'.join(args.task.split('_')[:2])
	study_dir = '%s_20x_features' % study

	dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/text_%s_all_clean.csv.zip' % (args.dataset_path, study),
                                           mode = args.mode,
										   apply_sig = args.apply_sig,
										   data_dir= os.path.join(args.data_root_dir, feature_dir),
										   shuffle = False, 
										   seed = args.seed, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=5,
										   label_col = 'survival_months',
										   ignore=[])

elif args.task_type == 'classification':
	study = '_'.join(args.task.split('_')[:2])	
	study_dir = '%s_20x_features' % study

	dataset = Generic_MIL_Classification_Dataset(csv_path = './%s/tp53_%s_all_clean.csv.zip' % (args.dataset_path, study),
												 mode = args.mode,
												 apply_sig = args.apply_sig,
												 data_dir= os.path.join(args.data_root_dir, feature_dir),
												 shuffle = False, 
												 seed = args.seed, 
												 print_info = True,
												 patient_strat= False,
												 n_classes=args.n_classes,
												 label_col = 'label',
												 ignore=[])

elif args.task_type == 'multi_label':
	study = '_'.join(args.task.split('_')[:2])	
	study_dir = '%s_20x_features' % study

	dataset = Generic_MIL_MultiLabel_Dataset(
		csv_path = './%s/multilabel_%s_all_clean.csv.zip' % (args.dataset_path, feature_dir),
		mode = args.mode,
		apply_sig = args.apply_sig,
		data_dir= os.path.join(args.data_root_dir, study_dir),
		shuffle = False, 
		seed = args.seed, 
		print_info = True,
		patient_strat= False,
		n_labels=args.n_labels,
		label_cols = args.gene_names,
		ignore=[]
	)

else:
	raise NotImplementedError

# 根据任务类型自动设置默认损失函数
if args.task_type == 'classification' and args.bag_loss == 'nll_surv':
    args.bag_loss = 'ce'
elif args.task_type == 'multi_label' and args.bag_loss in ['nll_surv', 'ce']:
    args.bag_loss = 'bce'

# 🔥 单模态配置
if args.single_modality is not None:
    print(f"\n{'='*70}")
    print(f"🔥 Single Modality Mode: {args.single_modality.upper()}")
    
    if args.single_modality == 'path':
        # 不改 args.mode，dataset 已用 'pathomictext' 构建，
        # collate 继续按 6 元素处理，模型因 fc_omic/fc_text=None 忽略对应模态
        args.omic_input_dim = 0
        args.text_input_dim = 0
        print(f"   Mode: path only (keeping args.mode={args.mode})")
        print(f"   Omic dim: 0 (disabled)")
        print(f"   Text dim: 0 (disabled)")
        
    elif args.single_modality == 'omic':
        # 不改 args.mode，理由同 path-only
        args.path_input_dim = 0
        args.text_input_dim = 0
        print(f"   Mode: omic only (keeping args.mode={args.mode})")
        print(f"   Path dim: 0 (disabled)")
        print(f"   Text dim: 0 (disabled)")
        
    elif args.single_modality == 'text':
        args.mode = 'text'
        args.path_input_dim = 0
        args.omic_input_dim = 0
        print(f"   Mode: text only")
        print(f"   Path dim: 0 (disabled)")
        print(f"   Omic dim: 0 (disabled)")
    
    print(f"{'='*70}\n")

# 🔥 自动设置class_names（如果用户没有提供）
if args.enable_interpretability:
    if args.class_names is None:
        if args.task_type == 'classification':
            args.class_names = [f'Class{i}' for i in range(args.n_classes)]
            print(f"🔍 Auto-generated class names: {args.class_names}")
        elif args.task_type == 'survival':
            args.class_names = [f'Time{i}' for i in range(args.n_classes)]
            print(f"🔍 Auto-generated class names: {args.class_names}")
    
    if args.task_type == 'multi_label' and args.label_names is None:
        args.label_names = args.gene_names
        print(f"🔍 Using gene names as label names: {args.label_names}")

# 更新 settings
args.n_classes = dataset.num_classes
print(f"\n" + "=" * 60)
if args.task_type == 'survival':
    print(f"✓ Survival 任务配置:")
    print(f"  n_bins (时间区间数): 4")
    print(f"  num_classes (模型输出类别数): {args.n_classes}")
    print(f"  关系: num_classes = n_bins × 2 = {args.n_classes}")
elif args.task_type == 'classification':
    print(f"✓ Classification 任务配置:")
    print(f"  num_classes: {args.n_classes}")
elif args.task_type == 'multi_label':
    print(f"✓ Multi-Label 任务配置:")
    print(f"  num_labels: {args.n_labels}")
print(f"=" * 60 + "\n")

# 🔥 显示可解释性配置
if args.enable_interpretability:
    print(f"\n" + "=" * 60)
    print(f"🔍 Interpretability Configuration:")
    print(f"  Enabled: True")
    print(f"  Frequency: Every {args.interpretability_freq} epochs")
    print(f"  Max visualizations per epoch: {args.max_visualize_per_epoch}")
    print(f"  Class names: {args.class_names if args.task_type != 'multi_label' else args.label_names}")
    print(f"  Save directory: {args.results_dir}/interpretability_foldX/")
    if args.use_h5:
        print(f"  Using H5 coordinates: Yes")
        print(f"  Preprocessing dir: {args.preprocessing_dir}")
    else:
        print(f"  Using H5 coordinates: No (will use grid layout)")
    print(f"=" * 60 + "\n")

# 🔥 在它之后添加：
if args.enable_explainability or args.enable_text_centric:
    print(f"\n" + "=" * 60)
    print(f"🔍 Text-Centric Explainability Configuration:")
    print(f"  Model explainability: {args.enable_explainability}")
    print(f"  Text-centric mode: {args.enable_text_centric}")
    print(f"  Number of QA pairs: {args.n_qa_pairs}")
    print(f"  Number of pathways: {args.n_pathways}")
    if args.pathway_names_file:
        exists = "✅" if os.path.exists(args.pathway_names_file) else "❌"
        print(f"  Pathway names file: {args.pathway_names_file} {exists}")
    print(f"=" * 60 + "\n")

settings.update({'n_classes': args.n_classes})

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

### Appends to the results_dir path
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
	print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
	sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

settings.update({
    'use_gram_fusion': args.use_gram_fusion,
    'use_gram_contrastive': args.use_gram_contrastive,
    'contrastive_weight': args.contrastive_weight,
    'contra_dim': args.contra_dim,
    'contra_temp': args.contra_temp
})

# 🔥 更新可解释性相关设置
settings.update({
    'use_h5': args.use_h5,
    'preprocessing_dir': args.preprocessing_dir,
    'enable_interpretability': args.enable_interpretability,
    'interpretability_freq': args.interpretability_freq,
    'max_visualize_per_epoch': args.max_visualize_per_epoch,
    'class_names': args.class_names if hasattr(args, 'class_names') else None,
    'label_names': args.label_names if hasattr(args, 'label_names') else None,
    # 🔥 新增可解释性设置
    'enable_explainability': args.enable_explainability,
    'enable_text_centric': args.enable_text_centric,
    'n_qa_pairs': args.n_qa_pairs,
    'n_pathways': args.n_pathways,
    'pathway_names_file': args.pathway_names_file,
})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
	print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
