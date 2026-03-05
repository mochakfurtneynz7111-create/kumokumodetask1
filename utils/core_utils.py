from argparse import Namespace
from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from datasets.dataset_generic import save_splits
from models.model_genomic import SNN
from models.model_set_mil import MIL_Sum_FC_surv, MIL_Attention_FC_surv, MIL_Cluster_FC_surv
from models.model_coattn import MCAT_Surv
# from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL, PorpoiseMMF_Fast
from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL
from utils.utils import *
from utils.loss_func import NLLSurvLoss

from utils.coattn_train_utils import *
from utils.cluster_train_utils import *

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np

class AccuracyEarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
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
        score = val_acc  # 使用准确率，值越高越好

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, ckpt_name)
        elif score <= self.best_score:  # 准确率没有提高
            self.counter += 1
            print(f'AccuracyEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:  # 准确率提高了
            self.best_score = score
            self.save_checkpoint(val_acc, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, ckpt_name):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_acc_max = val_acc
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=20, stop_epoch=30, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


# 1. 分类任务的监控器
class Monitor_Acc:
    """Monitor accuracy for classification tasks."""
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

# 1. 生存分析任务的监控器
class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)

class Monitor_MultiLabel:
    """Monitor AUC for multi-label classification tasks."""
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


def train_loop_multilabel(epoch, model, loader, optimizer, n_labels, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    """多标签分类的训练循环（单模态版本）- 参数与其他任务一致"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    
    print('\n')
    for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
        data_WSI = data_WSI.to(device).float()
        data_omic = data_omic.to(device).float()
        label = label.to(device).float()  # 多标签使用float
        
        logits = model(x_path=data_WSI, x_omic=data_omic)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = loss_fn(logits, label)
        
        if reg_fn is not None:
            loss += reg_fn(model) * lambda_reg
            
        loss_value = loss.item()
        train_loss += loss_value
        
        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(
                batch_idx, loss_value, label[0].cpu().numpy(), data_WSI.size(0)))
        
        # backward pass
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)


def validate_multilabel(cur, epoch, model, loader, n_labels, early_stopping=None, monitor_multilabel=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    """多标签分类的验证函数（单模态版本）- 参数与其他任务一致"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    prob = np.zeros((len(loader), n_labels))
    labels = np.zeros((len(loader), n_labels))
    
    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
            data_WSI = data_WSI.to(device).float()
            data_omic = data_omic.to(device).float()
            label = label.to(device).float()
            
            logits = model(x_path=data_WSI, x_omic=data_omic)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            loss = loss_fn(logits, label)
            
            if reg_fn is not None:
                loss += reg_fn(model) * lambda_reg
                
            val_loss += loss.item()
            
            prob[batch_idx] = torch.sigmoid(logits).cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()

    val_loss /= len(loader)
    
    # 计算多标签指标
    try:
        auc_scores = []
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:  # 确保有正负样本
                auc = roc_auc_score(labels[:, i], prob[:, i])
                auc_scores.append(auc)
        
        val_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        # 使用阈值0.5计算F1分数
        pred_binary = (prob > 0.5).astype(int)
        val_f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        val_f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
    except:
        val_auc = val_f1_macro = val_f1_micro = 0.0

    print('Epoch: {}, val_loss: {:.4f}, val_auc: {:.4f}, val_f1_macro: {:.4f}, val_f1_micro: {:.4f}'.format(
        epoch, val_loss, val_auc, val_f1_macro, val_f1_micro))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', val_auc, epoch)
        writer.add_scalar('val/f1_macro', val_f1_macro, epoch)
        writer.add_scalar('val/f1_micro', val_f1_micro, epoch)

    if monitor_multilabel:
        monitor_multilabel(val_auc, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_multilabel(model, loader, n_labels):
    """多标签分类的总结函数（单模态版本）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    
    prob = np.zeros((len(loader), n_labels))
    labels = np.zeros((len(loader), n_labels))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
            data_WSI = data_WSI.to(device).float()
            data_omic = data_omic.to(device).float()
            label = label.to(device).float()
            slide_id = slide_ids.iloc[batch_idx]
            
            logits = model(x_path=data_WSI, x_omic=data_omic)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            prob[batch_idx] = torch.sigmoid(logits).cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()
            
            patient_results[slide_id] = {
                'slide_id': slide_id,
                'Y': label.cpu().numpy(),
                'Y_hat': torch.sigmoid(logits).cpu().numpy(),
                'prob': torch.sigmoid(logits).cpu().numpy()
            }

    # 计算指标
    try:
        auc_scores = []
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:
                auc = roc_auc_score(labels[:, i], prob[:, i])
                auc_scores.append(auc)
        
        test_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        pred_binary = (prob > 0.5).astype(int)
        test_f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        test_f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
        
        # 每个标签的详细指标
        label_metrics = {}
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:
                label_auc = roc_auc_score(labels[:, i], prob[:, i])
                label_f1 = f1_score(labels[:, i], pred_binary[:, i], zero_division=0)
                label_metrics[f'label_{i}'] = {'auc': label_auc, 'f1': label_f1}
    except:
        test_auc = test_f1_macro = test_f1_micro = 0.0
        label_metrics = {}

    # return test_auc, test_f1_macro, test_f1_micro, label_metrics
    return patient_results, test_auc, test_f1_macro, test_f1_micro


def train_loop_multilabel_pathomictext(epoch, model, loader, optimizer, n_labels, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    """多标签分类的训练循环（pathomictext版本）- 参数与其他任务一致"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    
    print('\n')
    for batch_idx, (data_WSI, data_omic, data_text, label) in enumerate(loader):
        data_WSI = data_WSI.to(device).float()
        data_omic = data_omic.to(device).float()
        data_text = data_text.to(device).float()
        label = label.to(device).float()  # 多标签使用float
        
        logits = model(x_path=data_WSI, x_omic=data_omic, x_text=data_text)
        
        loss = loss_fn(logits, label)
        
        if reg_fn is not None:
            loss += reg_fn(model) * lambda_reg
            
        loss_value = loss.item()
        train_loss += loss_value
        
        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(
                batch_idx, loss_value, label[0].cpu().numpy(), data_WSI.size(0)))
        
        # backward pass
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)


def validate_multilabel_pathomictext(cur, epoch, model, loader, n_labels, early_stopping=None, monitor_multilabel=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    """多标签分类的验证函数（pathomictext版本）- 参数与其他任务一致"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    prob = np.zeros((len(loader), n_labels))
    labels = np.zeros((len(loader), n_labels))
    
    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic, data_text, label) in enumerate(loader):
            data_WSI = data_WSI.to(device).float()
            data_omic = data_omic.to(device).float()
            data_text = data_text.to(device).float()
            label = label.to(device).float()
            
            logits = model(x_path=data_WSI, x_omic=data_omic, x_text=data_text)
            
            loss = loss_fn(logits, label)
            
            if reg_fn is not None:
                loss += reg_fn(model) * lambda_reg
                
            val_loss += loss.item()
            
            prob[batch_idx] = torch.sigmoid(logits).cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()

    val_loss /= len(loader)
    
    # 计算多标签指标
    try:
        auc_scores = []
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:  # 确保有正负样本
                auc = roc_auc_score(labels[:, i], prob[:, i])
                auc_scores.append(auc)
        
        val_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        # 使用阈值0.5计算F1分数
        pred_binary = (prob > 0.5).astype(int)
        val_f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        val_f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
    except:
        val_auc = val_f1_macro = val_f1_micro = 0.0

    print('Epoch: {}, val_loss: {:.4f}, val_auc: {:.4f}, val_f1_macro: {:.4f}, val_f1_micro: {:.4f}'.format(
        epoch, val_loss, val_auc, val_f1_macro, val_f1_micro))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', val_auc, epoch)
        writer.add_scalar('val/f1_macro', val_f1_macro, epoch)
        writer.add_scalar('val/f1_micro', val_f1_micro, epoch)

    if monitor_multilabel:
        monitor_multilabel(val_auc, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_multilabel_pathomictext(model, loader, n_labels):
    """多标签分类的总结函数（pathomictext版本）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    prob = np.zeros((len(loader), n_labels))
    labels = np.zeros((len(loader), n_labels))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic, data_text, label) in enumerate(loader):
            data_WSI = data_WSI.to(device).float()
            data_omic = data_omic.to(device).float()
            data_text = data_text.to(device).float()
            slide_id = slide_ids.iloc[batch_idx]
            label = label.to(device).float()
            
            logits = model(x_path=data_WSI, x_omic=data_omic, x_text=data_text)
            
            prob[batch_idx] = torch.sigmoid(logits).cpu().numpy()
            labels[batch_idx] = label.cpu().numpy()
            
            patient_results[slide_id] = {
                'slide_id': slide_id,
                'Y': label.cpu().numpy(),
                'Y_hat': torch.sigmoid(logits).cpu().numpy(),
                'prob': torch.sigmoid(logits).cpu().numpy()
            }

    # 计算指标
    try:
        auc_scores = []
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:
                auc = roc_auc_score(labels[:, i], prob[:, i])
                auc_scores.append(auc)
        
        test_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        pred_binary = (prob > 0.5).astype(int)
        test_f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        test_f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
        
        # 每个标签的详细指标
        label_metrics = {}
        for i in range(n_labels):
            if len(np.unique(labels[:, i])) > 1:
                label_auc = roc_auc_score(labels[:, i], prob[:, i])
                label_f1 = f1_score(labels[:, i], pred_binary[:, i], zero_division=0)
                label_metrics[f'label_{i}'] = {'auc': label_auc, 'f1': label_f1}
    except:
        test_auc = test_f1_macro = test_f1_micro = 0.0
        label_metrics = {}

    return patient_results, test_auc, test_f1_macro, test_f1_micro
"""
##############################################################################################################
"""

def train(datasets: tuple, cur: int, args: Namespace):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    #损失函数初始化部分（生存分析+分类+多标签）
    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        else:
            raise NotImplementedError
    elif args.task_type == 'multi_label':
        if args.bag_loss == 'bce':
            # 多标签二分类交叉熵损失
            if hasattr(args, 'pos_weight') and args.pos_weight is not None:
                pos_weight = torch.tensor([args.pos_weight] * args.n_labels).float()
                if torch.cuda.is_available():
                    pos_weight = pos_weight.cuda()
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
    else:  # classification
        if args.bag_loss == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')

    # 模型初始化代码
    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'porpoise_mmf':
        if args.task_type == 'multi_label':
            output_dim = args.n_labels  # 多标签任务使用n_labels
        else:
            output_dim = args.n_classes  # 其他任务使用n_classes
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': output_dim, 'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,}
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'porpoise_amil':
        model_dict = {'n_classes': args.n_classes}
        model = PorpoiseAMIL(**model_dict)
    elif args.model_type =='snn':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'deepset':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)
    elif args.model_type =='amil':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)
    elif args.model_type == 'mi_fcn':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'num_clusters': 10, 'n_classes': args.n_classes}
        model = MIL_Cluster_FC_surv(**model_dict)
    elif args.model_type == 'mcat':
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'gram_porpoise_mmf':
        from models.gram_porpoise import GRAMPorpoiseMMF
        if args.task_type == 'multi_label':
            output_dim = args.n_labels
        else:
            output_dim = args.n_classes
        model_dict = {
            'omic_input_dim': args.omic_input_dim,
            'text_input_dim': getattr(args, 'text_input_dim', 768),
            'path_input_dim': args.path_input_dim,
            'fusion': args.fusion,
            'n_classes': output_dim,
            'contra_dim': getattr(args, 'contra_dim', 256),
            'contra_temp': getattr(args, 'contra_temp', 0.07),
            'use_contrastive': getattr(args, 'use_gram_contrastive', True),
            'dropout': 0.25 if args.drop_out else 0.0,
            'task_mode': args.task_type
        }
        model = GRAMPorpoiseMMF(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    """
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, 
    weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size, task_type=args.task_type)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, 
    batch_size=args.batch_size, task_type=args.task_type)
    print('Done!')
    """
    print('\nInit Loaders...', end=' ')
    # 对多标签分类任务禁用weighted_sample，因为Generic_MultiLabel_Split没有slide_cls_ids属性
    if args.task_type == 'multi_label':
        train_weighted = False
        print("Warning: Weighted sampling disabled for multi-label classification")
    else:
        train_weighted = args.weighted_sample

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, 
    weighted=train_weighted, mode=args.mode, batch_size=args.batch_size, task_type=args.task_type)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, 
    batch_size=args.batch_size, task_type=args.task_type)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=5, patience=20, stop_epoch=30, verbose = True)
    else:
        early_stopping = None

    # 【新增】准确率早停设置
    if args.task_type == 'classification':  # 只对分类任务使用
        if hasattr(args, 'acc_early_stopping') and args.acc_early_stopping:
            acc_stopping = AccuracyEarlyStopping(warmup=5, patience=20, stop_epoch=30, verbose=True)
            print('\nSetup Accuracy-based Early Stopping...Done!')
        else:
            acc_stopping = None
    else:
        acc_stopping = None

    # 监控器设置部分
    if args.task_type == 'survival':
        print('\nSetup Validation C-Index Monitor...', end=' ')
        monitor_cindex = Monitor_CIndex()
        print('Done!')
    elif args.task_type == 'multi_label':
        print('\nSetup Validation Multi-Label Monitor...', end=' ')
        monitor_multilabel = Monitor_MultiLabel()
        print('Done!')
    else:
        print('\nSetup Validation Accuracy Monitor...', end=' ')
        monitor_acc = Monitor_Acc()
        print('Done!')

    # 训练循环部分
    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.mode == 'pathomictext':
                train_loop_survival_pathomictext(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_survival_pathomictext(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
            else:
                train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
        elif args.task_type == 'multi_label':
            if args.mode == 'pathomictext':
                train_loop_multilabel_pathomictext(epoch, model, train_loader, optimizer, args.n_labels, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_multilabel_pathomictext(cur, epoch, model, val_loader, args.n_labels, early_stopping, monitor_multilabel, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
            else:
                train_loop_multilabel(epoch, model, train_loader, optimizer, args.n_labels, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_multilabel(cur, epoch, model, val_loader, args.n_labels, early_stopping, monitor_multilabel, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
        else:  # classification
            if args.mode == 'pathomictext':
                train_loop_classification_pathomictext(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_classification_pathomictext(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_acc, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, acc_stopping)
            else:
                train_loop_classification(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_classification(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_acc, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, acc_stopping)
        if stop:
            print("Early stopping triggered!")
            break

    # 最终结果返回部分 #####加不加这行很重要！
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    # 加载最高 acc 的模型
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # 加载最低 loss 的模型
    # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))

    if args.task_type == 'survival':
        if args.mode == 'pathomictext':
            results_val_dict, val_cindex = summary_survival_pathomictext(model, val_loader, args.n_classes)
        else:
            results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)
        print('Val c-Index: {:.4f}'.format(val_cindex))
        writer.close()
        return results_val_dict, val_cindex
    elif args.task_type == 'multi_label':
        if args.mode == 'pathomictext':
            results_val_dict, val_auc, val_f1_macro, val_f1_micro = summary_multilabel_pathomictext(model, val_loader, args.n_labels)
        else:
            results_val_dict, val_auc, val_f1_macro, val_f1_micro = summary_multilabel(model, val_loader, args.n_labels)
        print('Val AUC: {:.4f}, Val F1-Macro: {:.4f}, Val F1-Micro: {:.4f}'.format(val_auc, val_f1_macro, val_f1_micro))
        writer.close()
        return results_val_dict, val_auc, val_f1_macro, val_f1_micro
    else:  # classification
        if args.mode == 'pathomictext':
            results_val_dict, val_acc, val_auc, val_f1 = summary_classification_pathomictext(model, val_loader, args.n_classes)
        else:
            results_val_dict, val_acc, val_auc, val_f1 = summary_classification(model, val_loader, args.n_classes)
        print('Val Acc: {:.4f}, Val AUC: {:.4f}, Val F1: {:.4f}'.format(val_acc, val_auc, val_f1))
        writer.close()
        return results_val_dict, val_acc, val_auc, val_f1
"""
#####################################################################
"""



def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    #all_risk_scores = np.zeros((len(loader)))
    #all_censorships = np.zeros((len(loader)))
    #all_event_times = np.zeros((len(loader)))
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)
        #pdb.set_trace()
        h = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
        
        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            h_path, h_omic, h_mm = h
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            h = h_mm
        
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        #pdb.set_trace()
        #all_risk_scores[batch_idx] = risk
        #all_censorships[batch_idx] = censor.detach().cpu().item()
        #all_event_times[batch_idx] = event_time
        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        #pdb.set_trace()
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data_WSI.size(0)))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), data_WSI.size(0)))
        
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    #pdb.set_trace()

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            h_path, h_omic, h_mm = h
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            h = h_mm

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic)
        
        if isinstance(h, tuple):
            h = h[2]

        if h.shape[1] > 1:
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        # event_time = np.asscalar(event_time)
        event_time = event_time.item()
        # censor = np.asscalar(censor)
        censor = censor.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': y_disc.item(), 'survival': event_time, 'censorship': censor}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index

def train_loop_survival_pathomictext(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, (data_WSI, data_omic, data_text, y_disc, event_time, censor) in enumerate(loader): # 修改这里
        
        data_WSI, data_omic, data_text = data_WSI.to(device), data_omic.to(device), data_text.to(device) #text
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        h = model(x_path=data_WSI, x_omic=data_omic, x_text=data_text) #text

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            print("utils/core_utils.py:train_loop_survival!!!进入else，len（h）问题!!!")
            if len(h) == 3:  # h 是一个包含 h_path, h_omic, h_mm 的 tuple (多模态模型)
                h_path, h_omic, h_mm = h
                loss = 0.5 * loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()
                h = h_mm
            elif len(h) == 5:  # h 是一个包含 hazards, S, Y_hat, None, None 的 tuple (单模态模型)
                hazards, S, Y_hat, None_val1, None_val2 = h  # 解包 tuple
                loss = loss_fn(h=hazards, y=y_disc, t=event_time, c=censor)  # 使用 hazards 计算 loss
                loss_value = loss.item()
                h = hazards  # h 现在是 hazards
            else:  # h 的长度不是 3 也不是 5， 可能是出现了错误
                raise ValueError(f"Unexpected length of tuple h: {len(h)}.  Expected 3 or 5.")

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data_WSI.size(0)))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), data_WSI.size(0)))
        
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

def validate_survival_pathomictext(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, data_text, y_disc, event_time, censor) in enumerate(loader): # 修改这里
        data_WSI, data_omic, data_text = data_WSI.to(device), data_omic.to(device), data_text.to(device) #text
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic,x_text=data_text) #text

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            print("utils/core_utils.py:validate_survival!!!进入else，len（h）问题!!!")
            if len(h) == 3:  # h 是一个包含 h_path, h_omic, h_mm 的 tuple (多模态模型)
                h_path, h_omic, h_mm = h
                loss = 0.5 * loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()
                h = h_mm
            elif len(h) == 5:  # h 是一个包含 hazards, S, Y_hat, None, None 的 tuple (单模态模型)
                hazards, S, Y_hat, None_val1, None_val2 = h  # 解包 tuple
                loss = loss_fn(h=hazards, y=y_disc, t=event_time, c=censor)  # 使用 hazards 计算 loss
                loss_value = loss.item()
                h = hazards  # h 现在是 hazards
            else:  # h 的长度不是 3 也不是 5， 可能是出现了错误
                raise ValueError(f"Unexpected length of tuple h: {len(h)}.  Expected 3 or 5.")

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_survival_pathomictext(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, data_text, y_disc, event_time, censor) in enumerate(loader): # 修改这里
        data_WSI, data_omic, data_text = data_WSI.to(device), data_omic.to(device), data_text.to(device) #text
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic, x_text=data_text) #text

        # === 关键修改：处理模型输出 ===
        if isinstance(h, tuple):
            if len(h) == 3:  # 多模态输出 (h_path, h_omic, h_mm)
                h = h[0]  # 使用病理分支输出
            elif len(h) == 5:  # 单模态生存输出 (hazards, S, Y_hat, ...)
                h = h[0]  # 取hazards
        
        # 计算风险分数
        if h.dim() > 1 and h.shape[1] > 1:  # 离散生存
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:  # 连续风险
            risk = h.detach().cpu().numpy().squeeze()

        # 收集结果
        all_risk_scores.append(risk)
        all_censorships.append(censor.item())
        all_event_times.append(event_time.item())
        patient_results[slide_id] = {
            'slide_id': slide_id,
            'risk': risk,
            'disc_label': y_disc.item(),
            'survival': event_time.item(),
            'censorship': censor.item()
        }

    # 计算c-index
    c_index = concordance_index_censored(
        (1-np.array(all_censorships)).astype(bool),
        np.array(all_event_times),
        np.concatenate(all_risk_scores),
        tied_tol=1e-08
    )[0]
    
    return patient_results, c_index







"""
### 下面是分类的东西
"""
# 修改后的分类训练函数 - 支持批处理 (保持函数名不变)
def train_loop_classification(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    """分类训练循环 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    
    print('\n')
    print(f"Starting training epoch {epoch}")
    print(f"Loader length: {len(loader)}")
    print(f"Gradient accumulation steps: {gc}")
    print(f"Expected optimizer updates: {len(loader) // gc + (1 if len(loader) % gc != 0 else 0)}")
    
    optimizer_step_count = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 解包批次数据
        if len(batch_data) == 3:  # pathomic模式
            data_WSI_list, data_omic, label = batch_data
        else:
            raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
        
        # 检查数据格式
        if batch_idx == 0:
            print(f"Batch 0 data types:")
            print(f"  data_WSI_list type: {type(data_WSI_list)}")
            if isinstance(data_WSI_list, list):
                print(f"  data_WSI_list length: {len(data_WSI_list)}")
                if len(data_WSI_list) > 0:
                    print(f"  First WSI element shape: {data_WSI_list[0].shape}")
            else:
                print(f"  data_WSI_list shape: {data_WSI_list.shape}")
            print(f"  data_omic shape: {data_omic.shape}")
            print(f"  label shape: {label.shape}")
        
        # 移动到设备
        data_omic, label = data_omic.to(device).float(), label.to(device)
        
        # 处理path数据 - 确保格式正确
        if not isinstance(data_WSI_list, list):
            # 如果不是列表，尝试转换
            if data_WSI_list.dim() == 3:  # [batch_size, n_patches, path_dim]
                data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                if batch_idx == 0:
                    print(f"  Converted tensor to list with {len(data_WSI_list)} elements")
            else:  # [n_patches, path_dim] - 单个样本
                data_WSI_list = [data_WSI_list.to(device).float()]
                if batch_idx == 0:
                    print(f"  Wrapped single tensor as list")
        else:
            # 如果已经是列表，确保每个元素都在正确设备上
            data_WSI_list = [x.to(device).float() for x in data_WSI_list]
        
        # 逐个样本处理 (因为PorpoiseMMF不支持真正的batch处理)
        batch_logits = []
        batch_labels = []
        
        for i in range(len(data_WSI_list)):
            # 准备单个样本的输入
            sample_path = data_WSI_list[i]  # [n_patches, path_dim]
            sample_omic = data_omic[i:i+1]  # [1, omic_dim] 保持批次维度
            sample_label = label[i:i+1]     # [1,] 保持批次维度
            
            # PorpoiseMMF前向传播
            logits = model(x_path=sample_path, x_omic=sample_omic)
            
            # 处理模型输出
            if isinstance(logits, tuple):
                logits = logits[0] if len(logits) > 1 else logits
            
            batch_logits.append(logits)
            batch_labels.append(sample_label)
        
        # 合并批次结果
        combined_logits = torch.cat(batch_logits, dim=0)  # [batch_size, n_classes]
        combined_labels = torch.cat(batch_labels, dim=0)  # [batch_size]
        
        # 计算损失
        loss = loss_fn(combined_logits, combined_labels.squeeze())
        
        if reg_fn is not None:
            loss += reg_fn(model) * lambda_reg
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            batch_size = data_omic.size(0)
            total_patches = sum([x.size(0) for x in data_WSI_list])
            print('batch {}, loss: {:.4f}, label: {}, batch_size: {}, total_patches: {}'.format(
                batch_idx, loss.item(), label.cpu().numpy(), batch_size, total_patches))
        
        # 反向传播
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step_count += 1
            if batch_idx < 100:  # 只在前100个batch打印优化器步骤
                print(f"Optimizer step {optimizer_step_count} at batch {batch_idx + 1}")
    
    # 处理最后的不完整batch
    if len(loader) % gc != 0:
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step_count += 1
        print(f"Final optimizer step {optimizer_step_count} for incomplete batch (remaining: {len(loader) % gc})")
    
    print(f"Total optimizer steps in epoch {epoch}: {optimizer_step_count}")
    
    train_loss /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))
    
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)


def train_loop_classification_pathomictext(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    """pathomictext分类训练循环 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    
    print('\n')
    print(f"Starting training epoch {epoch}")
    print(f"Loader length: {len(loader)}")
    print(f"Gradient accumulation steps: {gc}")
    print(f"Expected optimizer updates: {len(loader) // gc + (1 if len(loader) % gc != 0 else 0)}")
    
    optimizer_step_count = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # 解包批次数据
        if len(batch_data) == 4:  # pathomictext模式
            data_WSI_list, data_omic, data_text, label = batch_data
        else:
            raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
        
        # 检查数据格式
        if batch_idx == 0:
            print(f"Batch 0 data types:")
            print(f"  data_WSI_list type: {type(data_WSI_list)}")
            if isinstance(data_WSI_list, list):
                print(f"  data_WSI_list length: {len(data_WSI_list)}")
                if len(data_WSI_list) > 0:
                    print(f"  First WSI element shape: {data_WSI_list[0].shape}")
            else:
                print(f"  data_WSI_list shape: {data_WSI_list.shape}")
            print(f"  data_omic shape: {data_omic.shape}")
            print(f"  data_text shape: {data_text.shape}")
            print(f"  label shape: {label.shape}")
        
        # 移动到设备
        data_omic, data_text, label = data_omic.to(device).float(), data_text.to(device).float(), label.to(device)
        
        # 处理path数据 - 确保格式正确
        if not isinstance(data_WSI_list, list):
            # 如果不是列表，尝试转换
            if data_WSI_list.dim() == 3:  # [batch_size, n_patches, path_dim]
                data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                if batch_idx == 0:
                    print(f"  Converted tensor to list with {len(data_WSI_list)} elements")
            else:  # [n_patches, path_dim] - 单个样本
                data_WSI_list = [data_WSI_list.to(device).float()]
                if batch_idx == 0:
                    print(f"  Wrapped single tensor as list")
        else:
            # 如果已经是列表，确保每个元素都在正确设备上
            data_WSI_list = [x.to(device).float() for x in data_WSI_list]
        
        # 逐个样本处理 (因为PorpoiseMMF不支持真正的batch处理)
        batch_logits = []
        batch_labels = []
        
        for i in range(len(data_WSI_list)):
            # 准备单个样本的输入
            sample_path = data_WSI_list[i]     # [n_patches, path_dim]
            sample_omic = data_omic[i:i+1]     # [1, omic_dim] 保持批次维度
            sample_text = data_text[i:i+1]     # [1, text_dim] 保持批次维度
            sample_label = label[i:i+1]        # [1,] 保持批次维度
            
            # PorpoiseMMF前向传播
            logits = model(x_path=sample_path, x_omic=sample_omic, x_text=sample_text)
            
            # 处理模型输出
            if isinstance(logits, tuple):
                logits = logits[0] if len(logits) > 1 else logits
            
            batch_logits.append(logits)
            batch_labels.append(sample_label)
        
        # 合并批次结果
        combined_logits = torch.cat(batch_logits, dim=0)  # [batch_size, n_classes]
        combined_labels = torch.cat(batch_labels, dim=0)  # [batch_size]

        # 安全的label处理，避免squeeze将[1]变成标量
        if combined_labels.dim() > 1 and combined_labels.size(-1) == 1:
            processed_labels = combined_labels.view(-1)
        else:
            processed_labels = combined_labels
        
        loss = loss_fn(combined_logits, processed_labels)
        
        if reg_fn is not None:
            loss += reg_fn(model) * lambda_reg
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            batch_size = data_omic.size(0)
            total_patches = sum([x.size(0) for x in data_WSI_list])
            print('batch {}, loss: {:.4f}, label: {}, batch_size: {}, total_patches: {}'.format(
                batch_idx, loss.item(), label.cpu().numpy(), batch_size, total_patches))
        
        # 反向传播
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step_count += 1
            """
            if batch_idx < 100:  # 只在前100个batch打印优化器步骤
                print(f"Optimizer step {optimizer_step_count} at batch {batch_idx + 1}")
            """
    
    # 处理最后的不完整batch
    if len(loader) % gc != 0:
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step_count += 1
        print(f"Final optimizer step {optimizer_step_count} for incomplete batch (remaining: {len(loader) % gc})")
    
    print(f"Total optimizer steps in epoch {epoch}: {optimizer_step_count}")
    
    train_loss /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))
    
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)


def validate_classification(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_acc=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, acc_stopping=None):
    """分类验证函数 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_preds = []
    all_labels = []
    valid_batch_count = 0
    
    print(f"\nValidation epoch {epoch} - checking {len(loader)} batches")
    
    for batch_idx, batch_data in enumerate(loader):
        # 检查批次是否为空或无效
        if not batch_data or len(batch_data) == 0:
            print(f"Warning: Empty batch {batch_idx} in validation")
            continue
            
        # 解包批次数据
        if len(batch_data) == 3:  # pathomic模式
            data_WSI_list, data_omic, label = batch_data
        else:
            print(f"Warning: Unexpected batch data length: {len(batch_data)} at batch {batch_idx}")
            continue
        
        # 检查批次数据是否有效
        if len(data_WSI_list) == 0 or data_omic.size(0) == 0 or label.size(0) == 0:
            print(f"Warning: Invalid batch {batch_idx} - empty data detected")
            continue
        
        try:
            # 移动到设备
            data_omic, label = data_omic.to(device).float(), label.to(device)
            
            # 处理path数据
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:  # [batch_size, n_patches, path_dim]
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:  # [n_patches, path_dim] - 单个样本
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            # 检查处理后的数据是否有效
            if len(data_WSI_list) == 0:
                print(f"Warning: No valid WSI data in batch {batch_idx}")
                continue
            
            # 逐个样本处理验证
            batch_logits = []
            batch_labels = []
            
            with torch.no_grad():
                for i in range(len(data_WSI_list)):
                    # 准备单个样本的输入
                    sample_path = data_WSI_list[i]  # [n_patches, path_dim]
                    sample_omic = data_omic[i:i+1]  # [1, omic_dim] 保持批次维度
                    sample_label = label[i:i+1]     # [1,] 保持批次维度
                    
                    # PorpoiseMMF前向传播
                    logits = model(x_path=sample_path, x_omic=sample_omic)
                    
                    # 处理模型输出
                    if isinstance(logits, tuple):
                        logits = logits[0] if len(logits) > 1 else logits
                    
                    batch_logits.append(logits)
                    batch_labels.append(sample_label)
            
            # 合并批次结果
            combined_logits = torch.cat(batch_logits, dim=0)  # [batch_size, n_classes]
            combined_labels = torch.cat(batch_labels, dim=0)  # [batch_size]
            
            # 计算损失
            loss = loss_fn(combined_logits, combined_labels.squeeze())
            val_loss += loss.item()
            valid_batch_count += 1
            
            # 收集预测和标签
            _, preds = torch.max(combined_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(combined_labels.cpu().numpy())
            
            # 在前几个batch打印详细信息
            if batch_idx < 3:
                print(f"  Val batch {batch_idx}: loss={loss.item():.4f}, "
                      f"batch_size={combined_logits.size(0)}, "
                      f"preds={preds.cpu().numpy()}, "
                      f"labels={combined_labels.cpu().numpy()}")
                
        except Exception as e:
            print(f"Error processing validation batch {batch_idx}: {e}")
            continue
    
    # 检查是否有有效的验证数据
    if valid_batch_count == 0:
        print("Warning: No valid batches processed during validation!")
        return True  # 触发早停
    
    # 计算平均损失和准确率
    val_loss /= valid_batch_count
    
    if len(all_labels) == 0:
        print("Warning: No valid predictions collected during validation!")
        return True
    
    from sklearn.metrics import accuracy_score
    val_acc = accuracy_score(all_labels, all_preds)
    
    print('Epoch: {}, val_loss: {:.4f}, val_acc: {:.4f} ({} valid batches)'.format(
        epoch, val_loss, val_acc, valid_batch_count))
    
    # 打印预测分布统计
    import numpy as np
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print(f"  Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    print(f"  Label distribution: {dict(zip(unique_labels, label_counts))}")
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
    
    # 使用准确率监控最佳模型
    if monitor_acc:
        monitor_acc(val_acc, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))


    if acc_stopping:
        acc_stopping(epoch, val_acc, model, 
                    ckpt_name=os.path.join(results_dir, "s_{}_maxacc_checkpoint.pt".format(cur)))
        if acc_stopping.early_stop:
            print("Early stopping triggered by accuracy!")
            return True
            
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            return True
    
    return False


def validate_classification_pathomictext(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_acc=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, acc_stopping=None):
    """pathomictext分类验证函数 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_preds = []
    all_labels = []
    valid_batch_count = 0
    
    print(f"\nValidation epoch {epoch} - checking {len(loader)} batches")
    
    for batch_idx, batch_data in enumerate(loader):
        # 检查批次是否为空或无效
        if not batch_data or len(batch_data) == 0:
            print(f"Warning: Empty batch {batch_idx} in validation")
            continue
            
        # 解包批次数据
        if len(batch_data) == 4:  # pathomictext模式
            data_WSI_list, data_omic, data_text, label = batch_data
        else:
            print(f"Warning: Unexpected batch data length: {len(batch_data)} at batch {batch_idx}")
            continue
        
        # 检查批次数据是否有效
        if len(data_WSI_list) == 0 or data_omic.size(0) == 0 or label.size(0) == 0:
            print(f"Warning: Invalid batch {batch_idx} - empty data detected")
            continue
        
        try:
            # 移动到设备
            data_omic, data_text, label = data_omic.to(device).float(), data_text.to(device).float(), label.to(device)
            
            # 处理path数据
            if not isinstance(data_WSI_list, list):
                if data_WSI_list.dim() == 3:  # [batch_size, n_patches, path_dim]
                    data_WSI_list = [data_WSI_list[i] for i in range(data_WSI_list.size(0))]
                else:  # [n_patches, path_dim] - 单个样本
                    data_WSI_list = [data_WSI_list.to(device).float()]
            else:
                data_WSI_list = [x.to(device).float() for x in data_WSI_list if x.numel() > 0]
            
            # 检查处理后的数据是否有效
            if len(data_WSI_list) == 0:
                print(f"Warning: No valid WSI data in batch {batch_idx}")
                continue
            
            # 逐个样本处理验证
            batch_logits = []
            batch_labels = []
            
            with torch.no_grad():
                for i in range(len(data_WSI_list)):
                    # 准备单个样本的输入
                    sample_path = data_WSI_list[i]     # [n_patches, path_dim]
                    sample_omic = data_omic[i:i+1]     # [1, omic_dim] 保持批次维度
                    sample_text = data_text[i:i+1]     # [1, text_dim] 保持批次维度
                    sample_label = label[i:i+1]        # [1,] 保持批次维度
                    
                    # PorpoiseMMF前向传播
                    logits = model(x_path=sample_path, x_omic=sample_omic, x_text=sample_text)
                    
                    # 处理模型输出
                    if isinstance(logits, tuple):
                        logits = logits[0] if len(logits) > 1 else logits
                    
                    batch_logits.append(logits)
                    batch_labels.append(sample_label)
            
            # 合并批次结果
            combined_logits = torch.cat(batch_logits, dim=0)  # [batch_size, n_classes]
            combined_labels = torch.cat(batch_labels, dim=0)  # [batch_size]
            
            # 计算损失
            loss = loss_fn(combined_logits, combined_labels.squeeze())
            val_loss += loss.item()
            valid_batch_count += 1
            
            # 收集预测和标签
            _, preds = torch.max(combined_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(combined_labels.cpu().numpy())
            
            # 在前几个batch打印详细信息
            if batch_idx < 3:
                print(f"  Val batch {batch_idx}: loss={loss.item():.4f}, "
                      f"batch_size={combined_logits.size(0)}, "
                      f"preds={preds.cpu().numpy()}, "
                      f"labels={combined_labels.cpu().numpy()}")
                
        except Exception as e:
            print(f"Error processing validation batch {batch_idx}: {e}")
            continue
    
    # 检查是否有有效的验证数据
    if valid_batch_count == 0:
        print("Warning: No valid batches processed during validation!")
        return True  # 触发早停
    
    # 计算平均损失和准确率
    val_loss /= valid_batch_count
    
    if len(all_labels) == 0:
        print("Warning: No valid predictions collected during validation!")
        return True
    
    from sklearn.metrics import accuracy_score
    val_acc = accuracy_score(all_labels, all_preds)
    
    print('Epoch: {}, val_loss: {:.4f}, val_acc: {:.4f} ({} valid batches)'.format(
        epoch, val_loss, val_acc, valid_batch_count))
    
    # 打印预测分布统计
    import numpy as np
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print(f"  Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    print(f"  Label distribution: {dict(zip(unique_labels, label_counts))}")
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
    
    # 使用准确率监控最佳模型
    if monitor_acc:
        monitor_acc(val_acc, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            return True

    if acc_stopping:
        acc_stopping(epoch, val_acc, model, 
                    ckpt_name=os.path.join(results_dir, "s_{}_maxacc_checkpoint.pt".format(cur)))
        if acc_stopping.early_stop:
            print("Early stopping triggered by accuracy!")
            return True
    
    return False


def summary_classification(model, loader, n_classes):
    """分类总结函数 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    for batch_idx, batch_data in enumerate(loader):
        # 解包批次数据
        if len(batch_data) == 3:  # pathomic模式
            data_WSI_list, data_omic, label = batch_data
        else:
            raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
        
        # 移动到设备
        data_omic, label = data_omic.to(device).float(), label.to(device)
        
        # 处理path数据
        if not isinstance(data_WSI_list, list):
            data_WSI_list = [data_WSI_list.to(device).float()]
        else:
            data_WSI_list = [x.to(device).float() for x in data_WSI_list]
        
        # 逐个样本处理
        with torch.no_grad():
            for i in range(len(data_WSI_list)):
                # 准备单个样本的输入
                sample_path = data_WSI_list[i]  # [n_patches, path_dim]
                sample_omic = data_omic[i:i+1]  # [1, omic_dim] 保持批次维度
                sample_label = label[i] if label.dim() > 0 else label  # 单个标签
                
                # 获取对应的slide_id
                if len(data_WSI_list) == 1:
                    slide_id = slide_ids.iloc[batch_idx]
                else:
                    slide_id = f"{slide_ids.iloc[batch_idx]}_{i}"
                
                # PorpoiseMMF前向传播
                logits = model(x_path=sample_path, x_omic=sample_omic)
                
                if isinstance(logits, tuple):
                    logits = logits[0] if len(logits) > 1 else logits
                
                # 确保维度正确
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probs = F.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)
                
                pred_val = pred.item()
                label_val = sample_label.item() if hasattr(sample_label, 'item') else sample_label
                prob_val = probs.cpu().numpy()[0]
                
                all_preds.append(pred_val)
                all_labels.append(label_val)
                all_probs.append(prob_val)
                
                patient_results[slide_id] = {
                    'slide_id': slide_id,
                    'Y': label_val,
                    'Y_hat': pred_val,
                    'p': prob_val
                }
    
    # 计算指标
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    else:
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
        except:
            auc = 0.0
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return patient_results, acc, auc, f1


def summary_classification_pathomictext(model, loader, n_classes):
    """pathomictext分类总结函数 - 支持批处理 (PorpoiseMMF版本)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    for batch_idx, batch_data in enumerate(loader):
        # 解包批次数据
        if len(batch_data) == 4:  # pathomictext模式
            data_WSI_list, data_omic, data_text, label = batch_data
        else:
            raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
        
        # 移动到设备
        data_omic, data_text, label = data_omic.to(device).float(), data_text.to(device).float(), label.to(device)
        
        # 处理path数据
        if not isinstance(data_WSI_list, list):
            data_WSI_list = [data_WSI_list.to(device).float()]
        else:
            data_WSI_list = [x.to(device).float() for x in data_WSI_list]
        
        # 逐个样本处理
        with torch.no_grad():
            for i in range(len(data_WSI_list)):
                # 准备单个样本的输入
                sample_path = data_WSI_list[i]     # [n_patches, path_dim]
                sample_omic = data_omic[i:i+1]     # [1, omic_dim] 保持批次维度
                sample_text = data_text[i:i+1]     # [1, text_dim] 保持批次维度
                sample_label = label[i] if label.dim() > 0 else label  # 单个标签
                
                # 获取对应的slide_id
                if len(data_WSI_list) == 1:
                    slide_id = slide_ids.iloc[batch_idx]
                else:
                    slide_id = f"{slide_ids.iloc[batch_idx]}_{i}"
                
                # PorpoiseMMF前向传播
                logits = model(x_path=sample_path, x_omic=sample_omic, x_text=sample_text)
                
                if isinstance(logits, tuple):
                    logits = logits[0] if len(logits) > 1 else logits
                
                # 确保维度正确
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probs = F.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)
                
                pred_val = pred.item()
                label_val = sample_label.item() if hasattr(sample_label, 'item') else sample_label
                prob_val = probs.cpu().numpy()[0]
                
                all_preds.append(pred_val)
                all_labels.append(label_val)
                all_probs.append(prob_val)
                
                patient_results[slide_id] = {
                    'slide_id': slide_id,
                    'Y': label_val,
                    'Y_hat': pred_val,
                    'p': prob_val
                }
    
    # 计算指标
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    else:
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
        except:
            auc = 0.0
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return patient_results, acc, auc, f1      

"""
"""