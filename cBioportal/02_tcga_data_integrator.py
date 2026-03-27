#!/usr/bin/env python3
"""
TCGA多组学数据整合脚本
读取预处理后的数据，进行质量检查和合并
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TCGADataIntegrator:
    def __init__(self, tcga_type):
        """
        初始化TCGA数据整合器
        
        Args:
            tcga_type (str): TCGA癌症类型，如 "HNSC", "LUSC", "BRCA" 等
        """
        self.tcga_type = tcga_type
        self.base_path = Path(tcga_type)
        
        # 数据文件路径
        self.data_files = {
            'clinical': self.base_path / "clinical_data_processed.csv",
            'cna': self.base_path / "cna_data_processed.csv", 
            'mutation': self.base_path / "mutation_data_processed.csv",
            'mrna': self.base_path / "mrna_data_processed.csv"
        }
        
    def load_processed_data(self):
        """加载所有预处理后的数据"""
        print(f"\n=== 加载 {self.tcga_type} 预处理数据 ===")
        
        loaded_data = {}
        
        # 加载临床数据
        if self.data_files['clinical'].exists():
            clinical = pd.read_csv(self.data_files['clinical'])
            # 创建sample_id用于匹配
            clinical['sample_id'] = clinical['case_id'].str[:12]
            clinical = clinical.drop_duplicates('sample_id').set_index('sample_id')
            loaded_data['clinical'] = clinical
            print(f"临床数据: {clinical.shape}")
        else:
            print("警告: 临床数据文件不存在")
            
        # 加载组学数据
        for data_type in ['cna', 'mutation', 'mrna']:
            file_path = self.data_files[data_type]
            if file_path.exists():
                data = pd.read_csv(file_path, index_col=0)
                # 标准化样本ID
                data.index = data.index.str[:12]
                # 去除重复样本（保留第一个）
                data = data[~data.index.duplicated(keep='first')]
                loaded_data[data_type] = data
                print(f"{data_type.upper()}数据: {data.shape}")
            else:
                print(f"警告: {data_type.upper()}数据文件不存在")
                loaded_data[data_type] = None
        
        return loaded_data
    
    def find_common_samples(self, data_dict):
        """找到所有数据类型的共同样本"""
        print(f"\n=== 寻找 {self.tcga_type} 共同样本 ===")
        
        sample_sets = []
        
        # 收集所有数据类型的样本集合
        for data_type, data in data_dict.items():
            if data is not None:
                samples = set(data.index)
                sample_sets.append((data_type, samples))
                print(f"{data_type.upper()}样本数: {len(samples)}")
        
        if not sample_sets:
            print("错误: 没有可用的数据")
            return set()
        
        # 找交集
        common_samples = sample_sets[0][1]
        for data_type, samples in sample_sets[1:]:
            common_samples = common_samples & samples
            print(f"与{data_type.upper()}的交集: {len(common_samples)}")
        
        print(f"\n最终共同样本数: {len(common_samples)}")
        return common_samples
    
    def quality_check(self, data_dict, common_samples):
        """对数据进行质量检查"""
        print(f"\n=== {self.tcga_type} 数据质量检查 ===")
        
        quality_report = {}
        
        for data_type, data in data_dict.items():
            if data is None:
                continue
                
            # 筛选共同样本
            data_filtered = data.loc[list(common_samples)]
            
            # 基本统计
            report = {
                'samples': len(data_filtered),
                'features': len(data_filtered.columns) if len(data_filtered.shape) > 1 else 0,
                'missing_values': data_filtered.isnull().sum().sum() if len(data_filtered.shape) > 1 else data_filtered.isnull().sum(),
                'missing_percentage': (data_filtered.isnull().sum().sum() / data_filtered.size * 100) if len(data_filtered.shape) > 1 else (data_filtered.isnull().sum() / len(data_filtered) * 100)
            }
            
            quality_report[data_type] = report
            
            print(f"\n{data_type.upper()}数据质量:")
            print(f"  样本数: {report['samples']}")
            if data_type != 'clinical':
                print(f"  特征数: {report['features']}")
            print(f"  缺失值: {report['missing_values']}")
            print(f"  缺失率: {report['missing_percentage']:.2f}%")
        
        return quality_report
    
    def create_integrated_dataset(self, data_dict, common_samples):
        """创建整合数据集"""
        print(f"\n=== 创建 {self.tcga_type} 整合数据集 ===")
        
        # 确保样本顺序一致
        common_samples_sorted = sorted(list(common_samples))
        
        # 准备合并的数据框列表
        data_to_merge = []
        
        # 添加临床数据
        if 'clinical' in data_dict and data_dict['clinical'] is not None:
            clinical_filtered = data_dict['clinical'].loc[common_samples_sorted]
            data_to_merge.append(clinical_filtered)
            print(f"添加临床数据: {clinical_filtered.shape}")
        
        # 添加组学数据
        for data_type in ['cna', 'mutation', 'mrna']:
            if data_type in data_dict and data_dict[data_type] is not None:
                data_filtered = data_dict[data_type].loc[common_samples_sorted]
                data_to_merge.append(data_filtered)
                print(f"添加{data_type.upper()}数据: {data_filtered.shape}")
        
        # 合并所有数据
        if data_to_merge:
            integrated_data = pd.concat(data_to_merge, axis=1)
            print(f"\n整合后数据维度: {integrated_data.shape}")
            
            # 检查是否有重复列名
            duplicate_cols = integrated_data.columns[integrated_data.columns.duplicated()]
            if len(duplicate_cols) > 0:
                print(f"警告: 发现重复列名: {duplicate_cols.tolist()}")
                # 删除重复列（保留第一个）
                integrated_data = integrated_data.loc[:, ~integrated_data.columns.duplicated()]
                print(f"删除重复列后维度: {integrated_data.shape}")
            
            return integrated_data
        else:
            print("错误: 没有数据可以合并")
            return None
    
    def generate_data_summary(self, integrated_data):
        """生成数据摘要报告"""
        print(f"\n=== 生成 {self.tcga_type} 数据摘要 ===")
        
        if integrated_data is None:
            return None
            
        summary = {
            'dataset_info': {
                'tcga_type': self.tcga_type,
                'total_samples': len(integrated_data),
                'total_features': len(integrated_data.columns),
                'missing_values': integrated_data.isnull().sum().sum(),
                'missing_percentage': integrated_data.isnull().sum().sum() / integrated_data.size * 100
            },
            'feature_types': {}
        }
        
        # 分析不同类型的特征
        clinical_cols = [col for col in integrated_data.columns 
                        if not (col.endswith('_cnv') or col.endswith('_rnaseq') or col.endswith('_mut'))]
        
        cnv_cols = [col for col in integrated_data.columns if col.endswith('_cnv')]
        mut_cols = [col for col in integrated_data.columns if col.endswith('_mut')]
        rnaseq_cols = [col for col in integrated_data.columns if col.endswith('_rnaseq')]
        
        summary['feature_types'] = {
            'clinical': len(clinical_cols),
            'copy_number': len(cnv_cols), 
            'mutation': len(mut_cols),
            'rna_expression': len(rnaseq_cols)
        }
        
        # 打印摘要
        print(f"数据集信息:")
        print(f"  癌症类型: {summary['dataset_info']['tcga_type']}")
        print(f"  样本数: {summary['dataset_info']['total_samples']}")
        print(f"  特征数: {summary['dataset_info']['total_features']}")
        print(f"  缺失值: {summary['dataset_info']['missing_values']}")
        print(f"  缺失率: {summary['dataset_info']['missing_percentage']:.2f}%")
        
        print(f"\n特征类型分布:")
        for feature_type, count in summary['feature_types'].items():
            print(f"  {feature_type}: {count}")
            
        return summary
    
    def save_integrated_data(self, integrated_data, summary=None):
        """保存整合数据和摘要报告"""
        print(f"\n=== 保存 {self.tcga_type} 整合数据 ===")
        
        if integrated_data is None:
            print("错误: 没有数据可以保存")
            return False
            
        try:
            # 保存整合数据
            output_file = self.base_path / f"{self.tcga_type}_integrated_omics_data.csv"
            integrated_data.to_csv(output_file)
            print(f"整合数据已保存: {output_file}")
            
            # 保存摘要报告
            if summary:
                summary_file = self.base_path / f"{self.tcga_type}_data_summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"TCGA {self.tcga_type} 数据摘要报告\n")
                    f.write("=" * 50 + "\n\n")
                    
                    f.write("数据集信息:\n")
                    for key, value in summary['dataset_info'].items():
                        f.write(f"  {key}: {value}\n")
                    
                    f.write("\n特征类型分布:\n")
                    for key, value in summary['feature_types'].items():
                        f.write(f"  {key}: {value}\n")
                        
                print(f"摘要报告已保存: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            return False
    
    def run_integration(self):
        """运行完整的数据整合流程"""
        print(f"\n{'='*60}")
        print(f"开始整合 {self.tcga_type} 多组学数据")
        print(f"{'='*60}")
        
        # 1. 加载预处理数据
        data_dict = self.load_processed_data()
        
        if not data_dict:
            print("错误: 没有找到预处理数据")
            return None
        
        # 2. 找到共同样本
        common_samples = self.find_common_samples(data_dict)
        
        if len(common_samples) == 0:
            print("错误: 没有找到共同样本")
            return None
        
        # 3. 质量检查
        quality_report = self.quality_check(data_dict, common_samples)
        
        # 4. 创建整合数据集
        integrated_data = self.create_integrated_dataset(data_dict, common_samples)
        
        # 5. 生成摘要报告
        summary = self.generate_data_summary(integrated_data)
        
        # 6. 保存结果
        success = self.save_integrated_data(integrated_data, summary)
        
        if success:
            print(f"\n✅ {self.tcga_type} 数据整合完成！")
            print(f"整合数据已保存到 {self.base_path} 目录")
        else:
            print(f"\n❌ {self.tcga_type} 数据整合过程中出现错误")
        
        return integrated_data


def main():
    """主函数"""
    # 在这里修改TCGA类型（与第一个脚本保持一致）
    TCGA_TYPE = "BRCA"  # 例如: "HNSC", "LUSC", "BRCA" 等
    
    # 创建整合器并运行
    integrator = TCGADataIntegrator(TCGA_TYPE)
    integrated_data = integrator.run_integration()
    
    if integrated_data is not None:
        print(f"\n{'='*60}")
        print(f"整合完成！最终数据集维度: {integrated_data.shape}")
        print(f"{'='*60}")
        
        # 显示前几列的信息
        print("\n数据集前5列信息:")
        print(integrated_data.iloc[:, :5].info())


if __name__ == "__main__":
    main()