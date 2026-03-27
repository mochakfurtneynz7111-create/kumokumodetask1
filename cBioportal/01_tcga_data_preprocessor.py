#!/usr/bin/env python3
"""
TCGA多组学数据预处理脚本 - 支持混合数据源
可以从Pub版本获取临床数据，从PanCancer Atlas获取RNA-seq数据
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats

class TCGADataPreprocessor:
    def __init__(self, tcga_type, use_pub_for_clinical=True):
        """
        初始化TCGA数据预处理器 - 支持混合数据源
        
        Args:
            tcga_type (str): TCGA癌症类型，如 "BRCA"
            use_pub_for_clinical (bool): 是否使用Pub版本的临床数据
        """
        self.tcga_type = tcga_type
        self.base_path = Path(tcga_type)
        
        # 数据源路径
        self.pub_path = self.base_path / f"{tcga_type.lower()}_tcga_pub"
        self.pancan_path = self.base_path / f"{tcga_type.lower()}_tcga_pan_can_atlas_2018"
        
        # 根据设置选择数据源
        self.clinical_path = self.pub_path if use_pub_for_clinical else self.pancan_path
        self.omics_path = self.pancan_path  # 组学数据优先使用PanCancer Atlas
        
        self.pt_folder = Path(f"../PORPOISE/dataroot/tcga_{tcga_type.lower()}_20x_features/pt_files")
        
        # 确保输出目录存在
        self.base_path.mkdir(exist_ok=True)
        
        print(f"\n数据源配置:")
        print(f"  ✓ 临床数据来源: {self.clinical_path}")
        print(f"  ✓ 组学数据来源: {self.omics_path}")
        
    def process_clinical_data(self):
        """处理临床数据：从原始txt转换为标准化csv格式"""
        print(f"\n=== 处理 {self.tcga_type} 临床数据 ===")
        
        input_file = self.clinical_path / "data_clinical_patient.txt"
        
        try:
            # 读取临床数据文件
            clinical_data = pd.read_csv(
                input_file, 
                sep="\t", 
                comment="#", 
                na_values=['NA', 'NaN', '[Not Available]', '']
            )
            
            print(f"原始临床数据: {clinical_data.shape}")
            
            # 列名映射
            column_mapping = {
                'PATIENT_ID': 'case_id',
                'SEX': 'is_female', 
                'AGE': 'age',
                'OS_MONTHS': 'survival_months',
                'OS_STATUS': 'censorship'
            }
            clinical_data = clinical_data.rename(columns=column_mapping)
            
            # 数据转换
            # 性别转换 (Male -> 0, Female -> 1)
            if 'is_female' in clinical_data.columns:
                clinical_data['is_female'] = clinical_data['is_female'].map({'Male': 0, 'Female': 1})
            
            # 生存状态处理
            if 'censorship' in clinical_data.columns:
                clinical_data = clinical_data.dropna(subset=['censorship'])
                clinical_data['censorship'] = (
                    clinical_data['censorship']
                    .str.split(':').str[0]
                    .astype(int)
                    .map({1: 0, 0: 1})  # 反转：1→0，0→1
                )
            
            # 设置oncotree_code
            clinical_data['oncotree_code'] = self.tcga_type
            
            # 筛选所需列并删除空值
            desired_columns = ['case_id', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship']
            clinical_data = clinical_data[desired_columns].dropna()
            
            print(f"清洗后临床数据: {clinical_data.shape}")
            return clinical_data
            
        except Exception as e:
            print(f"处理临床数据时出错: {e}")
            return None
    
    def add_slide_info(self, clinical_data):
        """为临床数据添加slide信息"""
        print(f"\n=== 添加 {self.tcga_type} Slide信息 ===")
        
        try:
            if not self.pt_folder.exists():
                print(f"警告: PT文件夹不存在 {self.pt_folder}")
                return clinical_data
            
            # 获取pt文件列表
            pt_files = [f.name for f in self.pt_folder.glob("*.pt")]
            print(f"发现 {len(pt_files)} 个pt文件")
            
            # 创建文件名映射字典
            pt_case_map = {}
            for f in pt_files:
                case_id = f[:12]
                if case_id not in pt_case_map:
                    pt_case_map[case_id] = []
                pt_case_map[case_id].append(f)
            
            # 匹配临床数据
            matched_data = []
            for _, row in clinical_data.iterrows():
                case_id = row['case_id']
                if case_id in pt_case_map:
                    for pt_file in pt_case_map[case_id]:
                        new_row = row.to_dict()
                        new_row['slide_id'] = pt_file.replace('.pt', '.svs')
                        new_row['site'] = case_id[5:7]
                        matched_data.append(new_row)
            
            if matched_data:
                result_df = pd.DataFrame(matched_data)
                cols = ['case_id', 'slide_id', 'site'] + \
                      [c for c in result_df.columns if c not in ['case_id', 'slide_id', 'site']]
                result_df = result_df[cols]
                print(f"匹配后数据: {result_df.shape}")
                return result_df
            else:
                print("警告：未找到匹配的slide信息，返回原始临床数据")
                return clinical_data
                
        except Exception as e:
            print(f"添加slide信息时出错: {e}")
            return clinical_data
    
    def process_cna_data(self):
        """处理拷贝数变异(CNA)数据 - 优先使用PanCancer Atlas"""
        print(f"\n=== 处理 {self.tcga_type} CNA数据 ===")
        
        try:
            # 1. 筛选高频CNA基因
            cna_genes_file = self.base_path / f"{self.tcga_type}_CNA_Genes.txt"
            if not cna_genes_file.exists():
                print(f"CNA基因文件不存在: {cna_genes_file}")
                print(f"尝试不筛选基因，使用所有CNA数据...")
                high_freq_genes = None
            else:
                cna_genes = pd.read_csv(cna_genes_file, sep='\t')
                cna_genes['Freq'] = cna_genes['Freq'].str.replace('%', '').astype(float)
                high_freq_genes = cna_genes[cna_genes['Freq'] >= 10]['Gene'].tolist()
                print(f"筛选出 {len(high_freq_genes)} 个高频CNA基因 (≥10%)")
            
            # 2. 读取CNA数据 - 优先使用PanCancer Atlas
            data_cna_file = self.omics_path / "data_cna.txt"
            if not data_cna_file.exists():
                data_cna_file = self.clinical_path / "data_cna.txt"
                print(f"使用备用CNA文件: {data_cna_file}")
            
            data_cna = pd.read_csv(data_cna_file, sep='\t')
            
            # 筛选高频基因（如果有基因列表）
            if high_freq_genes:
                data_cna_filtered = data_cna[data_cna['Hugo_Symbol'].isin(high_freq_genes)]
            else:
                data_cna_filtered = data_cna
            
            # 转置数据
            data_cna_t = data_cna_filtered.set_index('Hugo_Symbol').drop(columns=['Entrez_Gene_Id'], errors='ignore').T
            data_cna_t.index = data_cna_t.index.str[:12]  # 标准化样本名
            
            # 修改列名格式为 xxx_cnv
            data_cna_t.columns = [f"{gene}_cnv" for gene in data_cna_t.columns]
            
            print(f"处理后CNA数据: {data_cna_t.shape}")
            return data_cna_t
            
        except Exception as e:
            print(f"处理CNA数据时出错: {e}")
            return None
    
    def process_mutation_data(self):
        """处理突变数据 - 优先使用PanCancer Atlas"""
        print(f"\n=== 处理 {self.tcga_type} 突变数据 ===")
        
        try:
            # 尝试PanCancer Atlas的突变文件
            input_maf = self.omics_path / "data_mutations.txt"
            if not input_maf.exists():
                input_maf = self.clinical_path / "data_mutations.txt"
                print(f"使用备用突变文件: {input_maf}")
            
            if not input_maf.exists():
                print(f"突变文件不存在: {input_maf}")
                return None
                
            maf_df = pd.read_csv(input_maf, sep="\t", comment="#", low_memory=False)
            
            # 创建突变矩阵
            all_genes = maf_df["Hugo_Symbol"].unique()
            all_samples = maf_df["Tumor_Sample_Barcode"].unique()
            mut_matrix = pd.DataFrame(0, index=all_samples, columns=all_genes)
            
            # 标记突变
            for _, row in maf_df.iterrows():
                sample = row["Tumor_Sample_Barcode"]
                gene = row["Hugo_Symbol"]
                mut_matrix.at[sample, gene] = 1
            
            print(f"原始突变矩阵: {mut_matrix.shape}")
            
            # 2. 筛选高频突变基因
            mut_genes_file = self.base_path / f"{self.tcga_type}_Mutated_Genes.txt"
            if mut_genes_file.exists():
                mut_genes = pd.read_csv(mut_genes_file, sep='\t')
                mut_genes['Freq'] = mut_genes['Freq'].str.replace('%', '').astype(float)
                high_freq_mut_genes = mut_genes[mut_genes['Freq'] >= 5]['Gene'].tolist()
                
                # 获取实际存在的基因
                available_genes = [gene for gene in high_freq_mut_genes if gene in mut_matrix.columns]
                mut_matrix = mut_matrix[available_genes]
                print(f"筛选出 {len(available_genes)} 个高频突变基因 (≥5%)")
            
            # 标准化样本名
            mut_matrix.index = mut_matrix.index.str[:12]
            
            # 修改列名格式为 xxx_mut
            mut_matrix.columns = [f"{gene}_mut" for gene in mut_matrix.columns]
            
            print(f"处理后突变数据: {mut_matrix.shape}")
            return mut_matrix
            
        except Exception as e:
            print(f"处理突变数据时出错: {e}")
            return None
    
    def process_mrna_data(self, top_k=2000):
        """处理mRNA表达数据 - 强制使用PanCancer Atlas的RNA-seq数据"""
        print(f"\n=== 处理 {self.tcga_type} mRNA数据 (筛选top {top_k}特征) ===")
        
        try:
            # 强制使用PanCancer Atlas的RNA-seq数据
            mrna_file = self.pancan_path / "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
            
            if not mrna_file.exists():
                print(f"❌ RNA-seq文件不存在: {mrna_file}")
                print(f"请确保已下载PanCancer Atlas数据集到: {self.pancan_path}")
                return None
            
            print(f"✓ 使用RNA-seq数据: {mrna_file}")
            
            # 读取并修复列名
            with open(mrna_file) as f:
                header = f.readline().strip().split('\t')
                if not header[0].startswith('Hugo'):
                    header = ['Hugo_Symbol'] + header[1:]
            
            data_mrna = pd.read_csv(mrna_file, sep='\t', header=0, names=header)
            
            # 删除基因名为空的行
            data_mrna = data_mrna.dropna(subset=['Hugo_Symbol'])
            print(f"原始mRNA数据: {data_mrna.shape}")
            
            # 转置数据
            data_mrna_t = data_mrna.set_index('Hugo_Symbol').T
            
            # 删除包含NaN的基因列
            data_mrna_t = data_mrna_t.dropna(axis=1)
            
            # 标准化样本名
            data_mrna_t.index = data_mrna_t.index.str[:12]
            
            print(f"清理后mRNA数据: {data_mrna_t.shape}")
            print(f"开始筛选top {top_k}个最具变异性的特征...")
            
            # === MAD筛选top K特征 ===
            if data_mrna_t.shape[1] > top_k:
                # 标准化数据
                scaler = StandardScaler()
                mrna_scaled = scaler.fit_transform(data_mrna_t)
                mrna_scaled_df = pd.DataFrame(mrna_scaled, columns=data_mrna_t.columns, index=data_mrna_t.index)
                
                # 计算MAD (Median Absolute Deviation)
                mad = stats.median_abs_deviation(mrna_scaled_df, axis=0, scale='normal')
                
                # 选择MAD最大的top_k个特征
                top_k_indices = np.argsort(mad)[-top_k:]
                selected_features = [data_mrna_t.columns[i] for i in top_k_indices]
                
                # 提取选定的特征（使用标准化后的数据）
                final_mrna_data = mrna_scaled_df[selected_features]
                
                print(f"MAD筛选完成: {len(data_mrna_t.columns)} → {len(selected_features)} 个特征")
            else:
                print(f"原始特征数({data_mrna_t.shape[1]})少于{top_k}，保留所有特征")
                # 仍然进行标准化
                scaler = StandardScaler()
                mrna_scaled = scaler.fit_transform(data_mrna_t)
                final_mrna_data = pd.DataFrame(mrna_scaled, columns=data_mrna_t.columns, index=data_mrna_t.index)
            
            # 修改列名格式为 xxx_rnaseq
            final_mrna_data.columns = [f"{gene.split('_')[0]}_rnaseq" for gene in final_mrna_data.columns]
            
            print(f"✅ 最终mRNA数据: {final_mrna_data.shape}")
            return final_mrna_data
            
        except Exception as e:
            print(f"❌ 处理mRNA数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_processed_data(self, clinical_data, cna_data, mut_data, mrna_data):
        """保存所有处理后的数据"""
        print(f"\n=== 保存 {self.tcga_type} 处理后数据 ===")
        
        try:
            # 保存临床数据
            if clinical_data is not None:
                clinical_file = self.base_path / "clinical_data_processed.csv"
                clinical_data.to_csv(clinical_file, index=False)
                print(f"临床数据已保存: {clinical_file}")
            
            # 保存组学数据
            for data, name in [(cna_data, "cna"), (mut_data, "mutation"), (mrna_data, "mrna")]:
                if data is not None:
                    output_file = self.base_path / f"{name}_data_processed.csv"
                    data.to_csv(output_file)
                    print(f"{name.upper()}数据已保存: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            return False
    
    def run_preprocessing(self, top_k_rna=2000):
        """运行完整的数据预处理流程"""
        print(f"\n{'='*50}")
        print(f"开始处理 {self.tcga_type} 数据（混合数据源模式）")
        print(f"{'='*50}")
        
        # 1. 处理临床数据（从Pub版本）
        clinical_data = self.process_clinical_data()
        if clinical_data is not None:
            clinical_data = self.add_slide_info(clinical_data)
        
        # 2. 处理组学数据（优先从PanCancer Atlas）
        cna_data = self.process_cna_data()
        mut_data = self.process_mutation_data()
        mrna_data = self.process_mrna_data(top_k=top_k_rna)
        
        # 3. 保存处理后的数据
        success = self.save_processed_data(clinical_data, cna_data, mut_data, mrna_data)
        
        if success:
            print(f"\n✅ {self.tcga_type} 数据预处理完成！")
            print(f"所有处理后的数据已保存到 {self.base_path} 目录")
            print(f"\n数据来源总结:")
            print(f"  - 临床数据: {self.clinical_path.name}")
            print(f"  - RNA-seq数据: {self.pancan_path.name}")
            if mrna_data is not None:
                print(f"  - RNA-seq特征已筛选至top {top_k_rna}个")
        else:
            print(f"\n❌ {self.tcga_type} 数据预处理过程中出现错误")
        
        return {
            'clinical': clinical_data,
            'cna': cna_data, 
            'mutation': mut_data,
            'mrna': mrna_data
        }


def main():
    """主函数"""
    # 配置参数
    TCGA_TYPE = "BRCA"
    TOP_K_RNA = 2000
    
    print("="*60)
    print("TCGA数据预处理 - 混合数据源模式")
    print("="*60)
    print(f"使用场景: Pub版本有完整临床数据，PanCancer Atlas有RNA-seq")
    print(f"策略: 临床数据用Pub，RNA-seq用PanCancer Atlas")
    print("="*60)
    
    # 创建预处理器并运行
    preprocessor = TCGADataPreprocessor(TCGA_TYPE, use_pub_for_clinical=True)
    results = preprocessor.run_preprocessing(top_k_rna=TOP_K_RNA)
    
    # 显示处理结果摘要
    print(f"\n{'='*50}")
    print(f"处理结果摘要")
    print(f"{'='*50}")
    for data_type, data in results.items():
        if data is not None:
            print(f"{data_type.upper()}: {data.shape}")
            if data_type != 'clinical' and hasattr(data, 'columns') and len(data.columns) > 0:
                print(f"  列名示例: {list(data.columns[:3])}")
        else:
            print(f"{data_type.upper()}: 处理失败 ❌")


if __name__ == "__main__":
    main()