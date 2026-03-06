"""
utils/interpretability/
=======================
独立面板式可解释性模块

每个面板：
  - 只接收 numpy 数组 / Python 原生类型
  - 只负责保存一张 PNG
  - 面板之间零依赖

目录结构：
  data_extractor.py      从模型输出提取原始数据（唯一接触 Tensor 的地方）
  panel_wsi_heatmap.py   面板1：WSI 注意力热力图
  panel_fusion_weights.py面板2：多模态融合权重柱状图
  panel_omic_pathways.py 面板3：Omic 通路重要性
  panel_qa_pathway.py    面板4：QA→Pathway 注意力热力图
  panel_qa_patch.py      面板5：QA→Patch 空间注意力
  panel_patch_pathway.py 面板6：Patch↔Pathway 对应热力图
  runner.py              统一调度入口（每面板独立 try/except）
"""

from utils.interpretability import (
    panel_wsi_heatmap,
    panel_fusion_weights,
    panel_omic_pathways,
    panel_qa_pathway,
    panel_qa_patch,
    panel_patch_pathway,
)
from utils.interpretability.data_extractor import InterpretabilityDataExtractor, ExtractedData
from utils.interpretability.runner import InterpretabilityRunner

__all__ = [
    "InterpretabilityRunner",
    "InterpretabilityDataExtractor",
    "ExtractedData",
    "panel_wsi_heatmap",
    "panel_fusion_weights",
    "panel_omic_pathways",
    "panel_qa_pathway",
    "panel_qa_patch",
    "panel_patch_pathway",
]
