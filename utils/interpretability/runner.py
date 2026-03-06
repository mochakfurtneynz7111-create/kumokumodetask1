"""
runner.py
=========
可解释性面板独立运行器

职责：
  1. 调用 InterpretabilityDataExtractor 提取数据
  2. 独立调用每个面板的 save_panel()
  3. 每个面板的错误不影响其他面板
  4. 输出目录结构：
       {save_dir}/{sample_id}/
           panel1_wsi_heatmap.png
           panel2_fusion_weights.png
           panel3_omic_pathways.png
           panel4_qa_pathway.png
           panel5_qa_patch.png
           panel6_patch_pathway.png

使用示例：
    from utils.interpretability.runner import InterpretabilityRunner

    runner = InterpretabilityRunner(
        model=model,
        save_dir="./results/interpretability",
        device="cuda",
        preprocessing_dir="./outputs/preprocessing",
    )

    runner.run(
        data_WSI=data_WSI,
        data_omic=data_omic,
        data_text=data_text,
        slide_id="TCGA-AB-1234.svs",
        sample_id="fold0_e10_s0",
        true_label=2,
        class_names=["LUAD", "LUSC", "HNSC", "STAD"],
        pathway_names=pathway_names,   # list[str], 可选
        qa_texts=qa_texts,             # list[str], 可选
        survival_time=24.5,
        event=1,
    )
"""

import os
import traceback

from utils.interpretability.data_extractor import InterpretabilityDataExtractor

from utils.interpretability import (
    panel_wsi_heatmap,
    panel_fusion_weights,
    panel_omic_pathways,
    panel_qa_pathway,
    panel_qa_patch,
    panel_patch_pathway,
)


class InterpretabilityRunner:

    # 面板配置：(模块, 文件名, 描述)
    _PANELS = [
        (panel_wsi_heatmap,    "panel1_wsi_heatmap.png",     "WSI Heatmap"),
        (panel_fusion_weights, "panel2_fusion_weights.png",  "Fusion Weights"),
        (panel_omic_pathways,  "panel3_omic_pathways.png",   "Omic Pathways"),
        (panel_qa_pathway,     "panel4_qa_pathway.png",      "QA→Pathway"),
        (panel_qa_patch,       "panel5_qa_patch.png",        "QA→Patch"),
        (panel_patch_pathway,  "panel6_patch_pathway.png",   "Patch↔Pathway"),
    ]

    def __init__(
        self,
        model,
        save_dir: str,
        device: str = "cuda",
        preprocessing_dir: str = "./outputs/preprocessing",
    ):
        self.save_dir = save_dir
        self.extractor = InterpretabilityDataExtractor(
            model=model,
            device=device,
            preprocessing_dir=preprocessing_dir,
        )

    # ─────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        data_WSI,
        data_omic,
        data_text,
        slide_id: str,
        sample_id: str = "sample",
        true_label=None,
        class_names=None,
        pathway_names=None,
        qa_texts=None,
        survival_time=None,
        event=None,
    ) -> dict:
        """
        为单个样本生成所有面板，每个面板独立运行，互不干扰。

        Returns
        -------
        dict  键=面板描述，值=True/False（是否成功）
        """
        print(f"\n{'='*60}")
        print(f"🔍 Interpretability: {sample_id}")
        print(f"   Slide: {slide_id}")
        print(f"{'='*60}")

        # 1. 数据提取（只运行一次）
        data = self.extractor.extract(
            data_WSI=data_WSI,
            data_omic=data_omic,
            data_text=data_text,
            slide_id=slide_id,
            sample_id=sample_id,
            true_label=true_label,
            class_names=class_names,
            pathway_names=pathway_names,
            qa_texts=qa_texts,
            survival_time=survival_time,
            event=event,
        )

        # 2. 每个面板独立保存
        out_dir = os.path.join(self.save_dir, sample_id)
        os.makedirs(out_dir, exist_ok=True)

        results = {}
        for module, filename, desc in self._PANELS:
            path = os.path.join(out_dir, filename)
            try:
                ok = self._call_panel(module, desc, data, path)
                results[desc] = ok
            except Exception as e:
                print(f"  ❌ [{desc}] crashed: {e}")
                traceback.print_exc()
                results[desc] = False

        # 3. 汇总
        n_ok = sum(results.values())
        print(f"\n📦 Saved to: {out_dir}")
        print(f"   {n_ok}/{len(results)} panels succeeded")
        for desc, ok in results.items():
            print(f"   {'✅' if ok else '❌'} {desc}")

        return results

    def run_batch(
        self,
        data_WSI_list,
        data_omic,
        data_text,
        slide_ids: list,
        sample_ids: list,
        true_labels=None,
        class_names=None,
        pathway_names=None,
        qa_texts=None,
        survival_times=None,
        events=None,
        max_visualize: int = 3,
    ) -> int:
        """
        批量运行，最多处理 max_visualize 个样本。

        Returns
        -------
        int  成功处理的样本数
        """
        n = min(len(slide_ids), max_visualize)
        n_success = 0

        for i in range(n):
            try:
                wsi_i   = data_WSI_list[i] if isinstance(data_WSI_list, list) \
                          else data_WSI_list[i:i+1]
                omic_i  = data_omic[i] if data_omic is not None else None
                text_i  = data_text[i] if data_text is not None else None
                tl_i    = int(true_labels[i]) if true_labels is not None else None
                st_i    = float(survival_times[i]) if survival_times is not None else None
                ev_i    = int(events[i]) if events is not None else None

                results = self.run(
                    data_WSI=wsi_i,
                    data_omic=omic_i,
                    data_text=text_i,
                    slide_id=slide_ids[i],
                    sample_id=sample_ids[i],
                    true_label=tl_i,
                    class_names=class_names,
                    pathway_names=pathway_names,
                    qa_texts=qa_texts,
                    survival_time=st_i,
                    event=ev_i,
                )
                if any(results.values()):
                    n_success += 1
            except Exception as e:
                print(f"  ❌ Batch item {i} ({slide_ids[i]}) failed: {e}")
                traceback.print_exc()

        return n_success

    # ─────────────────────────────────────────────────────────────
    # 私有：panel dispatch
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _call_panel(module, desc, data, path) -> bool:
        """将 ExtractedData 路由到对应面板的 save_panel()"""

        if module.__name__.endswith("panel_wsi_heatmap"):
            return module.save_panel(
                thumbnail=data.thumbnail,
                coords=data.coords,
                attention_scores=data.attention_scores,
                save_path=path,
                slide_id=data.slide_id,
            )

        if module.__name__.endswith("panel_fusion_weights"):
            return module.save_panel(
                fusion_weights=data.fusion_weights,
                save_path=path,
                pred_label=data.pred_label,
                true_label=data.true_label,
                class_names=data.class_names,
            )

        if module.__name__.endswith("panel_omic_pathways"):
            return module.save_panel(
                pathway_scores=data.pathway_scores,
                pathway_names=data.pathway_names,
                save_path=path,
            )

        if module.__name__.endswith("panel_qa_pathway"):
            return module.save_panel(
                qa2pathway_attn=data.qa2pathway_attn,
                qa_texts=data.qa_texts,
                pathway_names=data.pathway_names,
                save_path=path,
            )

        if module.__name__.endswith("panel_qa_patch"):
            return module.save_panel(
                qa2patch_attn=data.qa2patch_attn,
                qa_texts=data.qa_texts,
                save_path=path,
                thumbnail=data.thumbnail,
                coords=data.coords,
            )

        if module.__name__.endswith("panel_patch_pathway"):
            return module.save_panel(
                patch_pathway_map=data.patch_pathway_map,
                pathway_names=data.pathway_names,
                save_path=path,
            )

        print(f"  ⚠️ Unknown panel module: {module.__name__}")
        return False
