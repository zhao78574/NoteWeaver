"""Policy Engine — 将 Router 输出映射为具体管线参数

职责：
  visual_density → frame_interval, max_images, vision_budget
  content_structure → note_template, chunk_strategy
  domain → correction_keywords, composer_hints

Policy 与 Router 分离的好处：
  - Router 只输出信号，不做参数决策
  - 修改策略不需要重新训练 Router
  - 可以按用户等级 / 成本预算切换不同 Policy
"""

from typing import Any, Dict, List


# ── 视觉密度 → 截图参数 ─────────────────────────────────────

_VISUAL_DENSITY_POLICY = {
    "low": {
        "frame_interval": 120,
        "max_images": 10,
        "vision_budget": "low",
        "description": "纯讲课/播客，画面变化极少",
    },
    "medium": {
        "frame_interval": 60,
        "max_images": 25,
        "vision_budget": "medium",
        "description": "PPT+讲解，图文混合",
    },
    "high": {
        "frame_interval": 15,
        "max_images": 60,
        "vision_budget": "high",
        "description": "频繁切画面/演示/实验操作",
    },
}

# ── 内容结构 → 排版模板 ─────────────────────────────────────

_CONTENT_STRUCTURE_POLICY = {
    "lecture": {
        "note_style": "detailed",
        "composer_template": "lecture_default",
        "chunk_strategy": "section_by_topic",
        "description": "课程/讲座：章节式知识展开",
    },
    "tutorial": {
        "note_style": "step_by_step",
        "composer_template": "tutorial_procedure",
        "chunk_strategy": "step_by_step",
        "description": "操作教程/实验：步骤式流程记录",
    },
    "research_talk": {
        "note_style": "outline",
        "composer_template": "research_presentation",
        "chunk_strategy": "section_by_topic",
        "description": "学术报告/会议演讲：背景→方法→结果→结论",
    },
    "meeting": {
        "note_style": "outline",
        "composer_template": "meeting_minutes",
        "chunk_strategy": "agenda_based",
        "description": "会议记录：按议程整理要点",
    },
}

# ── 领域 → 纠错关键词 / 排版提示 ────────────────────────────

_DOMAIN_HINTS = {
    "semiconductor": {
        "correction_keywords": [
            "光刻", "刻蚀", "沉积", "扩散", "离子注入",
            "外延", "PN结", "CMOS", "MOSFET", "FinFET",
            "CVD", "PVD", "ALD", "CMP", "STI", "LOCOS",
            "晶圆", "衬底", "栅氧", "多晶硅", "金属化",
        ],
        "composer_hint": "半导体制造工艺词汇较多，注意术语英文缩写标注。",
    },
    "ai_ml": {
        "correction_keywords": [
            "Transformer", "Embedding", "Attention", "Backpropagation",
            "Gradient", "Loss", "Epoch", "Batch", "Layer",
            "CNN", "RNN", "LSTM", "GAN", "VAE", "RLHF",
            "RAG", "Agent", "Prompt", "Fine-tuning", "Inference",
        ],
        "composer_hint": "AI/ML 术语和技术名词较多，注意英文缩写首次出现时标注全称。",
    },
    "medicine": {
        "correction_keywords": [
            "病理", "细胞", "组织", "基因", "蛋白",
            "临床", "诊断", "治疗", "手术", "药物",
            "炎症", "肿瘤", "感染", "免疫", "代谢",
        ],
        "composer_hint": "医学术语较多，注意药物名和疾病名称的准确性。",
    },
    "physics": {
        "correction_keywords": [
            "量子", "电子", "光子", "能带", "费米",
            "掺杂", "载流子", "迁移率", "电阻率",
            "势垒", "隧穿", "复合", "漂移", "扩散",
        ],
        "composer_hint": "物理概念和公式较多，注意单位和符号的准确表达。",
    },
    "chemistry": {
        "correction_keywords": [
            "反应", "催化", "合成", "分子", "浓度",
            "氧化", "还原", "沉淀", "溶解", "滴定",
            "pH", "摩尔", "当量", "收率", "纯度",
        ],
        "composer_hint": "化学术语和反应式较多，注意化学式和实验数据的准确性。",
    },
    "exam_prep": {
        "correction_keywords": [],
        "composer_hint": "考研/应试内容，注意知识点归纳和考点标注。",
    },
    "general": {
        "correction_keywords": [],
        "composer_hint": "",
    },
}


# ── 默认策略（兜底） ──────────────────────────────────────────

_DEFAULT_POLICY = {
    "frame_interval": 60,
    "max_images": 20,
    "vision_budget": "medium",
    "note_style": "detailed",
    "composer_template": "default",
    "chunk_strategy": "section_by_topic",
    "correction_keywords": [],
    "composer_hint": "",
}


class PolicyEngine:
    """策略引擎 — 单例，Router 输出 → 管线参数"""

    def __init__(self):
        self._custom_overrides: Dict[str, Any] = {}

    def compute(self, router_output: Dict[str, Any]) -> Dict[str, Any]:
        """根据 Router 输出计算管线参数

        Args:
            router_output: Router 的输出
                {
                    "domain": "semiconductor",
                    "content_structure": "lecture",
                    "visual_density": "medium",
                    "keywords": [...],
                    ...
                }

        Returns:
            {
                "frame_interval": 60,
                "max_images": 25,
                "vision_budget": "medium",
                "note_style": "detailed",
                "composer_template": "lecture_default",
                "chunk_strategy": "section_by_topic",
                "correction_keywords": [...],
                "composer_hint": "...",
            }
        """
        domain = router_output.get("domain", "general")
        structure = router_output.get("content_structure", "lecture")
        density = router_output.get("visual_density", "medium")

        # ── 从三个维度查表 ──
        density_cfg = _VISUAL_DENSITY_POLICY.get(density, _VISUAL_DENSITY_POLICY["medium"])
        structure_cfg = _CONTENT_STRUCTURE_POLICY.get(structure, _CONTENT_STRUCTURE_POLICY["lecture"])
        domain_hints = _DOMAIN_HINTS.get(domain, _DOMAIN_HINTS["general"])

        # ── 融合 ──
        policy = dict(_DEFAULT_POLICY)
        policy.update({
            "frame_interval": density_cfg["frame_interval"],
            "max_images": density_cfg["max_images"],
            "vision_budget": density_cfg["vision_budget"],
            "note_style": structure_cfg["note_style"],
            "composer_template": structure_cfg["composer_template"],
            "chunk_strategy": structure_cfg["chunk_strategy"],
        })

        # 领域特定关键词 + Router 动态关键词 合并去重
        all_keywords = list(domain_hints["correction_keywords"])
        for kw in router_output.get("keywords", []):
            if kw not in all_keywords:
                all_keywords.append(kw)
        policy["correction_keywords"] = all_keywords

        policy["composer_hint"] = domain_hints["composer_hint"]

        # ── 用户自定义覆盖（最高优先级） ──
        for k, v in self._custom_overrides.items():
            if k in policy:
                policy[k] = v

        return policy

    def set_override(self, key: str, value: Any):
        """设置用户自定义覆盖（例如用户手动指定"这是半导体课程"）"""
        self._custom_overrides[key] = value

    def clear_overrides(self):
        """清除所有用户自定义覆盖"""
        self._custom_overrides.clear()

    @staticmethod
    def available_domains() -> List[str]:
        """返回所有支持的领域列表"""
        return list(_DOMAIN_HINTS.keys())

    @staticmethod
    def available_structures() -> List[str]:
        """返回所有支持的内容结构列表"""
        return list(_CONTENT_STRUCTURE_POLICY.keys())

    @staticmethod
    def visual_density_from_frames(frame_diff_scores: List[float]) -> str:
        """从相邻帧差异分数估算 visual_density

        Args:
            frame_diff_scores: 相邻帧的直方图差异值列表 [0~1]

        Returns:
            "low" | "medium" | "high"
        """
        if not frame_diff_scores:
            return "medium"

        avg_diff = sum(frame_diff_scores) / len(frame_diff_scores)

        if avg_diff < 0.05:
            return "low"
        elif avg_diff < 0.25:
            return "medium"
        else:
            return "high"
