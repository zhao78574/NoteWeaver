"""测试 PolicyEngine — Router 输出 → 管线参数映射

纯逻辑测试，无需 mock LLM 或外部依赖。
"""
import pytest
from note_weaver.agents.policy import (
    PolicyEngine,
    _VISUAL_DENSITY_POLICY,
    _CONTENT_STRUCTURE_POLICY,
    _DOMAIN_HINTS,
    _DEFAULT_POLICY,
)


# ════════════════════════════════════════════════════════════════
# PolicyEngine.compute() 核心逻辑
# ════════════════════════════════════════════════════════════════

class TestPolicyCompute:
    """测试 compute() — 三维查表 + 融合 + 覆盖"""

    def test_default_fallback(self):
        """空 Router 输出 → 默认策略（medium density + lecture structure + general domain）"""
        engine = PolicyEngine()
        result = engine.compute({})
        # density=medium → frame_interval=60, max_images=25
        assert result["frame_interval"] == 60
        assert result["max_images"] == 25
        assert result["vision_budget"] == "medium"
        # structure=lecture → detailed + section_by_topic
        assert result["note_style"] == "detailed"
        assert result["chunk_strategy"] == "section_by_topic"
        # domain=general → 空关键词
        assert result["correction_keywords"] == []
        assert result["composer_hint"] == ""

    def test_visual_density_low(self):
        """visual_density=low → 慢速间隔 + 少图"""
        engine = PolicyEngine()
        result = engine.compute({"visual_density": "low"})
        assert result["frame_interval"] == 120
        assert result["max_images"] == 10
        assert result["vision_budget"] == "low"

    def test_visual_density_high(self):
        """visual_density=high → 快速间隔 + 多图"""
        engine = PolicyEngine()
        result = engine.compute({"visual_density": "high"})
        assert result["frame_interval"] == 15
        assert result["max_images"] == 60
        assert result["vision_budget"] == "high"

    def test_visual_density_medium(self):
        """visual_density=medium → 默认间隔"""
        engine = PolicyEngine()
        result = engine.compute({"visual_density": "medium"})
        assert result["frame_interval"] == 60
        assert result["max_images"] == 25
        assert result["vision_budget"] == "medium"

    def test_content_structure_tutorial(self):
        """content_structure=tutorial → step_by_step"""
        engine = PolicyEngine()
        result = engine.compute({"content_structure": "tutorial"})
        assert result["note_style"] == "step_by_step"
        assert result["chunk_strategy"] == "step_by_step"

    def test_content_structure_research_talk(self):
        """content_structure=research_talk → outline + research_presentation"""
        engine = PolicyEngine()
        result = engine.compute({"content_structure": "research_talk"})
        assert result["note_style"] == "outline"
        assert result["composer_template"] == "research_presentation"

    def test_content_structure_meeting(self):
        """content_structure=meeting → agenda_based chunking"""
        engine = PolicyEngine()
        result = engine.compute({"content_structure": "meeting"})
        assert result["note_style"] == "outline"
        assert result["chunk_strategy"] == "agenda_based"

    def test_domain_semiconductor_keywords(self):
        """semiconductor 领域 → 包含工艺术语"""
        engine = PolicyEngine()
        result = engine.compute({"domain": "semiconductor"})
        kw = result["correction_keywords"]
        assert "光刻" in kw
        assert "CMOS" in kw
        assert "CVD" in kw
        assert "PVD" in kw
        assert result["composer_hint"] != ""

    def test_domain_ai_ml_keywords(self):
        """ai_ml 领域 → 包含 Transformer 等"""
        engine = PolicyEngine()
        result = engine.compute({"domain": "ai_ml"})
        kw = result["correction_keywords"]
        assert "Transformer" in kw
        assert "RAG" in kw

    def test_domain_unknown_falls_back_to_general(self):
        """未知领域 → general（空关键词）"""
        engine = PolicyEngine()
        result = engine.compute({"domain": "basketball"})
        assert result["correction_keywords"] == []
        assert result["composer_hint"] == ""

    def test_router_keywords_merged_with_domain(self):
        """Router 输出的动态关键词与领域关键词合并去重"""
        engine = PolicyEngine()
        result = engine.compute({
            "domain": "semiconductor",
            "keywords": ["FinFET", "GAA", "光刻"],  # "光刻" 与领域关键词重复
        })
        kw = result["correction_keywords"]
        # 领域关键词在前，动态关键词在后
        assert "光刻" in kw
        assert "FinFET" in kw
        assert "GAA" in kw
        # "光刻" 不应重复
        assert kw.count("光刻") == 1

    def test_all_dimensions_combined(self):
        """三维同时指定 → 各自产生正确映射"""
        engine = PolicyEngine()
        result = engine.compute({
            "domain": "physics",
            "content_structure": "research_talk",
            "visual_density": "high",
            "keywords": ["Mott transition"],
        })
        # 视觉密度 → high
        assert result["frame_interval"] == 15
        assert result["max_images"] == 60
        # 内容结构 → research_talk
        assert result["note_style"] == "outline"
        # 领域 → physics keywords
        assert "载流子" in result["correction_keywords"]
        assert "Mott transition" in result["correction_keywords"]

    def test_override_overrides_computed(self):
        """set_override 覆盖计算结果"""
        engine = PolicyEngine()
        engine.set_override("frame_interval", 999)
        result = engine.compute({"visual_density": "high"})
        assert result["frame_interval"] == 999  # 覆盖生效
        assert result["max_images"] == 60       # 未覆盖的仍正常

    def test_clear_overrides_restores_default(self):
        """clear_overrides 后恢复默认"""
        engine = PolicyEngine()
        engine.set_override("frame_interval", 999)
        engine.clear_overrides()
        result = engine.compute({"visual_density": "high"})
        assert result["frame_interval"] == 15  # 恢复

    def test_override_non_existent_key(self):
        """覆盖不存在的 key 不影响输出"""
        engine = PolicyEngine()
        engine.set_override("nonexistent", "value")
        result = engine.compute({})
        assert "nonexistent" not in result
        assert result["frame_interval"] == 60  # 正常

    def test_output_contains_all_expected_keys(self):
        """输出字典包含所有预期键"""
        engine = PolicyEngine()
        result = engine.compute({})
        expected_keys = {
            "frame_interval", "max_images", "vision_budget",
            "note_style", "composer_template", "chunk_strategy",
            "correction_keywords", "composer_hint",
        }
        assert expected_keys.issubset(result.keys())


# ════════════════════════════════════════════════════════════════
# 静态方法
# ════════════════════════════════════════════════════════════════

class TestAvailableDomains:
    def test_returns_all_domain_keys(self):
        domains = PolicyEngine.available_domains()
        assert "semiconductor" in domains
        assert "ai_ml" in domains
        assert "general" in domains
        assert "medicine" in domains
        assert "physics" in domains
        assert "chemistry" in domains

    def test_domains_count(self):
        domains = PolicyEngine.available_domains()
        assert len(domains) == len(_DOMAIN_HINTS)


class TestAvailableStructures:
    def test_returns_all_structure_keys(self):
        structures = PolicyEngine.available_structures()
        assert "lecture" in structures
        assert "tutorial" in structures
        assert "research_talk" in structures
        assert "meeting" in structures

    def test_structures_count(self):
        structures = PolicyEngine.available_structures()
        assert len(structures) == len(_CONTENT_STRUCTURE_POLICY)


class TestVisualDensityFromFrames:
    """visual_density_from_frames — 从帧差异值估算视觉密度"""

    def test_low_density(self):
        """平均帧差 < 0.05 → low"""
        assert PolicyEngine.visual_density_from_frames([0.01, 0.02, 0.03]) == "low"

    def test_medium_density(self):
        """0.05 ≤ avg < 0.25 → medium"""
        assert PolicyEngine.visual_density_from_frames([0.10, 0.10, 0.10]) == "medium"
        assert PolicyEngine.visual_density_from_frames([0.05]) == "medium"   # 边界
        assert PolicyEngine.visual_density_from_frames([0.24]) == "medium"

    def test_high_density(self):
        """avg ≥ 0.25 → high"""
        assert PolicyEngine.visual_density_from_frames([0.5, 0.5, 0.5]) == "high"
        assert PolicyEngine.visual_density_from_frames([0.25]) == "high"     # 边界

    def test_empty_list(self):
        """空列表 → medium（兜底）"""
        assert PolicyEngine.visual_density_from_frames([]) == "medium"

    def test_mixed_values(self):
        """混合值 """
        scores = [0.01, 0.50, 0.01]  # avg=0.173 → medium
        assert PolicyEngine.visual_density_from_frames(scores) == "medium"

    def test_all_zeros(self):
        assert PolicyEngine.visual_density_from_frames([0.0, 0.0]) == "low"


# ════════════════════════════════════════════════════════════════
# 策略表数据完整性
# ════════════════════════════════════════════════════════════════

class TestPolicyTableIntegrity:
    """验证策略表本身的完整性"""

    def test_all_density_entries_have_required_keys(self):
        required = {"frame_interval", "max_images", "vision_budget", "description"}
        for level in ("low", "medium", "high"):
            assert required.issubset(_VISUAL_DENSITY_POLICY[level].keys()), \
                f"{level} 缺少必要字段"

    def test_all_structure_entries_have_required_keys(self):
        required = {"note_style", "composer_template", "chunk_strategy", "description"}
        for structure in ("lecture", "tutorial", "research_talk", "meeting"):
            assert required.issubset(_CONTENT_STRUCTURE_POLICY[structure].keys()), \
                f"{structure} 缺少必要字段"

    def test_all_domain_entries_have_required_keys(self):
        for domain, hints in _DOMAIN_HINTS.items():
            assert "correction_keywords" in hints, f"{domain} 缺少 correction_keywords"
            assert "composer_hint" in hints, f"{domain} 缺少 composer_hint"

    def test_density_intervals_are_ordered(self):
        """密度越高 → 间隔越小"""
        low_interval = _VISUAL_DENSITY_POLICY["low"]["frame_interval"]
        med_interval = _VISUAL_DENSITY_POLICY["medium"]["frame_interval"]
        high_interval = _VISUAL_DENSITY_POLICY["high"]["frame_interval"]
        assert low_interval > med_interval > high_interval

    def test_density_max_images_are_ordered(self):
        """密度越高 → 图片上限越大"""
        low_max = _VISUAL_DENSITY_POLICY["low"]["max_images"]
        med_max = _VISUAL_DENSITY_POLICY["medium"]["max_images"]
        high_max = _VISUAL_DENSITY_POLICY["high"]["max_images"]
        assert low_max < med_max < high_max
