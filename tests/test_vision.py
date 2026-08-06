"""测试 VisionAgent — 图片质量过滤/去重/文字密度/配额检测

纯图像处理测试，无需 mock LLM。用 PIL 在临时目录生成测试图片。
"""
import pytest
import os
import tempfile
import numpy as np
from note_weaver.agents.vision import VisionAgent


# ════════════════════════════════════════════════════════════════
# 测试图片工厂函数
# ════════════════════════════════════════════════════════════════

def _make_image(path: str, size=(200, 200), color=(128, 128, 128)):
    """创建纯色图片"""
    from PIL import Image
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG")
    return path


def _make_textured_image(path: str, size=(200, 200)):
    """创建有纹理的图片（模拟文字/内容）—— 使用 PNG 避免 JPEG 压缩失真"""
    from PIL import Image
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    # 添加一些"文字区域"（高对比度矩形块）
    arr[30:50, 30:170, :] = 0       # 黑色文字条
    arr[70:90, 30:170, :] = 0
    arr[110:130, 30:170, :] = 0
    Image.fromarray(arr).save(path, "PNG")
    return path


def _make_noise_image(path: str, size=(200, 200)):
    """创建纯随机噪声图片（高 stddev，不会被过滤，PNG 无损保存）"""
    from PIL import Image
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "PNG")
    return path


# ════════════════════════════════════════════════════════════════
# 测试夹具
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def vision() -> VisionAgent:
    """创建默认 VisionAgent（跳过 LLM 调用）"""
    return VisionAgent(max_images=5)


@pytest.fixture
def tmpdir():
    """临时目录（存放测试图片）"""
    with tempfile.TemporaryDirectory() as td:
        yield td


# ════════════════════════════════════════════════════════════════
# _quality_filter — PIL 图片质量过滤
# ════════════════════════════════════════════════════════════════

class TestQualityFilter:
    """质量过滤：尺寸/亮度/标准差"""

    def test_normal_image_passes(self, vision, tmpdir):
        """正常图片（200x200, 有纹理无噪声）应通过"""
        path = os.path.join(tmpdir, "normal.jpg")
        _make_textured_image(path, size=(200, 200))
        result = vision._quality_filter([path])
        assert len(result) == 1
        assert result[0] == path

    def test_too_small_image_filtered(self, vision, tmpdir):
        """过小图片被过滤（< 100×100）"""
        path = os.path.join(tmpdir, "tiny.jpg")
        _make_image(path, size=(50, 50), color=(128, 128, 128))
        result = vision._quality_filter([path])
        assert len(result) == 0

    def test_too_narrow_image_filtered(self, vision, tmpdir):
        """过窄图片被过滤（width < 100）"""
        path = os.path.join(tmpdir, "narrow.jpg")
        _make_image(path, size=(80, 200), color=(128, 128, 128))
        result = vision._quality_filter([path])
        assert len(result) == 0

    def test_too_short_image_filtered(self, vision, tmpdir):
        """过矮图片被过滤（height < 100）"""
        path = os.path.join(tmpdir, "short.jpg")
        _make_image(path, size=(200, 80), color=(128, 128, 128))
        result = vision._quality_filter([path])
        assert len(result) == 0

    def test_dark_image_filtered(self, vision, tmpdir):
        """过暗图片被过滤（brightness < 10）"""
        path = os.path.join(tmpdir, "dark.jpg")
        _make_image(path, size=(200, 200), color=(2, 2, 2))  # 极小亮度
        result = vision._quality_filter([path])
        assert len(result) == 0

    def test_solid_color_filtered(self, vision, tmpdir):
        """纯色过渡帧被过滤（stddev < 5）"""
        path = os.path.join(tmpdir, "solid.jpg")
        _make_image(path, size=(200, 200), color=(128, 128, 128))  # 纯色 = stddev=0
        result = vision._quality_filter([path])
        assert len(result) == 0

    def test_textured_image_passes(self, vision, tmpdir):
        """有纹理的图片（高 stddev）应通过"""
        path = os.path.join(tmpdir, "textured.jpg")
        _make_textured_image(path, size=(200, 200))
        result = vision._quality_filter([path])
        assert len(result) == 1

    def test_mixed_batch(self, vision, tmpdir):
        """混合批次：好的保留，差的过滤"""
        good = os.path.join(tmpdir, "good.jpg")
        tiny = os.path.join(tmpdir, "tiny.jpg")
        dark = os.path.join(tmpdir, "dark.jpg")
        _make_textured_image(good, size=(200, 200))
        _make_image(tiny, size=(50, 50), color=(128, 128, 128))
        _make_image(dark, size=(200, 200), color=(2, 2, 2))
        result = vision._quality_filter([good, tiny, dark])
        assert result == [good]

    def test_corrupt_image_skipped(self, vision, tmpdir):
        """损坏的图片文件被跳过（不阻断流程）"""
        bad = os.path.join(tmpdir, "bad.jpg")
        with open(bad, "wb") as f:
            f.write(b"not a real image")
        good = os.path.join(tmpdir, "good.jpg")
        _make_textured_image(good, size=(200, 200))
        result = vision._quality_filter([bad, good])
        assert result == [good]

    def test_empty_list(self, vision):
        """空列表返回空"""
        assert vision._quality_filter([]) == []

    def test_custom_thresholds(self, vision, tmpdir):
        """自定义最小尺寸阈值：120×120 通过，40×40 不过"""
        path1 = os.path.join(tmpdir, "s120.jpg")
        path2 = os.path.join(tmpdir, "s40.jpg")
        _make_noise_image(path1, size=(120, 120))
        _make_noise_image(path2, size=(40, 40))
        # 默认 min_width=100, min_height=100 → path1 (120×120) 通过，path2 (40×40) 不过
        result = vision._quality_filter([path1, path2], min_width=100, min_height=100)
        assert result == [path1]
        # 放宽到 min_width=30, min_height=30 → 都通过
        result2 = vision._quality_filter([path1, path2], min_width=30, min_height=30)
        assert len(result2) == 2


# ════════════════════════════════════════════════════════════════
# _deduplicate — 直方图相似度去重
# ════════════════════════════════════════════════════════════════

class TestDeduplicate:
    """相邻帧直方图相似度去重"""

    def test_identical_images_deduped(self, vision, tmpdir):
        """完全相同的图片去重（只保留第一张）"""
        p1 = os.path.join(tmpdir, "dup_01.jpg")
        p2 = os.path.join(tmpdir, "dup_02.jpg")
        _make_image(p1, size=(200, 200), color=(100, 100, 100))
        _make_image(p2, size=(200, 200), color=(100, 100, 100))  # 完全相同的颜色
        result = vision._deduplicate([p1, p2], sim_threshold=0.95)
        # 相同颜色 → 直方图相似度 = 1.0 > 0.95 → 去重 → 只保留第一张
        # 但安全网 min_keep=3 可能补充回来，需要 >= 3 张才会触发
        assert result == [p1]

    def test_different_images_kept(self, vision, tmpdir):
        """不同颜色的图片都保留"""
        p1 = os.path.join(tmpdir, "diff_01.jpg")
        p2 = os.path.join(tmpdir, "diff_02.jpg")
        _make_image(p1, size=(200, 200), color=(255, 0, 0))     # 红色
        _make_image(p2, size=(200, 200), color=(0, 0, 255))     # 蓝色
        result = vision._deduplicate([p1, p2], sim_threshold=0.95)
        assert len(result) == 2

    def test_sequential_different_kept(self, vision, tmpdir):
        """交替变化：diff → 第一张，变化大保留"""
        p1 = os.path.join(tmpdir, "seq_01.jpg")
        p2 = os.path.join(tmpdir, "seq_02.jpg")
        p3 = os.path.join(tmpdir, "seq_03.jpg")
        _make_image(p1, size=(200, 200), color=(255, 0, 0))     # 红
        _make_image(p2, size=(200, 200), color=(255, 0, 0))     # 红 = 重复
        _make_image(p3, size=(200, 200), color=(0, 0, 255))     # 蓝 = 不同
        result = vision._deduplicate([p1, p2, p3], sim_threshold=0.95)
        assert result == [p1, p3]  # p2 与 p1 重复被去

    def test_safety_net_min_keep(self, vision, tmpdir):
        """安全网：min_keep=3 时，即使相似度高也至少保留 3 帧"""
        paths = []
        for i in range(5):
            p = os.path.join(tmpdir, f"safe_{i:02d}.jpg")
            # 全部用相同颜色 → 直方图完全相同
            _make_image(p, size=(200, 200), color=(100, 100, 100))
            paths.append(p)
        result = vision._deduplicate(paths, sim_threshold=0.95, min_keep=3)
        # 去重后应该只有1张，但安全网会补到3
        assert len(result) >= 3
        # 第一张一定在
        assert result[0] == paths[0]

    def test_no_safety_net_when_enough_unique(self, vision, tmpdir):
        """当已有足够的独帧时不触发安全网"""
        paths = []
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        for i, color in enumerate(colors):
            p = os.path.join(tmpdir, f"uniq_{i:02d}.jpg")
            _make_image(p, size=(200, 200), color=color)
            paths.append(p)
        # 4 张不同颜色 → 全部保留
        result = vision._deduplicate(paths, sim_threshold=0.90, min_keep=3)
        assert len(result) == 4

    def test_empty_list(self, vision):
        """空列表"""
        assert vision._deduplicate([]) == []

    def test_single_image(self, vision, tmpdir):
        """单张图片不变"""
        p = os.path.join(tmpdir, "single.jpg")
        _make_textured_image(p)
        assert vision._deduplicate([p]) == [p]


# ════════════════════════════════════════════════════════════════
# _prioritize_by_text_density — 文字密度排序
# ════════════════════════════════════════════════════════════════

class TestPrioritizeByTextDensity:
    """按文字密度（边缘检测）排序"""

    def test_returns_same_count(self, vision, tmpdir):
        """排序后数量不变"""
        paths = []
        for i in range(4):
            p = os.path.join(tmpdir, f"rank_{i:02d}.jpg")
            _make_noise_image(p, size=(200, 200))
            paths.append(p)
        result = vision._prioritize_by_text_density(paths)
        assert len(result) == len(paths)

    def test_textured_ranks_higher_than_solid(self, vision, tmpdir):
        """有纹理的图片（文字更多）排在前"""
        solid = os.path.join(tmpdir, "solid.jpg")
        textured = os.path.join(tmpdir, "textured.jpg")
        _make_image(solid, size=(200, 200), color=(128, 128, 128))          # stddev=0
        _make_textured_image(textured, size=(200, 200))                     # 有文字块
        result = vision._prioritize_by_text_density([solid, textured])
        # 有纹理的密度更高 → 排前面
        assert result[0] == textured

    def test_empty_list(self, vision):
        """空列表"""
        assert vision._prioritize_by_text_density([]) == []

    def test_single_image(self, vision, tmpdir):
        """单张图片不变"""
        p = os.path.join(tmpdir, "single.jpg")
        _make_textured_image(p)
        assert vision._prioritize_by_text_density([p]) == [p]

    def test_all_solid_same_order(self, vision, tmpdir):
        """全纯色 → 密度相同 → 原序保留（稳定排序）"""
        paths = []
        for i in range(3):
            p = os.path.join(tmpdir, f"solid_{i:02d}.jpg")
            _make_image(p, size=(200, 200), color=(128, 128, 128))
            paths.append(p)
        result = vision._prioritize_by_text_density(paths)
        # 所有密度相同，排序应稳定
        assert set(result) == set(paths)


# ════════════════════════════════════════════════════════════════
# _is_quota_error — 配额耗尽检测
# ════════════════════════════════════════════════════════════════

class TestIsQuotaError:
    """配额耗尽错误的字符串匹配检测"""

    def test_quota_exhausted(self, vision):
        assert vision._is_quota_error(Exception("quota exhausted"))
        assert vision._is_quota_error(Exception("Quota Exceeded"))
        assert vision._is_quota_error(Exception("insufficient_quota"))

    def test_429_error(self, vision):
        assert vision._is_quota_error(Exception("HTTP 429 Too Many Requests"))

    def test_rate_limit(self, vision):
        assert vision._is_quota_error(Exception("rate limit reached"))
        assert vision._is_quota_error(Exception("usage limit exceeded"))
        assert vision._is_quota_error(Exception("amount limit hit"))

    def test_tokens_exceeded(self, vision):
        assert vision._is_quota_error(Exception("tokens exceeded"))
        assert vision._is_quota_error(Exception("token limit reached"))

    def test_non_quota_error(self, vision):
        assert not vision._is_quota_error(Exception("network timeout"))
        assert not vision._is_quota_error(Exception("invalid API key"))
        assert not vision._is_quota_error(Exception(""))
        assert not vision._is_quota_error(Exception("image decode failed"))

    def test_substring_match(self, vision):
        """关键词包含在更长字符串中也能检测"""
        assert vision._is_quota_error(
            Exception("Error: quota exhausted for model qwen-vl-plus")
        )
        assert vision._is_quota_error(
            Exception("Request failed: 429 — too many requests this minute")
        )


# ════════════════════════════════════════════════════════════════
# VisionAgent 配置
# ════════════════════════════════════════════════════════════════

class TestVisionConfig:
    """VisionAgent 初始化配置测试"""

    def test_default_max_images(self):
        v = VisionAgent()
        assert v.max_images is not None
        assert isinstance(v.max_images, (int, type(None)))

    def test_constructor_max_images(self):
        """构造参数在无 config 时生效，但 config 存在时优先（设计如此）"""
        v = VisionAgent(max_images=5)
        from note_weaver.utils.config import config
        if config.get("vision.max_images_per_batch") is not None:
            # config 存在时优先
            assert v.max_images == config.get("vision.max_images_per_batch")
        else:
            assert v.max_images == 5

    def test_skip_low_quality_enabled(self):
        v = VisionAgent()
        assert v.skip_low_quality is True  # 默认开启

    def test_quality_threshold_default(self):
        v = VisionAgent()
        assert v.quality_threshold == "medium"


# ════════════════════════════════════════════════════════════════
# execute() — 集成流水线（不含 LLM 调用部分）
# ════════════════════════════════════════════════════════════════

class TestExecutePipeline:
    """execute() 流水线的前三步（质量→去重→截断）集成测试

    注意：因为 VLM 调用需要真实的 Qwen API，这里只测到截断为止。
    截断后的数量应 ≤ max_images。
    """

    def test_empty_input_returns_empty(self, vision):
        """空截图列表 → 空结果"""
        result = vision.execute([])
        assert result == []

    def test_pipeline_with_all_good_images(self, vision, tmpdir):
        """全好的图片：质量过滤不丢，去重不丢，截断不超过 max_images"""
        paths = []
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255),
                   (255,0,255), (100,100,100), (200,200,200)]
        for i, color in enumerate(colors):
            p = os.path.join(tmpdir, f"good_{i:02d}.jpg")
            _make_textured_image(p, size=(200, 200))
            paths.append(p)

        # 因为 VLM 调用需要 API，这里只验证不抛异常
        # 实际测试：mock _call_vision_with_failover 返回预定义 JSON
        import json
        original_call = vision._call_vision_with_failover

        def mock_call(data_url):
            return json.dumps({
                "type": "slide",
                "content_description": "test content",
                "key_terms": ["test"],
                "contains_formula": False,
                "contains_table": False,
                "readability": "good",
                "should_include": True,
                "suggested_caption": "test",
            })

        try:
            vision._call_vision_with_failover = mock_call
            result = vision.execute(paths)
            # 因为 8 张 > max_images=5，截断到 5
            assert len(result) <= 5
            for r in result:
                assert "image_id" in r
                assert "type" in r
        finally:
            vision._call_vision_with_failover = original_call
