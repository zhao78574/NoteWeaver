"""测试 PipelineCache — 本地磁盘计算缓存"""
import os
import tempfile
import shutil
import pytest
from note_weaver.core.cache import PipelineCache


class TestPipelineCache:
    """缓存系统核心行为测试"""

    @pytest.fixture
    def cache(self):
        tmp = tempfile.mkdtemp()
        c = PipelineCache(cache_dir=tmp)
        yield c
        shutil.rmtree(tmp, ignore_errors=True)

    def test_miss_returns_none(self, cache):
        result = cache.get("/nonexistent/video.mp4", "transcribe")
        assert result is None

    def test_set_and_get(self, cache):
        data = {"text": "hello world", "segments": []}
        cache.set("/test/video.mp4", "transcribe", data)
        result = cache.get("/test/video.mp4", "transcribe")
        assert result == data

    def test_cache_key_differs_by_stage(self, cache):
        cache.set("/test/video.mp4", "transcribe", "audio_data")
        cache.set("/test/video.mp4", "vision", "image_data")
        assert cache.get("/test/video.mp4", "transcribe") == "audio_data"
        assert cache.get("/test/video.mp4", "vision") == "image_data"

    def test_cache_key_differs_by_params(self, cache):
        cache.set("/test/video.mp4", "transcribe", "small", {"model": "small"})
        cache.set("/test/video.mp4", "transcribe", "large", {"model": "large"})
        assert cache.get("/test/video.mp4", "transcribe", {"model": "small"}) == "small"
        assert cache.get("/test/video.mp4", "transcribe", {"model": "large"}) == "large"

    def test_invalidate_stage(self, cache):
        cache.set("/test/video.mp4", "transcribe", "data")
        cache.invalidate("/test/video.mp4", "transcribe")
        assert cache.get("/test/video.mp4", "transcribe") is None

    def test_invalidate_all(self, cache):
        cache.set("/test/video.mp4", "transcribe", "data")
        cache.set("/test/video.mp4", "vision", "data")
        cache.invalidate("/test/video.mp4")
        assert cache.get("/test/video.mp4", "transcribe") is None
        assert cache.get("/test/video.mp4", "vision") is None

    def test_cache_isolation_by_input(self, cache):
        cache.set("/video1.mp4", "transcribe", "result1")
        cache.set("/video2.mp4", "transcribe", "result2")
        assert cache.get("/video1.mp4", "transcribe") == "result1"
        assert cache.get("/video2.mp4", "transcribe") == "result2"
        cache.invalidate("/video1.mp4", "transcribe")
        assert cache.get("/video2.mp4", "transcribe") == "result2"

    def test_stats_after_set(self, cache):
        cache.set("/test.mp4", "transcribe", "data")
        stats = cache.stats()
        assert stats["entries"] >= 1

    def test_clear_all(self, cache):
        cache.set("/test.mp4", "transcribe", "data")
        cache.set("/test2.mp4", "vision", "data")
        cache.clear_all()
        stats = cache.stats()
        assert stats["entries"] == 0

    def test_roundtrip_complex_data(self, cache):
        data = {
            "timestamped": "[00:00] Hello\n[01:00] World",
            "raw_text": "Hello World",
            "segments": [
                {"start": 0.0, "end": 12.5, "text": "Hello"},
                {"start": 12.5, "end": 30.0, "text": "World"},
            ],
            "duration": 30.0,
            "language": "zh",
        }
        cache.set("/test.mp4", "transcribe", data)
        loaded = cache.get("/test.mp4", "transcribe")
        assert loaded == data
