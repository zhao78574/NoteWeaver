"""测试 Job.from_input() — URL/路径/命令的分类路由"""
import pytest
import os
import tempfile
from note_weaver.core.job import Job, Modality, PipelineType


class TestJobFromInput:
    """所有输入类型的自动识别测试"""

    def test_empty_input(self):
        """空输入 → QA_ONLY"""
        job = Job.from_input("")
        assert job.pipeline == PipelineType.QA_ONLY
        assert job.modality == Modality.TEXT

    def test_plain_question(self):
        """普通文本 → QA_ONLY"""
        job = Job.from_input("阈值电压和开启电压有什么区别")
        assert job.pipeline == PipelineType.QA_ONLY

    def test_question_with_question_mark(self):
        """带问号的文本 → QA_ONLY"""
        job = Job.from_input("什么是PIE工艺？")
        assert job.pipeline == PipelineType.QA_ONLY

    # ── 视频路径 ──

    def test_video_mp4_path(self):
        """.mp4 文件路径 → FULL_NOTE"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            job = Job.from_input(path)
            assert job.pipeline == PipelineType.FULL_NOTE
            assert job.modality == Modality.VIDEO
        finally:
            os.unlink(path)

    def test_video_mkv_path(self):
        """.mkv 文件路径 → FULL_NOTE"""
        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            path = f.name
        try:
            job = Job.from_input(path)
            assert job.pipeline == PipelineType.FULL_NOTE
        finally:
            os.unlink(path)

    def test_audio_m4a_path(self):
        """.m4a 文件路径 → FULL_NOTE + AUDIO"""
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name
        try:
            job = Job.from_input(path)
            assert job.pipeline == PipelineType.FULL_NOTE
            assert job.modality == Modality.AUDIO
        finally:
            os.unlink(path)

    # ── PDF ──

    def test_pdf_path(self):
        """.pdf 文件路径 → PDF_NOTE"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            job = Job.from_input(path)
            assert job.pipeline == PipelineType.PDF_NOTE
            assert job.modality == Modality.PDF
        finally:
            os.unlink(path)

    # ── 笔记重排 ──

    def test_md_path(self):
        """.md 文件路径 → REGENERATE"""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            job = Job.from_input(path)
            assert job.pipeline == PipelineType.REGENERATE
            assert job.modality == Modality.NOTE
        finally:
            os.unlink(path)

    def test_regenerate_command(self):
        """'重排 阈值电压' → REGENERATE"""
        job = Job.from_input("重排 阈值电压")
        assert job.pipeline == PipelineType.REGENERATE
        assert job.metadata.get("search_mode") is True

    # ── URL ──

    def test_bilibili_video_url(self):
        """B站单视频 URL → FULL_NOTE + video_url"""
        job = Job.from_input("https://www.bilibili.com/video/BV1GJ411x7")
        assert job.pipeline == PipelineType.FULL_NOTE
        assert job.metadata.get("source") == "video_url"

    def test_bilibili_url_with_chinese_following(self):
        """B站 URL 后接中文 → URL 正确提取（修复的 bug）"""
        job = Job.from_input("帮我把https://www.bilibili.com/video/BV1xx转成笔记")
        assert job.pipeline == PipelineType.FULL_NOTE
        assert job.metadata.get("source") == "video_url"

    def test_youtube_url(self):
        """YouTube URL → FULL_NOTE"""
        job = Job.from_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert job.pipeline == PipelineType.FULL_NOTE
        assert job.metadata.get("source") == "video_url"

    def test_bilibili_playlist_url(self):
        """B站合集 URL → 合集处理"""
        job = Job.from_input("https://www.bilibili.com/list/xxx?sid=123")
        assert job.metadata.get("source") == "playlist_url"
        # 合集 URL 在 from_input 中返回 BATCH_VIDEO
        assert job.pipeline in (PipelineType.BATCH_VIDEO, PipelineType.MERGE_NOTES)

    # ── 特殊命令 ──

    def test_graph_command(self):
        """'graph' → QA_ONLY + command=graph"""
        job = Job.from_input("graph")
        assert job.metadata.get("command") == "graph"

    def test_stats_command(self):
        """'/stats' → QA_ONLY + command=stats"""
        job = Job.from_input("/stats")
        assert job.metadata.get("command") == "stats"

    # ── 目录 ──

    def test_video_directory(self):
        """含视频的目录 → BATCH_VIDEO"""
        with tempfile.TemporaryDirectory() as d:
            # 在临时目录中创建一个 .mp4 文件
            vid_path = os.path.join(d, "test.mp4")
            with open(vid_path, "w") as f:
                f.write("fake video")
            try:
                job = Job.from_input(d)
                assert job.pipeline == PipelineType.BATCH_VIDEO
            finally:
                os.unlink(vid_path)

    # ── from/下载 命令 ──

    def test_from_url(self):
        """'from https://...' → FULL_NOTE"""
        job = Job.from_input("from https://www.bilibili.com/video/BV1xx")
        assert job.pipeline == PipelineType.FULL_NOTE
        assert job.metadata.get("source") == "video_url"

    def test_download_command(self):
        """'下载 https://...' → FULL_NOTE"""
        job = Job.from_input("下载 https://www.bilibili.com/video/BV1xx")
        assert job.pipeline == PipelineType.FULL_NOTE
        assert job.metadata.get("source") == "video_url"

    # ── 合集命令 ──

    def test_playlist_command(self):
        """'合集 https://...' → MERGE_NOTES"""
        job = Job.from_input("合集 https://www.bilibili.com/list/xxx")
        assert job.pipeline == PipelineType.MERGE_NOTES
