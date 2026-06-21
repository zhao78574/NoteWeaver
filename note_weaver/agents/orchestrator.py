"""中央调度 Agent — 任务编排 + 状态管理 + 异常处理"""

import os
import re
import json
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime

from note_weaver.core.state_machine import Task, TaskStatus, TaskStateMachine
from note_weaver.core.extractor import (
    extract_audio,
    extract_screenshots,
    clean_screenshot_dir,
    get_video_duration,
)
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger

from .classifier import ClassifierAgent
from .transcriber import TranscriberAgent
from .vision import VisionAgent
from .composer import ComposerAgent
from .qa import QAAgent
from .memory_agent import MemoryAgent


class Orchestrator:
    """NoteWeaver 中央调度器 — 管理整个笔记生成流程"""

    def __init__(self):
        # 基础设施
        self.state_machine = TaskStateMachine()

        # Agent 实例（懒加载）
        self._classifier: Optional[ClassifierAgent] = None
        self._transcriber: Optional[TranscriberAgent] = None
        self._vision: Optional[VisionAgent] = None
        self._composer: Optional[ComposerAgent] = None
        self._qa: Optional[QAAgent] = None
        self._memory: Optional[MemoryAgent] = None

        # 确保输出目录存在
        for d in [config.txt_dir,
                   config.note_dir, config.memory_dir, config.log_dir]:
            os.makedirs(d, exist_ok=True)

        logger.info("[Orchestrator] NoteWeaver 初始化完成")

    # ---- Agent 懒加载 ----

    @property
    def classifier(self) -> ClassifierAgent:
        if self._classifier is None:
            self._classifier = ClassifierAgent()
        return self._classifier

    @property
    def transcriber(self) -> TranscriberAgent:
        if self._transcriber is None:
            self._transcriber = TranscriberAgent()
        return self._transcriber

    @property
    def vision(self) -> VisionAgent:
        if self._vision is None:
            self._vision = VisionAgent()
        return self._vision

    @property
    def composer(self) -> ComposerAgent:
        if self._composer is None:
            self._composer = ComposerAgent()
        return self._composer

    @property
    def qa(self) -> QAAgent:
        if self._qa is None:
            self._qa = QAAgent()
        return self._qa

    @property
    def memory(self) -> MemoryAgent:
        if self._memory is None:
            self._memory = MemoryAgent()
        return self._memory

    # =================================================================
    # 主处理流程
    # =================================================================

    def process_video(self, video_path: str) -> Optional[Task]:
        """处理单个视频的完整流程

        Args:
            video_path: 视频文件路径

        Returns:
            完成的任务对象，失败返回 None
        """
        file_name = os.path.basename(video_path)
        file_base = os.path.splitext(file_name)[0]
        # 去除 B站/UGC 常见后缀
        for suffix in ('_Auto', '_auto', '_720P', '_1080P', '_final'):
            if file_base.endswith(suffix):
                file_base = file_base[:-len(suffix)]
                file_name = file_base + os.path.splitext(file_name)[1]
                break

        # 计算相对子目录，保持 Video 源文件夹的层级结构
        rel_subdir = self._get_rel_subdir(video_path)
        if rel_subdir:
            logger.info(f"[DIR]  输出子目录: {rel_subdir}")
            os.makedirs(os.path.join(config.txt_dir, rel_subdir), exist_ok=True)
            os.makedirs(os.path.join(config.note_dir, rel_subdir), exist_ok=True)

        # 检查是否已处理过
        existing = self.state_machine.find_processed(file_name)
        if existing:
            logger.info(f"[SKIP] 跳过已处理: {file_name}")
            return existing

        # 创建任务
        task = self.state_machine.create_task(video_path)
        logger.info(f"=" * 60)
        logger.info(f"[START] 开始处理: {file_name} (task={task.task_id})")
        logger.info(f"=" * 60)

        # 临时音频路径
        audio_tmp = os.path.join(config.base_dir, f"temp_audio_{file_base}.mp3")

        try:
            # ====== Step 1: 提取音频 + 截图 (并行) ======
            self.state_machine.transition(task, TaskStatus.EXTRACTING)

            is_audio_only = file_name.lower().endswith(
                ('.m4a', '.mp3', '.wav', '.flac', '.aac')
            )

            # 截图目录（保持源视频目录结构）
            screenshot_dir = os.path.join(config.note_dir, rel_subdir, file_base)

            if is_audio_only:
                logger.info("检测到纯音频文件，跳过截图")
                extract_audio(video_path, audio_tmp)
                task.audio_path = audio_tmp
                task.screenshot_dir = screenshot_dir
                task.screenshot_files = []
            else:
                # 并行提取
                clean_screenshot_dir(screenshot_dir, file_base)

                audio_thread = threading.Thread(
                    target=lambda: extract_audio(video_path, audio_tmp)
                )
                # 自适应截图间隔（根据视频长度阶梯调整）
                dur = get_video_duration(video_path)
                if dur <= 300:       # ≤5min → 每30s一张（约10张）
                    interval = 30
                elif dur <= 900:     # 5-15min → 每60s一张（约10-15张）
                    interval = 60
                elif dur <= 1800:    # 15-30min → 每90s一张（约15-20张）
                    interval = 90
                elif dur <= 3600:    # 30-60min → 每120s一张（约20-30张）
                    interval = 120
                else:                # >60min → 每180s一张（20张+）
                    interval = 180
                logger.info(f"截图间隔: {interval}s (视频 {dur:.0f}s, 约 {dur//max(interval,1)} 张)")

                screenshot_thread = threading.Thread(
                    target=lambda: setattr(task, 'screenshot_files',
                        extract_screenshots(
                            video_path, screenshot_dir, file_base,
                            interval))
                )

                audio_thread.start()
                screenshot_thread.start()
                audio_thread.join()
                screenshot_thread.join()

                task.audio_path = audio_tmp
                task.screenshot_dir = screenshot_dir

            logger.info(
                f"[OK] 提取完成: 音频 + {len(task.screenshot_files)} 张截图"
            )

            # ====== Step 2: 转录 + 视觉分析 (并行), 然后分类 ======
            duration = get_video_duration(video_path)

            # 转录 + 视觉分析可并行（转录只跑一次！）
            self.state_machine.transition(task, TaskStatus.TRANSCRIBING)
            self.state_machine.transition(task, TaskStatus.VISION_ANALYZING)

            transcript_result = [None]
            vision_result = [None]

            def do_transcribe():
                transcript_result[0] = self.transcriber.execute(task.audio_path)

            def do_vision():
                if task.screenshot_files:
                    vision_result[0] = self.vision.execute(task.screenshot_files)
                else:
                    vision_result[0] = []

            t1 = threading.Thread(target=do_transcribe)
            t2 = threading.Thread(target=do_vision)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            task.transcript = transcript_result[0]
            task.vision_results = vision_result[0] or []

            # 分类 — 从转录结果中提取前30秒，不再单独跑 Whisper
            segments = task.transcript.get("segments", [])
            first_30s_text = " ".join(
                seg["text"] for seg in segments if seg["start"] <= 30
            )
            if not first_30s_text.strip() and segments:
                first_30s_text = segments[0]["text"]  # fallback: 取第一段

            self.state_machine.transition(task, TaskStatus.CLASSIFYING)
            classification = self.classifier.execute(
                filename=file_name,
                audio_sample=first_30s_text,
                duration=duration,
            )
            task.classification = classification

            logger.info(
                f"[OK] 分析完成: 转录 {len(task.transcript.get('raw_text', ''))}字, "
                f"视觉 {len(task.vision_results)}张"
            )

            # ====== Step 3: 保存转录结果（TXT + JSON） ======
            raw_text = task.transcript.get("raw_text", "")
            self._save_txt(file_base, raw_text, rel_subdir)
            task.txt_path = os.path.join(config.txt_dir, rel_subdir, f"{file_base}.txt")

            # 保存完整转录 JSON（含时间戳，供重排脚本使用）
            transcript_path = os.path.join(
                config.txt_dir, rel_subdir, f"{file_base}_transcript.json"
            )
            try:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(task.transcript, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[Orchestrator] 转录 JSON 保存失败（非致命）: {e}")

            # ====== Step 4: 排版 + 质检 (支持回退) ======
            self.state_machine.transition(task, TaskStatus.COMPOSING)

            domain = classification.get("domain", "")
            difficulty = classification.get("difficulty", "")
            user_context = self.memory.get_context(domain=domain, difficulty=difficulty)
            strategy = classification.get("suggested_strategy", {})

            for attempt in range(config.qa_max_retries + 1):
                if attempt > 0:
                    self.state_machine.transition(task, TaskStatus.RETRYING)
                    logger.info(f"[RETRY] 回退重排 第{attempt}次...")
                    task.retry_count = attempt

                revision_feedback = ""
                if attempt > 0 and task.qa_report:
                    revision_feedback = task.qa_report.get("revision_suggestions", "")

                note_content = self.composer.execute(
                    file_base=file_base,
                    timestamped_text=task.transcript.get("timestamped", ""),
                    vision_results=task.vision_results,
                    strategy=strategy,
                    user_context=user_context,
                    revision_feedback=revision_feedback,
                )
                task.note_content = note_content

                self.state_machine.transition(task, TaskStatus.QA_REVIEWING)

                # 递减阈值: 首轮 config.qa_pass_threshold, 重试逐次降低
                fb = config.qa_fallback_thresholds
                cur_threshold = fb[attempt] if attempt < len(fb) else fb[-1]

                qa_report = self.qa.execute(
                    note_content=note_content,
                    transcript_text=task.transcript.get("raw_text", ""),
                    vision_results=task.vision_results,
                    threshold=cur_threshold,
                )
                task.qa_report = qa_report
                task.qa_score = qa_report.get("total", 0)

                if qa_report.get("passed", True):
                    logger.info(f"[OK] QA通过 (score={task.qa_score})")
                    break
                else:
                    logger.warning(
                        f"[FAIL] QA不通过 (score={task.qa_score}, "
                        f"attempt={attempt+1}/{config.qa_max_retries+1})"
                    )

            # ====== Step 5: 保存 + 更新记忆 ======
            note_output_dir = os.path.join(config.note_dir, rel_subdir)
            md_path = self.composer.save_note(file_base, note_content, note_output_dir)
            task.md_path = md_path

            try:
                self.memory.update_after_note(
                    file_base=file_base,
                    note_content=note_content,
                    classification=classification,
                    qa_report=qa_report,
                )
            except Exception as mem_err:
                logger.warning(f"[Memory] 更新失败（非致命）: {mem_err}")

            # ====== Step 6: 清理临时文件 ======
            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)

            self.state_machine.transition(task, TaskStatus.COMPLETED)
            logger.info(
                f"[DONE] 处理完成: {file_name} "
                f"(耗时 {task.elapsed_seconds:.0f}s, QA={task.qa_score})"
            )

            return task

        except Exception as e:
            import traceback
            task.error_message = str(e)
            task.error_traceback = traceback.format_exc()
            self.state_machine.transition(task, TaskStatus.FAILED)
            logger.error(f"[FAIL] 处理失败: {file_name}")
            logger.error(traceback.format_exc())

            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)

            return None

    # =================================================================
    # Skills（交互式命令）
    # =================================================================

    def search_notes(self, query: str) -> str:
        """搜索笔记库"""
        concepts = self.memory.search_concepts(query)
        if concepts:
            lines = [f"## 搜索「{query}」— 知识图谱匹配:\n"]
            for c in concepts:
                lines.append(
                    f"- **{c.get('name', '?')}** ({c.get('name_en', '')}): "
                    f"{c.get('definition', '')}"
                )
            return "\n".join(lines)
        return f"未找到与「{query}」相关的概念"

    def get_stats(self) -> str:
        """获取学习统计"""
        stats = self.memory.get_learning_stats()
        active = self.state_machine.get_active_count()
        history = self.state_machine.get_history_summary(5)

        lines = [
            "## NoteWeaver 统计",
            f"- 总笔记数: {stats['total_notes']}",
            f"- 知识图谱概念: {stats['kg_concepts']}",
            f"- 关系数: {stats['kg_relations']}",
            f"- 活跃任务: {active}",
            f"- 薄弱点: {', '.join(stats['weak_points'][:5]) or '无'}",
        ]
        if history:
            lines.append("\n### 最近处理")
            for h in history[:5]:
                lines.append(f"- {h['file_name']} (QA={h['qa_score']})")

        return "\n".join(lines)

    # =================================================================
    # 内部辅助方法
    # =================================================================

    @staticmethod
    def _get_rel_subdir(video_path: str) -> str:
        """计算视频相对于 source_video_dir 的子目录路径

        例如: video_path = "E:/.../Video/1.工艺速通/02_衬底.mp4"
              返回 "1.工艺速通"

        如果视频不在 source_video_dir 下，返回 ""（平铺，向后兼容）
        """
        src_dir = os.path.join(config.base_dir, config.source_video_dir)
        src_dir = os.path.abspath(src_dir)
        try:
            rel = os.path.relpath(os.path.dirname(video_path), src_dir)
            if rel.startswith(".."):
                return ""
            return rel if rel != "." else ""
        except ValueError:
            return ""

    def _save_txt(self, file_base: str, raw_text: str, rel_subdir: str = ""):
        """保存原始转录 TXT（按子目录组织）"""
        txt_dir = os.path.join(config.txt_dir, rel_subdir)
        os.makedirs(txt_dir, exist_ok=True)
        formatted = re.sub(r'([。！？；])', r'\1\n', raw_text)
        txt_path = os.path.join(txt_dir, f"{file_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted.strip())
        logger.info(f"TXT 已保存: {txt_path}")

