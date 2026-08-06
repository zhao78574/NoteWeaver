"""DAG 管线执行器 — 声明式阶段表 + 拓扑调度 + 各阶段实现

从 Orchestrator 拆分出来，负责 DAG 的调度执行和所有 stage 方法。
Orchestrator 通过 self.dag_runner 委托 DAG 执行。
"""

from __future__ import annotations

import os
import re
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from note_weaver.agents.orchestrator import Orchestrator

from note_weaver.core.state_machine import TaskStatus
from note_weaver.core.job import Job, PipelineType
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger


class DagRunner:
    """DAG 管线执行器 — 拓扑调度 + 所有 Stage 方法

    持有 Orchestrator 引用以访问共享的 Agent 实例和工具方法。
    """

    # 管线定义：每个 pipeline 是一个 stage 列表
    #   deps:      前置阶段 ID 列表，用于拓扑排序
    #   fn:        本类的方法名（不含 self）
    #   condition: 可选，lambda(ctx) → bool，决定是否执行
    #   parallel:  是否与同批就绪的 stage 并行执行
    #   loopback:  是否支持回流（QA 不通过时 re-enqueue 前置 stage）
    PIPELINE_DEFS = {
        PipelineType.FULL_NOTE: {
            "stages": [
                {"id": "prepare",      "deps": [],          "fn": "_stage_prepare"},
                {"id": "extract",      "deps": ["prepare"], "fn": "_stage_extract"},
                # ── 并行阶段 ──
                {"id": "transcribe",   "deps": ["extract"], "fn": "_stage_transcribe",
                 "parallel": True},
                {"id": "vision",       "deps": ["extract"], "fn": "_stage_vision",
                 "parallel": True,
                 "condition": lambda ctx: not ctx.get("is_audio_only", False)},
                {"id": "router",       "deps": ["extract"], "fn": "_stage_router",
                 "parallel": True,
                 "condition": lambda ctx: config.get("router.enabled", True)},
                # ── 纠错（依赖转写 + 路由完成） ──
                {"id": "corrector",    "deps": ["transcribe","router"], "fn": "_stage_corrector",
                 "condition": lambda ctx: config.get("corrector.enabled", True)},
                # ── 后续阶段 ──
                {"id": "chunk",        "deps": ["corrector","transcribe"], "fn": "_stage_chunk",
                 "parallel": True},
                {"id": "compose",      "deps": ["vision","router","corrector","chunk"],
                 "fn": "_stage_compose"},
                {"id": "qa",           "deps": ["compose"],  "fn": "_stage_qa",
                 "loopback": "compose"},
                {"id": "save",         "deps": ["qa"],       "fn": "_stage_save"},
                {"id": "cleanup",      "deps": ["save"],     "fn": "_stage_cleanup"},
            ],
            "max_retries": 2,
            "fallback_classify": True,    # Router 不可用时回退到旧 Classifier
        },
        PipelineType.PDF_NOTE: {
            "stages": [
                {"id": "extract_pdf",  "deps": [],          "fn": "_stage_extract_pdf"},
                {"id": "vision_pdf",   "deps": ["extract_pdf"], "fn": "_stage_vision",
                 "condition": lambda ctx: bool(ctx.get("pdf_images", []))},
                {"id": "compose_pdf",  "deps": ["extract_pdf","vision_pdf"],
                 "fn": "_stage_compose_pdf"},
                {"id": "qa_pdf",       "deps": ["compose_pdf"], "fn": "_stage_qa_pdf",
                 "loopback": "compose_pdf"},
                {"id": "save",         "deps": ["qa_pdf"],    "fn": "_stage_save_pdf"},
            ],
            "max_retries": 2,
        },
        PipelineType.WEB_NOTE: {
            "stages": [
                {"id": "extract_web",  "deps": [],          "fn": "_stage_extract_web"},
                {"id": "compose_web",  "deps": ["extract_web"], "fn": "_stage_compose_web"},
                {"id": "qa_web",       "deps": ["compose_web"], "fn": "_stage_qa_web",
                 "loopback": "compose_web"},
                {"id": "save",         "deps": ["qa_web"],    "fn": "_stage_save_web"},
            ],
            "max_retries": 2,
        },
    }

    def __init__(self, orchestrator: Orchestrator):
        self.orch = orchestrator

    # ── 便捷属性（代理到 Orchestrator 共享的 Agent 实例） ──────────

    @property
    def state_machine(self):
        return self.orch.state_machine

    @property
    def cache(self):
        return self.orch.cache

    @property
    def transcriber(self):
        return self.orch.transcriber

    @property
    def vision(self):
        return self.orch.vision

    @property
    def composer(self):
        return self.orch.composer

    @property
    def qa(self):
        return self.orch.qa

    @property
    def memory(self):
        return self.orch.memory

    @property
    def router(self):
        return self.orch.router

    @property
    def corrector(self):
        return self.orch.corrector

    @property
    def policy(self):
        return self.orch.policy

    @property
    def keyword_mgr(self):
        return self.orch.keyword_mgr

    @property
    def classifier(self):
        return self.orch.classifier

    # =================================================================
    # DAG 调度引擎
    # =================================================================

    def _run_dag(self, pipeline_type: PipelineType, ctx: dict) -> dict:
        """执行 DAG pipeline

        Args:
            pipeline_type: PipelineType 枚举
            ctx: 上下文 dict，各 stage 读取/写入

        Returns:
            结果 dict
        """
        pipeline_def = self.PIPELINE_DEFS.get(pipeline_type)
        if not pipeline_def:
            raise ValueError(f"未知 pipeline: {pipeline_type}")

        stages = pipeline_def["stages"]
        max_retries = pipeline_def.get("max_retries", 0)

        # 构建入度表和依赖图
        deps_in = {s["id"]: set(s.get("deps", [])) for s in stages}
        dep_graph = {s["id"]: s for s in stages}

        def _get_ready(dep_in, completed):
            """返回所有依赖已满足且未完成的 stage"""
            ready = []
            for sid, deps in dep_in.items():
                if sid in completed:
                    continue
                if deps.issubset(completed):
                    # 检查 condition
                    stage_def = dep_graph[sid]
                    cond = stage_def.get("condition")
                    if cond is None or cond(ctx):
                        ready.append(stage_def)
            return ready

        completed = set()
        loopback_count = 0
        ctx["results"] = {}

        # 将 on_token 注入 ctx，供 _stage_compose 使用
        progress_cb = self.orch._progress_callback
        if progress_cb and progress_cb.get("on_token"):
            ctx["_on_token"] = progress_cb["on_token"]

        pool = ThreadPoolExecutor(max_workers=4)
        try:
            while len(completed) < len(stages):
                ready = _get_ready(deps_in, completed)
                if not ready:
                    # 没有就绪的 stage 但还有未完成的 → 死锁
                    remaining = set(deps_in.keys()) - completed
                    if remaining:
                        raise RuntimeError(f"DAG 死锁: 剩余 {remaining} 无法就绪")
                    break

                # 提交就绪的 stage（含缓存检查）
                futures = {}
                input_path = ctx.get("video_path") or ctx.get("job.input", "")
                for stage_def in ready:
                    sid = stage_def["id"]
                    # 查询缓存（仅缓存有副作用、且昂贵的 stage）
                    use_cache = sid in ("transcribe", "vision", "classify", "chunk")
                    cached_result = None
                    if use_cache and input_path:
                        cache_params = self._cache_params_for_stage(sid, ctx)
                        cached_result = self.cache.get(input_path, sid, cache_params)

                    if cached_result is not None:
                        # 缓存命中，跳过执行
                        ctx["results"][sid] = cached_result
                        completed.add(sid)
                        logger.info(f"[DAG] {sid} 从缓存加载完成")

                        # 缓存旁路补偿：还原 stage 对 ctx 的副作用
                        if sid == "vision":
                            vision_results = []
                            if isinstance(cached_result, dict):
                                # 新缓存格式（含完整结果）
                                vision_results = cached_result.get("vision_results", [])
                            elif isinstance(cached_result, list):
                                # 兼容：直接列表
                                vision_results = cached_result
                            if not vision_results:
                                logger.warning(
                                    f"[DAG] vision 缓存回补: 未找到 vision_results，"
                                    f"缓存类型={type(cached_result).__name__}"
                                )
                            ctx["vision_results"] = vision_results
                            task_obj = ctx.get("task")
                            if task_obj:
                                task_obj.vision_results = vision_results
                            img_count = len(vision_results) if isinstance(vision_results, list) else 0
                            logger.info(f"[DAG] vision 缓存旁路补偿: {img_count} 张")

                        continue

                    self.orch._emit_phase("start", sid)
                    fn = getattr(self, stage_def["fn"])
                    fut = pool.submit(fn, ctx)
                    futures[fut] = stage_def

                for future in as_completed(futures):
                    stage_def = futures[future]
                    sid = stage_def["id"]
                    try:
                        result = future.result()
                        ctx["results"][sid] = result
                        completed.add(sid)
                        self.orch._emit_phase("done", sid)
                        logger.info(f"[DAG] {sid} 完成")

                        # 写入缓存（仅缓存昂贵 stage）
                        use_cache = sid in ("transcribe", "vision", "classify", "chunk")
                        if use_cache and input_path:
                            cache_params = self._cache_params_for_stage(sid, ctx)
                            self.cache.set(input_path, sid, result, cache_params)

                        # loopback 处理
                        loop_target = stage_def.get("loopback")
                        if loop_target and result is not None:
                            score = result if isinstance(result, (int, float)) else 0
                            if score == 0:
                                loopback_count += 1
                                if loopback_count <= max_retries:
                                    # 重置 compose 到未完成
                                    to_reset = {loop_target}
                                    # 同时也重置 QA
                                    for s in stages:
                                        if s.get("loopback") == loop_target:
                                            to_reset.add(s["id"])
                                    for sid_rm in to_reset:
                                        completed.discard(sid_rm)
                                    logger.info(
                                        f"[DAG] loopback #{loopback_count}: {sid} → "
                                        f"重置 {to_reset}, 重新调度"
                                    )
                    except Exception as e:
                        logger.error(f"[DAG] {sid} 失败: {e}")
                        self.orch._emit_phase("error", sid, str(e))
                        import traceback
                        logger.error(traceback.format_exc())
                        ctx["error"] = str(e)
                        progress_cb = self.orch._progress_callback
                        if progress_cb and progress_cb.get("on_error"):
                            progress_cb["on_error"](f"[{sid}] {e}")
                        raise

        finally:
            pool.shutdown(wait=False)
        return ctx["results"]

    @staticmethod
    def _cache_params_for_stage(sid: str, ctx: dict) -> Optional[dict]:
        """为每个 stage 生成缓存参数（模型/配置差异 → 不同缓存键）"""
        params = {}
        try:
            if sid == "transcribe":
                params["whisper_model"] = config.get("whisper.model_size", "small")
            elif sid == "vision" or sid == "vision_pdf":
                params["max_images"] = config.get("vision.max_images_per_batch", 10)
            elif sid == "extract":
                params["keyframe_strategy"] = "hybrid"
            elif sid == "router":
                params["enabled"] = config.get("router.enabled", True)
                params["density_check"] = config.get("router.visual_density_check", True)
            elif sid == "corrector":
                params["enabled"] = config.get("corrector.enabled", True)
                params["model"] = config.get("corrector.model", "fast")
            elif sid == "classify":
                params["model"] = config.get("api.deepseek.model_fast", "deepseek-chat")
        except Exception:
            pass
        return params if params else None

    # =================================================================
    # FULL_NOTE 入口
    # =================================================================

    def run_full_note_dag(self, job: Job) -> dict:
        """FULL_NOTE 管线的 DAG 入口 — 设置 ctx 并执行 _run_dag"""
        ctx = {"video_path": job.input, "job": job}
        cb = self.orch._progress_callback
        try:
            self.orch._emit_phase("start", "dag", "开始处理视频")
            results = self._run_dag(PipelineType.FULL_NOTE, ctx)
            task = ctx.get("task")
            if ctx.get("skipped") and task:
                self.orch._emit_phase("done", "dag", "视频已处理过，跳过")
                return {"ok": True, "skipped": True, "task_id": task.task_id}
            if "save" in results and results["save"].get("md_path"):
                md_path = results["save"]["md_path"]
                qa_score = results.get("cleanup", {}).get("qa_score", 0)
                self.orch._emit_phase("done", "dag", "处理完成")
                if cb and cb.get("on_complete"):
                    cb["on_complete"]({
                        "ok": True, "qa_score": qa_score, "note_path": md_path,
                    })
                return {
                    "ok": True, "qa_score": qa_score, "note_path": md_path,
                    "task_id": results["cleanup"].get("task_id", ""),
                }
            if ctx.get("error"):
                if cb and cb.get("on_error"):
                    cb["on_error"](ctx["error"])
                return {"ok": False, "error": ctx["error"]}
            return {"ok": True, "results": {k: v for k, v in results.items()
                    if isinstance(v, dict)}}
        except KeyboardInterrupt:
            self.orch._emit_phase("error", "dag", "用户中断")
            logger.warning("[DAG] 用户中断处理")
            return {"ok": False, "error": "用户中断"}
        except Exception as e:
            self.orch._emit_phase("error", "dag", str(e))
            import traceback
            logger.error(f"[DAG] FULL_NOTE 执行失败: {e}")
            logger.error(traceback.format_exc())
            return {"ok": False, "error": str(e)}

    # =================================================================
    # DAG Stage 方法
    # =================================================================

    def _stage_prepare(self, ctx: dict) -> dict:
        """Step 0: 文件信息解析 + 去重检查 + 创建 Task"""
        video_path = ctx["video_path"]
        file_name = os.path.basename(video_path)
        file_base = os.path.splitext(file_name)[0].replace(' ', '_')
        for suffix in ('_Auto', '_auto', '_720P', '_1080P', '_final'):
            if file_base.endswith(suffix):
                file_base = file_base[:-len(suffix)]
                file_name = file_base + os.path.splitext(file_name)[1]
                break

        rel_subdir = self.orch._get_rel_subdir(video_path)
        if rel_subdir:
            logger.info(f"[DIR]  输出子目录: {rel_subdir}")
            os.makedirs(os.path.join(config.txt_dir, rel_subdir), exist_ok=True)
            os.makedirs(os.path.join(config.note_dir, rel_subdir), exist_ok=True)

        existing = self.state_machine.find_processed(file_name)
        if existing:
            logger.info(f"[SKIP] 跳过已处理: {file_name}")
            ctx["skipped"] = True
            ctx["task"] = existing
            return {"skipped": True, "task_id": existing.task_id}

        task = self.state_machine.create_task(video_path)
        logger.info(f"=" * 60)
        logger.info(f"[START] 开始处理: {file_name} (task={task.task_id})")
        logger.info(f"=" * 60)

        audio_tmp = os.path.join(config.base_dir, f"temp_audio_{file_base}.mp3")
        is_audio_only = file_name.lower().endswith(
            ('.m4a', '.mp3', '.wav', '.flac', '.aac')
        )
        screenshot_dir = os.path.join(config.note_dir, rel_subdir, file_base)

        ctx.update({
            "video_path": video_path, "file_name": file_name, "file_base": file_base,
            "rel_subdir": rel_subdir, "task": task, "audio_tmp": audio_tmp,
            "is_audio_only": is_audio_only, "screenshot_dir": screenshot_dir,
            "screenshot_files": [], "skipped": False,
        })
        return {"task_id": task.task_id, "file_name": file_name}

    def _stage_extract(self, ctx: dict) -> dict:
        """Step 1: 提取音频 + 截图 (并行)"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.EXTRACTING)

        from note_weaver.core.extractor import (
            extract_audio, extract_keyframes, clean_screenshot_dir, get_video_duration,
        )

        if ctx["is_audio_only"]:
            logger.info("检测到纯音频文件，跳过截图")
            extract_audio(ctx["video_path"], ctx["audio_tmp"])
            task.audio_path = ctx["audio_tmp"]
            task.screenshot_dir = ctx["screenshot_dir"]
            task.screenshot_files = []
            ctx["screenshot_files"] = []
            return {"audio_only": True}

        clean_screenshot_dir(ctx["screenshot_dir"], ctx["file_base"])

        # 自适应截图间隔
        dur = get_video_duration(ctx["video_path"])
        ctx["duration"] = dur
        if dur <= 300:
            interval = 30
        elif dur <= 900:
            interval = 60
        elif dur <= 1800:
            interval = 90
        elif dur <= 3600:
            interval = 120
        else:
            interval = 180

        # 使用智能帧选择（hybrid 策略：scene change + 均匀采样）
        strategy = "hybrid"
        logger.info(
            f"[Keyframe] 策略={strategy}, 间隔={interval}s "
            f"(视频 {dur:.0f}s)"
        )

        from concurrent.futures import ThreadPoolExecutor as _TPool
        ext_pool = _TPool(max_workers=2)
        audio_future = ext_pool.submit(extract_audio, ctx["video_path"], ctx["audio_tmp"])
        screenshot_files = []
        screen_future = ext_pool.submit(
            extract_keyframes, ctx["video_path"],
            ctx["screenshot_dir"], ctx["file_base"],
            strategy, interval,
        )
        try:
            audio_future.result()
            screenshot_files = screen_future.result()
        except KeyboardInterrupt:
            logger.warning("[Extract] 用户中断提取，跳过结果收集")
            ctx["skipped"] = True
            ext_pool.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as e:
            logger.error(f"[Extract] 提取失败: {e}")
            ext_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            ext_pool.shutdown(wait=False)

        task.audio_path = ctx["audio_tmp"]
        task.screenshot_dir = ctx["screenshot_dir"]
        task.screenshot_files = screenshot_files
        ctx["screenshot_files"] = screenshot_files
        ctx["duration"] = dur

        logger.info(f"[OK] 提取完成: 音频 + {len(screenshot_files)} 张截图")
        return {"audio_only": False, "screenshot_count": len(screenshot_files)}

    def _stage_transcribe(self, ctx: dict) -> dict:
        """Step 2a: 语音转录"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.TRANSCRIBING)
        transcript = self.transcriber.execute(task.audio_path)
        task.transcript = transcript
        ctx["transcript"] = transcript
        logger.info(f"[OK] 转录完成: {len(transcript.get('raw_text', ''))}字")
        return {"text_length": len(transcript.get("raw_text", ""))}

    def _stage_vision(self, ctx: dict) -> dict:
        """Step 2b: Vision 截图分析"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.VISION_ANALYZING)
        screenshot_files = ctx.get("screenshot_files", [])
        if screenshot_files:
            vision_results = self.vision.execute(screenshot_files)
        else:
            vision_results = []
        task.vision_results = vision_results
        ctx["vision_results"] = vision_results
        logger.info(f"[OK] 视觉分析完成: {len(vision_results)}张")
        return {"vision_count": len(vision_results), "vision_results": vision_results}

    def _stage_router(self, ctx: dict) -> dict:
        """Step 2c: Router 路由分析（与 Transcribe/Vision 并行）

        快速分析视频领域/内容结构/视觉密度，产出 routing 信号。
        不依赖 Transcribe 结果——若 Transcribe 未完成，自动用
        tiny 模型独立转写前30s。
        """
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.ROUTING)

        # 准备 Router 输入
        transcript = ctx.get("transcript") or task.transcript or {}
        segments = transcript.get("segments", [])
        audio_first_30s = ""
        if segments:
            audio_first_30s = " ".join(
                seg["text"] for seg in segments if seg["start"] <= 30
            )
        if not audio_first_30s.strip() and segments:
            audio_first_30s = segments[0]["text"]

        duration = ctx.get("duration", 0)

        # 执行 Router（传入 audio_tmp 用于独立转写，不再依赖 Transcribe）
        router_result = self.router.execute(
            video_path=ctx.get("video_path", ""),
            file_name=ctx.get("file_name", ""),
            audio_first_30s=audio_first_30s,
            duration=duration,
            screenshot_dir=ctx.get("screenshot_dir", ""),
            file_base=ctx.get("file_base", ""),
            audio_path=ctx.get("audio_tmp", ""),
        )

        # Policy Engine: Router 输出 → 管线参数
        policy_result = self.policy.compute(router_result)

        # 合并动态热词（自动扩展的领域高频词）
        try:
            domain = router_result.get("domain", "general")
            merged = self.keyword_mgr.merge_with_static(
                domain, policy_result.get("correction_keywords", []),
            )
            policy_result["correction_keywords"] = merged
            if len(merged) > len(router_result.get("keywords", [])):
                logger.debug(
                    f"[KeywordMgr] 动态合并: {domain} "
                    f"共 {len(merged)} 词（含动态词）"
                )
        except Exception as e:
            logger.debug(f"[KeywordMgr] 动态合并跳过（非致命）: {e}")

        ctx["router_result"] = router_result
        ctx["policy_result"] = policy_result
        ctx["classification"] = {
            "domain": router_result.get("domain", "general"),
            "type": router_result.get("content_structure", "lecture"),
            "visual_density": router_result.get("visual_density", "medium"),
            "keywords": router_result.get("keywords", []),
            "suggested_strategy": {
                "screenshot_interval": policy_result.get("frame_interval", 60),
                "note_style": policy_result.get("note_style", "detailed"),
                "focus_areas": policy_result.get("correction_keywords", []),
            },
        }

        task.classification = ctx["classification"]

        logger.info(
            f"[Router] {ctx['file_name']} → "
            f"domain={router_result.get('domain')}, "
            f"structure={router_result.get('content_structure')}, "
            f"density={router_result.get('visual_density')}"
        )
        return {
            "domain": router_result.get("domain", ""),
            "structure": router_result.get("content_structure", ""),
            "density": router_result.get("visual_density", ""),
        }

    def _stage_corrector(self, ctx: dict) -> dict:
        """Step 3a: 转录纠错（领域感知）

        在 Transcribe 之后、Chunk 之前执行，使用 Router 的领域信号。
        """
        if ctx.get("skipped"):
            return {"skipped": True}
        if not config.corrector_enabled:
            logger.info("[Corrector] 已禁用，跳过纠错")
            return {"skipped": True, "reason": "disabled"}

        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.CORRECTING)

        transcript = ctx.get("transcript") or task.transcript or {}
        segments = transcript.get("segments", [])
        raw_text = transcript.get("raw_text", "")

        if not segments:
            logger.info("[Corrector] 无段落可纠错，跳过")
            return {"skipped": True, "reason": "no_segments"}

        # 从 Router/Policy 获取领域关键词
        policy_result = ctx.get("policy_result", {})
        domain_keywords = policy_result.get("correction_keywords", [])
        router_result = ctx.get("router_result", {})
        domain = router_result.get("domain", "general")

        # 执行纠错
        corrected = self.corrector.execute(
            segments=segments,
            domain_keywords=domain_keywords,
            domain=domain,
            raw_text=raw_text,
        )

        ctx["corrected_transcript"] = corrected

        # 同时更新 task.transcript 的引用（向下兼容）
        if corrected.get("segments"):
            task.transcript = {
                "timestamped": corrected.get("timestamped", transcript.get("timestamped", "")),
                "raw_text": corrected.get("raw_text", raw_text),
                "segments": corrected.get("segments", segments),
                "duration": transcript.get("duration", 0),
                "language": transcript.get("language", "zh"),
            }

        corrections = corrected.get("corrections_count", 0)
        logger.info(f"[Corrector] 完成: {corrections} 处修正")
        return {"corrections_count": corrections}

    def _stage_classify(self, ctx: dict) -> dict:
        """Step 2c: 视频分类"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.CLASSIFYING)

        segments = (ctx.get("transcript") or task.transcript or {}).get("segments", [])
        first_30s_text = " ".join(
            seg["text"] for seg in segments if seg["start"] <= 30
        )
        if not first_30s_text.strip() and segments:
            first_30s_text = segments[0]["text"]

        duration = ctx.get("duration", 0)
        classification = self.classifier.execute(
            filename=ctx["file_name"],
            audio_sample=first_30s_text,
            duration=duration,
        )
        task.classification = classification
        ctx["classification"] = classification
        return {"domain": classification.get("domain", "")}

    def _stage_chunk(self, ctx: dict) -> dict:
        """Step 3: 保存转录文件（优先使用纠错后的文本）"""
        if ctx.get("skipped"):
            return {"skipped": True}

        # 优先使用纠错后的转录
        corrected = ctx.get("corrected_transcript", {})
        if corrected and corrected.get("raw_text"):
            raw_text = corrected["raw_text"]
        else:
            transcript = ctx.get("transcript") or ctx["task"].transcript or {}
            raw_text = transcript.get("raw_text", "")
        self.orch._save_txt(ctx["file_base"], raw_text, ctx.get("rel_subdir", ""))
        txt_path = os.path.join(
            config.txt_dir, ctx.get("rel_subdir", ""), f"{ctx['file_base']}.txt"
        )
        ctx["task"].txt_path = txt_path

        # 保存转录 JSON
        transcript_path = txt_path.replace(".txt", "_transcript.json")
        try:
            transcript = ctx.get("transcript") or ctx["task"].transcript or {}
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[DagRunner] 转录 JSON 保存失败（非致命）: {e}")
        return {"txt_path": txt_path}

    def _stage_compose(self, ctx: dict) -> dict:
        """Step 4a: Composer 排版（支持流式输出）"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.COMPOSING)

        classification = ctx.get("classification", {}) or task.classification or {}
        domain = classification.get("domain", "")
        difficulty = classification.get("difficulty", "")

        # ── 使用纠错后的转录（如有） ──
        corrected = ctx.get("corrected_transcript", {})
        if corrected and corrected.get("segments"):
            transcript = {
                "timestamped": corrected.get("timestamped", ""),
                "raw_text": corrected.get("raw_text", ""),
                "segments": corrected.get("segments", []),
            }
        else:
            transcript = ctx.get("transcript") or task.transcript or {}

        vision_results = ctx.get("vision_results") or task.vision_results or []
        segments = transcript.get("segments", [])

        user_context = self.memory.get_context(domain=domain, difficulty=difficulty)

        policy_result = ctx.get("policy_result", {})
        if policy_result:
            strategy = {
                "note_style": policy_result.get("note_style", "detailed"),
                "focus_areas": policy_result.get("correction_keywords", [])[:5],
                "screenshot_interval": policy_result.get("frame_interval", 60),
            }
        else:
            strategy = classification.get("suggested_strategy", {})

        retry_count = ctx.get("retry_count", 0)
        revision_feedback = ""
        defects = None
        qa_report = ctx.get("qa_report") or ctx.get("results", {}).get("qa")
        if retry_count > 0 and qa_report:
            if isinstance(qa_report, dict):
                defects = qa_report.get("defects")
                revision_feedback = qa_report.get("revision_suggestions", "")
            logger.info(
                f"[DAG] compose loopback #{retry_count}, "
                f"缺陷={len(defects) if defects else '文本反馈'}"
            )

        # ── 检测是否需要流式输出 ──
        on_token = ctx.get("_on_token")
        if on_token:
            note_content = self.composer.stream_execute(
                file_base=ctx["file_base"],
                timestamped_text=transcript.get("timestamped", ""),
                vision_results=vision_results,
                on_token=on_token,
                strategy=strategy,
                user_context=user_context,
                revision_feedback=revision_feedback,
                defects=defects,
                segments=segments,
            )
        else:
            note_content = self.composer.execute(
                file_base=ctx["file_base"],
                timestamped_text=transcript.get("timestamped", ""),
                vision_results=vision_results,
                strategy=strategy,
                user_context=user_context,
                revision_feedback=revision_feedback,
                defects=defects,
                segments=segments,
            )
        task.note_content = note_content
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_qa(self, ctx: dict) -> float:
        """Step 4b: QA 质检 — 返回分数，0 分触发 loopback"""
        if ctx.get("skipped"):
            return 10.0  # 跳过则直接过
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.QA_REVIEWING)

        note_content = ctx.get("note_content") or task.note_content or ""
        transcript = ctx.get("transcript") or task.transcript or {}
        vision_results = ctx.get("vision_results") or task.vision_results or []
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=transcript.get("raw_text", ""),
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        task.qa_report = qa_report
        task.qa_score = qa_report.get("total", 0)
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[OK] QA通过 (score={score})")
            return score
        else:
            logger.warning(f"[FAIL] QA不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0  # 0 分 = 触发 loopback

    def _stage_save(self, ctx: dict) -> dict:
        """Step 5: 保存笔记 + 更新记忆"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        note_content = ctx.get("note_content") or task.note_content or ""
        classification = ctx.get("classification") or task.classification or {}

        note_output_dir = os.path.join(
            config.note_dir, ctx.get("rel_subdir", "")
        )
        md_path = self.composer.save_note(ctx["file_base"], note_content, note_output_dir)
        task.md_path = md_path
        self.orch._export_outputs(md_path)

        qa_report = ctx.get("qa_report") or task.qa_report or {}
        try:
            self.memory.update_after_note(
                file_base=ctx["file_base"],
                note_content=note_content,
                classification=classification,
                qa_report=qa_report,
            )
        except Exception as mem_err:
            logger.warning(f"[Memory] 更新失败（非致命）: {mem_err}")

        self.orch._rebuild_indexes()
        return {"md_path": md_path}

    def _stage_cleanup(self, ctx: dict) -> dict:
        """Step 6: 清理临时文件 + 标记完成"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        audio_tmp = ctx.get("audio_tmp", "")
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)
        self.state_machine.transition(task, TaskStatus.COMPLETED)
        logger.info(
            f"[DONE] 处理完成: {ctx['file_name']} "
            f"(耗时 {task.elapsed_seconds:.0f}s, QA={task.qa_score})"
        )

        # ── 自动热词扩展（从转录文本提取领域高频词） ──
        try:
            transcript = ctx.get("transcript") or task.transcript or {}
            raw_text = transcript.get("raw_text", "")
            router_result = ctx.get("router_result") or {}
            domain = router_result.get("domain", "")
            if raw_text and domain and len(raw_text) > 50:
                self.keyword_mgr.update_from_transcript(raw_text, domain)
        except Exception as e:
            logger.debug(f"[KeywordMgr] 热词更新跳过（非致命）: {e}")
        return {
            "task_id": task.task_id,
            "qa_score": task.qa_score,
            "md_path": task.md_path,
        }

    # ── PDF/Web stages ──

    def _stage_qa_pdf(self, ctx: dict) -> float:
        """PDF: QA 质检 + loopback 支持"""
        note_content = ctx.get("note_content", "")
        vision_results = ctx.get("vision_results", [])
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=ctx.get("pdf_text", "")[:15000],
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[PDF-QA] 通过 (score={score})")
            return score
        else:
            logger.warning(f"[PDF-QA] 不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0

    def _stage_qa_web(self, ctx: dict) -> float:
        """Web: QA 质检 + loopback 支持"""
        note_content = ctx.get("note_content", "")
        vision_results = ctx.get("vision_results", [])
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=ctx.get("web_text", "")[:15000],
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[Web-QA] 通过 (score={score})")
            return score
        else:
            logger.warning(f"[Web-QA] 不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0

    def _stage_extract_pdf(self, ctx: dict) -> dict:
        from note_weaver.utils.extractors import extract_from_pdf
        config.setup_proxy()
        p = pathlib.Path(ctx["job"].input)
        file_base = p.stem.replace(' ', '_')
        note_dir = pathlib.Path(config.base_dir) / "data" / "Note" / "pdf"
        img_dir = note_dir / file_base
        result = extract_from_pdf(ctx["job"].input, output_dir=str(img_dir))
        ctx.update({
            "file_base": file_base, "note_dir": str(note_dir),
            "pdf_text": result["text"], "pdf_images": result["images"],
        })
        logger.info(f"[PDF] 提取: {p.name} → {len(result['text'])}字, {len(result['images'])}张图")
        return {"text_len": len(result["text"]), "image_count": len(result["images"])}

    def _stage_compose_pdf(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        vision_results = ctx.get("vision_results", [])
        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=ctx["file_base"],
            timestamped_text=ctx["pdf_text"][:15000],
            vision_results=vision_results,
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback="这是从PDF提取的内容，整理成结构化学习笔记。",
        )
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_save_pdf(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        from note_weaver.run import insert_images_by_content, _fix_broken_markdown_images
        composer = ComposerAgent()
        vision_results = ctx.get("vision_results", [])
        note_with_images = insert_images_by_content(
            ctx["note_content"], vision_results, ctx["file_base"]
        )
        note_with_images = _fix_broken_markdown_images(note_with_images)
        note_path = composer.save_note(ctx["file_base"], note_with_images, ctx["note_dir"])
        self.orch._export_outputs(note_path)
        self.orch._rebuild_indexes()
        return {"note_path": note_path}

    def _stage_extract_web(self, ctx: dict) -> dict:
        from note_weaver.utils.extractors import extract_from_url
        config.setup_proxy()
        import urllib.parse
        parsed = urllib.parse.urlparse(ctx["job"].input)
        domain = parsed.netloc.replace("www.", "")
        file_base = re.sub(r'[^\w一-鿿]+', '_', domain)[:30]
        note_dir = pathlib.Path(config.base_dir) / "data" / "Note" / "web"
        img_dir = note_dir / file_base
        result = extract_from_url(ctx["job"].input, output_dir=str(img_dir))
        ctx.update({
            "file_base": file_base, "note_dir": str(note_dir),
            "web_text": result["text"],
        })
        if result.get("images"):
            from note_weaver.agents.vision import VisionAgent
            ctx["vision_results"] = VisionAgent().execute(result["images"])
        else:
            ctx["vision_results"] = []
        logger.info(f"[Web] 提取: {result.get('title','?')[:40]} → {len(result['text'])}字")
        return {"title": result.get("title",""), "text_len": len(result["text"])}

    def _stage_compose_web(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=ctx["file_base"],
            timestamped_text=ctx["web_text"][:15000],
            vision_results=ctx.get("vision_results", []),
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback="来源网页整理成结构化学习笔记。",
        )
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_save_web(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        from note_weaver.run import insert_images_by_content, _fix_broken_markdown_images
        composer = ComposerAgent()
        note_with_images = insert_images_by_content(
            ctx["note_content"], ctx.get("vision_results", []), ctx["file_base"]
        )
        note_with_images = _fix_broken_markdown_images(note_with_images)
        note_path = composer.save_note(ctx["file_base"], note_with_images, ctx["note_dir"])
        self.orch._export_outputs(note_path)
        self.orch._rebuild_indexes()
        return {"note_path": note_path}
