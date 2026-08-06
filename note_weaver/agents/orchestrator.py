"""中央调度 Agent — 任务编排 + 状态管理 + 异常处理

用法:
    orchestrator = Orchestrator()

    # 统一入口（推荐）
    job = Job.from_input("lecture.mp4")
    result = orchestrator.run(job)

    # 向后兼容入口
    task = orchestrator.process_video("lecture.mp4")

架构:
    Orchestrator（调度层）
    ├── DagRunner（DAG 管线执行 + 所有 Stage 方法）
    ├── MergeService（合集/播放列表合并）
    └── Agent 懒加载属性（transcriber, vision, composer, qa, ...）
"""

import os
import re
import json
import pathlib
from typing import Dict, Optional, Callable
from datetime import datetime

from note_weaver.core.state_machine import Task, TaskStateMachine
from note_weaver.core.job import Job, Modality, PipelineType
from note_weaver.core.cache import PipelineCache
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger

from .classifier import ClassifierAgent
from .transcriber import TranscriberAgent
from .vision import VisionAgent
from .composer import ComposerAgent
from .qa import QAAgent
from .memory_agent import MemoryAgent
from .router import RouterAgent
from .corrector import CorrectorAgent
from .policy import PolicyEngine
from .keyword_manager import KeywordManager
from .dag_runner import DagRunner
from .merge_service import MergeService


class Orchestrator:
    """NoteWeaver 中央调度器 — 管理整个笔记生成流程

    职责（瘦身后）:
    - 任务分发（run() 入口）
    - Agent 实例管理（懒加载 + 共享）
    - 后处理工具方法（索引重建、图谱生成、输出导出）
    - 向后兼容接口（process_video）
    """

    def __init__(self):
        # 基础设施
        self.state_machine = TaskStateMachine()

        # Agent 实例（懒加载，被 DagRunner/MergeService 共享）
        self._classifier: Optional[ClassifierAgent] = None
        self._transcriber: Optional[TranscriberAgent] = None
        self._vision: Optional[VisionAgent] = None
        self._composer: Optional[ComposerAgent] = None
        self._qa: Optional[QAAgent] = None
        self._memory: Optional[MemoryAgent] = None
        self._router: Optional[RouterAgent] = None
        self._corrector: Optional[CorrectorAgent] = None
        self._policy: Optional[PolicyEngine] = None
        self._keyword_mgr: Optional[KeywordManager] = None

        # 缓存系统
        self.cache = PipelineCache()

        # 模板系统
        self.template_name = "semiconductor"

        # 流式输出回调
        self._progress_callback: Optional[Dict[str, Callable]] = None

        # 子模块
        self.dag_runner = DagRunner(self)
        self.merge_service = MergeService(self)

        # 确保输出目录存在
        for d in [config.txt_dir,
                   config.note_dir, config.memory_dir, config.log_dir]:
            os.makedirs(d, exist_ok=True)

        logger.info("[Orchestrator] NoteWeaver 初始化完成")

    def set_template(self, name: str):
        """切换模板"""
        self.template_name = name
        from note_weaver.utils.logger import logger
        logger.info(f"[Orchestrator] 切换模板: {name}")

    def set_progress_callback(self, callbacks: Optional[Dict[str, Callable]]):
        """设置流式输出回调

        Args:
            callbacks: {
                "on_phase": fn(phase_name: str, status: str, detail: str),
                "on_token": fn(token: str),
                "on_error": fn(error: str),
                "on_complete": fn(result: dict),
            }
        """
        self._progress_callback = callbacks

    def _emit_phase(self, status: str, phase: str, detail: str = ""):
        """发出阶段事件"""
        cb = self._progress_callback
        if cb and cb.get("on_phase"):
            cb["on_phase"](phase, status, detail)

    # =================================================================
    # 统一入口
    # =================================================================

    def run(self, job: Job, progress_callback: Optional[Dict[str, Callable]] = None) -> dict:
        """统一任务入口 — 根据 Job.modality + Job.pipeline 分发到具体流程

        Args:
            job: 任务定义（由 Job.from_input() 创建）
            progress_callback: 可选，流式输出回调 {"on_phase": fn, "on_token": fn, ...}

        Returns:
            结果字典，含 output_paths/qa_score/stats 等
        """
        if progress_callback:
            self.set_progress_callback(progress_callback)

        # ── 特殊命令（非管线流程） ──
        cmd = job.metadata.get("command", "")
        if cmd == "graph":
            return self._run_graph()
        if cmd == "stats":
            return self._run_stats()

        # ── 按 PipelineType 分发 ──
        if job.pipeline == PipelineType.FULL_NOTE:
            return self.dag_runner.run_full_note_dag(job)

        elif job.pipeline == PipelineType.PDF_NOTE:
            ctx = {"job": job}
            try:
                results = self.dag_runner._run_dag(PipelineType.PDF_NOTE, ctx)
                return {"ok": True, "type": "pdf",
                        "note_path": results.get("save", {}).get("note_path", "")}
            except Exception as e:
                logger.error(f"[DAG] PDF_NOTE 失败: {e}")
                return {"ok": False, "type": "pdf", "error": str(e)}

        elif job.pipeline == PipelineType.WEB_NOTE:
            ctx = {"job": job}
            try:
                results = self.dag_runner._run_dag(PipelineType.WEB_NOTE, ctx)
                return {"ok": True, "type": "web",
                        "note_path": results.get("save", {}).get("note_path", "")}
            except Exception as e:
                logger.error(f"[DAG] WEB_NOTE 失败: {e}")
                return {"ok": False, "type": "web", "error": str(e)}

        elif job.pipeline == PipelineType.REGENERATE:
            if job.metadata.get("search_mode"):
                return self._run_search_regenerate(job)
            note_path = self._regenerate_note(job)
            return {"ok": True, "type": "regenerate", "note_path": note_path}

        elif job.pipeline == PipelineType.BATCH_VIDEO:
            return self._run_batch_video(job)

        elif job.pipeline == PipelineType.BATCH_REGENERATE:
            return self._run_batch_regenerate(job)

        elif job.pipeline == PipelineType.MERGE_NOTES:
            return self.merge_service.merge_notes(job)

        elif job.pipeline == PipelineType.QA_ONLY:
            return self._run_qa(job)

        else:
            logger.warning(f"[Orchestrator] 未知 Pipeline: {job.pipeline}")
            return {"ok": False, "error": f"未知管线类型: {job.pipeline}"}

    @staticmethod
    def _task_to_result(task, job) -> dict:
        """将 process_video 返回的 Task 对象转为统一结果字典"""
        if task is None or task.status.value == "failed":
            return {
                "ok": False,
                "type": "process_video",
                "error": task.error_message if task else "处理失败",
                "task_id": task.task_id if task else None,
            }
        return {
            "ok": True,
            "type": "process_video",
            "qa_score": task.qa_score,
            "note_path": task.md_path,
            "txt_path": task.txt_path,
            "elapsed": f"{task.elapsed_seconds:.0f}s",
            "task_id": task.task_id,
        }

    # =================================================================
    # 子流程分发
    # =================================================================

    def _run_graph(self) -> dict:
        """生成知识图谱"""
        try:
            self._regenerate_graph_on_disk()
            return {"ok": True, "type": "graph", "message": "知识图谱已生成"}
        except Exception as e:
            return {"ok": False, "type": "graph", "error": str(e)}

    def _run_stats(self) -> dict:
        """获取学习统计"""
        return {"ok": True, "type": "stats", "data": self.get_stats()}

    def _run_qa(self, job: Job) -> dict:
        """问答模式"""
        from note_weaver.skills.chat import chat as chat_notes
        config.setup_proxy()
        result = chat_notes(job.input)
        return {"ok": True, "type": "qa", "data": result}

    def _process_pdf(self, job: Job) -> dict:
        """PDF→笔记（向后兼容，直接调用 DAG）"""
        ctx = {"job": job}
        try:
            results = self.dag_runner._run_dag(PipelineType.PDF_NOTE, ctx)
            return {"ok": True, "type": "pdf",
                    "note_path": results.get("save", {}).get("note_path", "")}
        except Exception as e:
            logger.error(f"[PDF] 处理失败: {e}")
            return {"ok": False, "type": "pdf", "error": str(e)}

    def _process_web(self, job: Job) -> dict:
        """网页→笔记（向后兼容，直接调用 DAG）"""
        ctx = {"job": job}
        try:
            results = self.dag_runner._run_dag(PipelineType.WEB_NOTE, ctx)
            return {"ok": True, "type": "web",
                    "note_path": results.get("save", {}).get("note_path", "")}
        except Exception as e:
            logger.error(f"[Web] 处理失败: {e}")
            return {"ok": False, "type": "web", "error": str(e)}

    def _run_search_regenerate(self, job: Job) -> dict:
        """自然语言搜索笔记后重排"""
        from note_weaver.run import _search_and_regenerate
        _search_and_regenerate(job.input, job.metadata.get("raw_input", job.input))
        return {"ok": True, "type": "regenerate", "message": f"搜索重排: {job.input}"}

    def _regenerate_note(self, job: Job) -> str:
        """单个笔记重排"""
        from note_weaver.run import cmd_regenerate_note
        cmd_regenerate_note(job.input)
        return job.input

    def _run_batch_video(self, job: Job) -> dict:
        """批量处理视频目录"""
        from note_weaver.run import cmd_batch
        cmd_batch(job.input)
        return {"ok": True, "type": "batch_video", "directory": job.input}

    def _run_batch_regenerate(self, job: Job) -> dict:
        """批量重排目录"""
        from note_weaver.run import cmd_batch_regenerate
        cmd_batch_regenerate(job.input)
        return {"ok": True, "type": "batch_regenerate", "directory": job.input}

    # =================================================================
    # Agent 懒加载属性（被 DagRunner/MergeService 通过 self.orch 共享）
    # =================================================================

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

    @property
    def router(self) -> RouterAgent:
        if self._router is None:
            self._router = RouterAgent()
        return self._router

    @property
    def corrector(self) -> CorrectorAgent:
        if self._corrector is None:
            self._corrector = CorrectorAgent()
        return self._corrector

    @property
    def policy(self) -> PolicyEngine:
        if self._policy is None:
            self._policy = PolicyEngine()
        return self._policy

    @property
    def keyword_mgr(self) -> KeywordManager:
        if self._keyword_mgr is None:
            self._keyword_mgr = KeywordManager()
        return self._keyword_mgr

    # =================================================================
    # 后处理工具方法
    # =================================================================

    def _rebuild_indexes(self):
        """后处理：重建索引 + 图谱"""
        self._regenerate_graph_on_disk()
        try:
            from note_weaver.utils.embeddings import EmbeddingIndex
            count = EmbeddingIndex().build(force=True)
            if count:
                logger.info(f"[Embedding] 索引重建完成: {count} 条")
        except Exception as e:
            logger.warning(f"[Embedding] 索引重建失败（非致命）: {e}")

    def _regenerate_graph_on_disk(self):
        """从模板复制图谱 HTML 并嵌入 JSON 数据（动态+备用双模式）"""
        template = pathlib.Path(__file__).resolve().parent.parent / "templates" / "knowledge_graph.html"
        kg_path = pathlib.Path(config.memory_dir) / "knowledge_graph.json"
        output = pathlib.Path(config.memory_dir) / "knowledge_graph.html"
        if not template.exists():
            logger.warning(f"[Graph] 图谱模板文件不存在: {template}")
            return
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            html = template.read_text(encoding="utf-8")
            # 嵌入 JSON 备用数据
            if kg_path.exists():
                kg_data = kg_path.read_text(encoding="utf-8")
                html = html.replace("__FALLBACK_DATA_PLACEHOLDER__", kg_data)
            output.write_text(html, encoding="utf-8")
            logger.info(f"[Graph] 图谱 HTML 已就绪: {output}")
        except Exception as e:
            logger.warning(f"[Graph] 图谱模板复制失败: {e}")

    def _export_outputs(self, note_path: str):
        """笔记完成后生成三种输出形态的元数据

        在笔记目录下生成 _outputs.json，供 CLI/Web UI 展示三态输出入口。
        """
        if not note_path:
            return
        note_dir = pathlib.Path(note_path).parent
        outputs = {
            "markdown": note_path,
            "generated_at": datetime.now().isoformat(),
            "outputs": {
                "markdown": {"path": note_path, "label": "阅读模式"},
                "graph": {"path": str(note_dir / "_knowledge_graph.html"),
                          "label": "图谱模式",
                          "command": "weaver graph"},
                "qa": {"label": "问答模式",
                       "command": "weaver ask \"你的问题\""},
            }
        }
        try:
            with open(note_dir / "_outputs.json", "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"导出 _outputs.json 失败: {e}")

    # =================================================================
    # 向后兼容接口
    # =================================================================

    def process_video(self, video_path: str) -> Optional[Task]:
        """处理单个视频的完整流程（向后兼容）

        内部使用 DAG pipeline 执行，返回 Task 对象。
        """
        job = Job(input=video_path, modality=Modality.VIDEO,
                  pipeline=PipelineType.FULL_NOTE)
        result = self.dag_runner.run_full_note_dag(job)
        task = None
        for t in list(self.state_machine.history):
            if t.video_path == video_path or t.file_name == os.path.basename(video_path):
                task = t
                break
        if task is None and self.state_machine.tasks:
            for t in self.state_machine.tasks.values():
                if t.video_path == video_path:
                    task = t
                    break
        return task

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
    # 内部辅助方法（被 DagRunner stage 方法调用）
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
            rel = rel.replace(' ', '_')
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
