"""任务状态机 — 追踪每个视频处理任务的生命周期"""

import enum
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


class TaskStatus(enum.Enum):
    PENDING = "pending"                   # 刚入队
    CLASSIFYING = "classifying"           # 分类中
    EXTRACTING = "extracting"             # 提取音频+截图
    TRANSCRIBING = "transcribing"         # 语音识别中
    VISION_ANALYZING = "vision_analyzing" # 视觉分析中
    COMPOSING = "composing"               # 排版中
    QA_REVIEWING = "qa_reviewing"         # 质检中
    RETRYING = "retrying"                 # 回退重排中
    COMPLETED = "completed"               # 成功完成
    FAILED = "failed"                     # 失败
    SKIPPED = "skipped"                   # 跳过（已处理过）


@dataclass
class Task:
    """单个视频处理任务"""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    video_path: str = ""
    file_name: str = ""
    file_base: str = ""
    status: TaskStatus = TaskStatus.PENDING

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 中间产物路径
    audio_path: str = ""
    screenshot_dir: str = ""
    screenshot_files: List[str] = field(default_factory=list)

    # 各阶段结果
    classification: Optional[Dict[str, Any]] = None
    transcript: Optional[Dict[str, Any]] = None        # {"timestamped": ..., "raw_text": ...}
    vision_results: Optional[List[Dict[str, Any]]] = None
    note_content: Optional[str] = None
    qa_report: Optional[Dict[str, Any]] = None
    qa_score: float = 0.0
    retry_count: int = 0

    # 错误信息
    error_message: str = ""
    error_traceback: str = ""

    # 输出路径
    txt_path: str = ""
    md_path: str = ""

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at:
            end = self.completed_at or datetime.now()
            return (end - self.started_at).total_seconds()
        return 0.0

    @property
    def is_finished(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_name": self.file_name,
            "status": self.status.value,
            "elapsed": f"{self.elapsed_seconds:.1f}s",
            "qa_score": self.qa_score,
            "retry_count": self.retry_count,
            "error": self.error_message[:200] if self.error_message else None,
        }

    def __repr__(self):
        return f"<Task {self.task_id} [{self.status.value}] {self.file_name}>"


class TaskStateMachine:
    """管理所有任务的状态流转"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.history: List[Task] = []  # 已完成的任务（含failed）

    def create_task(self, video_path: str) -> Task:
        """创建新任务"""
        import os
        file_name = os.path.basename(video_path)
        file_base = os.path.splitext(file_name)[0]

        task = Task(
            video_path=video_path,
            file_name=file_name,
            file_base=file_base,
        )
        self.tasks[task.task_id] = task
        return task

    def transition(self, task: Task, new_status: TaskStatus):
        """状态转换"""
        old = task.status
        task.status = new_status

        if old == TaskStatus.PENDING and new_status != TaskStatus.PENDING:
            task.started_at = datetime.now()

        if new_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
            task.completed_at = datetime.now()
            self.history.append(task)
            if task.task_id in self.tasks:
                del self.tasks[task.task_id]

    def get_active_count(self) -> int:
        return len(self.tasks)

    def get_history_summary(self, limit: int = 20) -> List[Dict]:
        return [t.to_dict() for t in self.history[-limit:]]

    def find_processed(self, file_name: str) -> Optional[Task]:
        """检查是否已处理过同名文件"""
        for t in self.history:
            if t.file_name == file_name and t.status == TaskStatus.COMPLETED:
                return t
        return None
