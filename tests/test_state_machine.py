"""测试 TaskStateMachine — 任务生命周期管理"""
import pytest
from note_weaver.core.state_machine import Task, TaskStatus, TaskStateMachine


class TestTask:
    """单个任务的数据结构测试"""

    def test_task_creation(self):
        """创建时默认为 PENDING"""
        task = Task(video_path="/path/to/video.mp4", file_name="video.mp4")
        assert task.status == TaskStatus.PENDING
        assert task.task_id  # 自动生成
        assert len(task.task_id) == 8

    def test_task_elapsed_zero_when_not_started(self):
        """未开始的任务 elapsed = 0"""
        task = Task()
        assert task.elapsed_seconds == 0.0

    def test_task_is_not_finished_initially(self):
        """刚创建的任务不是 finished"""
        task = Task()
        assert not task.is_finished

    def test_task_is_finished_after_completion(self):
        """COMPLETED 状态 → is_finished = True"""
        task = Task()
        task.status = TaskStatus.COMPLETED
        assert task.is_finished

    def test_task_is_finished_after_failure(self):
        """FAILED 状态 → is_finished = True"""
        task = Task()
        task.status = TaskStatus.FAILED
        assert task.is_finished

    def test_to_dict_includes_key_fields(self):
        """to_dict 包含关键信息"""
        task = Task(
            video_path="test.mp4",
            file_name="test.mp4",
            qa_score=8.5,
        )
        d = task.to_dict()
        assert d["task_id"] == task.task_id
        assert d["file_name"] == "test.mp4"
        assert d["qa_score"] == 8.5
        assert "elapsed" in d


class TestTaskStateMachine:
    """状态机行为测试"""

    def test_create_task(self):
        """create_task 返回有效 Task"""
        sm = TaskStateMachine()
        task = sm.create_task("/data/video.mp4")
        assert task.file_name == "video.mp4"
        assert task.file_base == "video"
        assert task.task_id in sm.tasks

    def test_active_count(self):
        """活跃任务计数"""
        sm = TaskStateMachine()
        sm.create_task("a.mp4")
        sm.create_task("b.mp4")
        assert sm.get_active_count() == 2

    def test_transition_to_completed_adds_history(self):
        """完成后加入 history，从 tasks 移除"""
        sm = TaskStateMachine()
        task = sm.create_task("test.mp4")
        sm.transition(task, TaskStatus.COMPLETED)
        assert task in sm.history
        assert task.task_id not in sm.tasks

    def test_transition_to_failed_adds_history(self):
        """失败也加入 history"""
        sm = TaskStateMachine()
        task = sm.create_task("test.mp4")
        sm.transition(task, TaskStatus.FAILED)
        assert task in sm.history

    def test_find_processed_returns_completed(self):
        """find_processed 找到已完成的同名文件"""
        sm = TaskStateMachine()
        t1 = sm.create_task("lecture.mp4")
        sm.transition(t1, TaskStatus.COMPLETED)
        found = sm.find_processed("lecture.mp4")
        assert found is t1

    def test_find_processed_ignores_failed(self):
        """find_processed 忽略失败任务"""
        sm = TaskStateMachine()
        t1 = sm.create_task("lecture.mp4")
        sm.transition(t1, TaskStatus.FAILED)
        found = sm.find_processed("lecture.mp4")
        assert found is None

    def test_history_summary(self):
        """get_history_summary 返回 dict 列表"""
        sm = TaskStateMachine()
        t = sm.create_task("test.mp4")
        sm.transition(t, TaskStatus.COMPLETED)
        summary = sm.get_history_summary(limit=10)
        assert len(summary) == 1
        assert summary[0]["task_id"] == t.task_id

    def test_full_lifecycle(self):
        """完整生命周期：创建→提取→转录→排版→完成"""
        sm = TaskStateMachine()
        task = sm.create_task("lecture.mp4")

        sm.transition(task, TaskStatus.EXTRACTING)
        assert task.started_at is not None

        sm.transition(task, TaskStatus.TRANSCRIBING)
        sm.transition(task, TaskStatus.COMPOSING)
        sm.transition(task, TaskStatus.COMPLETED)

        assert task.is_finished
        assert task in sm.history
        assert sm.get_active_count() == 0
