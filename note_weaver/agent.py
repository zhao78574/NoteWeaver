"""NoteWeaver Agent — 真正的自主 AI 助理

不是管线，不是工具。是一个会观察、会决策、会行动、会学习的 Agent。

用法:
    from note_weaver.agent import NoteWeaverAgent
    agent = NoteWeaverAgent()
    agent.run("帮我把这个视频转成笔记")
    agent.run("我最近学了什么？")
    agent.run("")  # 自主模式：扫描环境，主动建议
"""

import os, sys, json, time, threading, re, pathlib
from datetime import datetime, timedelta
from typing import Optional

from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.agents.orchestrator import Orchestrator
from note_weaver.skills.search import search as search_notes
from note_weaver.skills.chat import chat as chat_notes


# ============================================================
# Agent 核心：感知 → 决策 → 行动 → 学习
# ============================================================

class NoteWeaverAgent:
    """NoteWeaver — 自主笔记数字助理

    Agent 生命周期:
        perceive() → decide() → act() → learn() → respond()
    """

    def __init__(self):
        self.orchestrator = Orchestrator()
        self._conversation_history: list = []
        self._last_check: Optional[datetime] = None

    # ================================================================
    # 公开入口
    # ================================================================

    def run(self, user_input: str = "") -> str:
        """Agent 主入口。用户说话 → Agent 响应。"""
        observations = self._perceive(user_input)
        action = self._decide(user_input, observations)
        result = self._act(action, user_input, observations)
        self._learn(action, result, user_input)
        response = self._respond(action, result, observations)
        self._conversation_history.append({
            "time": datetime.now().isoformat(),
            "user": user_input,
            "action": action,
            "response": response[:200],
        })
        return response

    # ================================================================
    # 1. 感知 (Perceive)
    # ================================================================

    def _perceive(self, user_input: str) -> dict:
        obs = {
            "user_intent": self._classify_intent(user_input),
            "pending_work": self.orchestrator.state_machine.get_active_count(),
            "knowledge_size": self._get_knowledge_stats(),
            "time_since_last_use": self._time_since_last_use(),
            "is_autonomous": (user_input.strip() == ""),
            "apis_ok": self._check_apis(),
        }
        return obs

    def _check_apis(self) -> dict:
        """API 健康检查 — 使用 config 的分层解析器，支持 .env / keychain 等"""
        return {
            "deepseek": bool(config.deepseek_api_key),
            "qwen": bool(config.qwen_api_key),
        }

    def _classify_intent(self, text: str) -> str:
        if not text or not text.strip():
            return "autonomous"
        t = text.strip().lower()

        if t in ("/quit", "/exit", "quit", "exit", "退出"):
            return "quit"

        process_keywords = ["处理", "转成笔记", "跑一下", "跑这个", "处理视频",
                           "process", "视频", ".mp4", "转笔记"]
        stats_keywords = ["统计", "仪表盘", "学了什么", "进度", "总结", "报告", "概况", "最近"]
        chat_keywords = ["搜索", "查找", "找", "有没有", "什么是",
                        "怎么", "如何", "区别", "对比",
                        "?", "？", "问", "聊", "说说", "解释", "为什么"]

        if any(k in t for k in process_keywords):
            return "process_video"
        if any(k in t for k in stats_keywords):
            return "stats"
        if any(k in t for k in chat_keywords):
            return "chat"
        return "chat"

    def _get_knowledge_stats(self) -> dict:
        kg_path = os.path.join(config.memory_dir, "knowledge_graph.json")
        if os.path.exists(kg_path):
            with open(kg_path, encoding="utf-8") as f:
                kg = json.load(f)
            return {
                "concepts": len(kg.get("concepts", [])),
                "relations": len(kg.get("relations", [])),
            }
        return {"concepts": 0, "relations": 0}

    def _time_since_last_use(self) -> Optional[float]:
        profile_path = os.path.join(config.memory_dir, "user_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
            last = profile.get("last_updated", "")
            if last:
                try:
                    last_dt = datetime.fromisoformat(last)
                    return (datetime.now() - last_dt).total_seconds() / 3600
                except Exception:
                    pass
        return None

    # ================================================================
    # 2. 决策 (Decide)
    # ================================================================

    def _decide(self, user_input: str, obs: dict) -> dict:
        intent = obs["user_intent"]

        if intent == "autonomous":
            if obs["time_since_last_use"] and obs["time_since_last_use"] > 24:
                intent = "proactive_checkin"
            else:
                intent = "idle"

        action = {"type": intent}

        if intent in ("process_video", "chat", "search"):
            apis = obs.get("apis_ok", {})
            if not apis.get("deepseek", False):
                action["type"] = "api_warning"

        if intent == "process_video":
            extracted = self._extract_path(user_input)
            if extracted:
                action["type"] = "process_video"
                action["video_path"] = extracted
                action["remaining"] = 0
            else:
                action["type"] = "guide"
                action["message"] = "no_videos"

        elif intent == "search":
            action["query"] = user_input

        elif intent == "chat":
            action["question"] = user_input

        elif intent == "proactive_checkin":
            action["message"] = "long_time_no_see"

        return action

    @staticmethod
    def _extract_path(text: str) -> Optional[str]:
        # 1) 引号内的路径（含空格也没问题）
        for q in ('"', '""', "'"):
            if q in text:
                parts = text.split(q)
                for i, part in enumerate(parts):
                    if i % 2 == 1 and part.strip():
                        p = pathlib.Path(part.strip())
                        if p.exists():
                            return str(p.resolve())

        # 2) 无空格路径
        for m in re.finditer(r'([A-Za-z]:[\\/][^\s,;)\]}"\']+)', text):
            p = pathlib.Path(m.group(1).strip())
            if p.exists():
                return str(p.resolve())

        # 3) 相对路径
        for m in re.finditer(r'([.][.\/\\][^\s,;)\]}"\']+)', text):
            p = pathlib.Path(m.group(1).strip())
            if p.exists():
                return str(p.resolve())

        # 4) 带空格的路径：匹配盘符到已知媒体扩展名之间的全部内容
        _EXT_PAT = r'\.(mp4|mkv|mov|avi|m4a|mp3|wav|flac|aac)'
        m = re.search(r'([A-Za-z]:[\\/].*?' + _EXT_PAT + r')', text, re.IGNORECASE)
        if m:
            full = m.group(1).strip()  # 含扩展名
            p = pathlib.Path(full)
            if p.exists():
                return str(p.resolve())

        return None

    # ================================================================
    # 3. 行动 (Act)
    # ================================================================

    def _act(self, action: dict, user_input: str, obs: dict) -> dict:
        atype = action["type"]
        result = {"type": atype, "ok": True, "data": None}

        try:
            if atype == "api_warning":
                result["data"] = "api_missing"

            elif atype == "process_video":
                config.setup_proxy()
                video_path = action["video_path"]
                if os.path.isdir(video_path):
                    _EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4a",
                             ".mp3", ".wav", ".flac", ".aac"}
                    videos = []
                    for f in sorted(os.listdir(video_path)):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in _EXTS:
                            videos.append(os.path.join(video_path, f))
                    if not videos:
                        result["ok"] = False
                        result["error"] = f"目录中没有视频: {video_path}"
                    else:
                        results = []
                        for v in videos:
                            task = self.orchestrator.process_video(v)
                            results.append(task)
                        ok_tasks = [t for t in results if t and t.status.value == "completed"]
                        result["data"] = {
                            "batch": True,
                            "total": len(videos),
                            "completed": len(ok_tasks),
                        }
                else:
                    task = self.orchestrator.process_video(video_path)
                    if task and task.status.value == "completed":
                        result["data"] = {
                            "qa_score": task.qa_score,
                            "note_path": task.md_path,
                            "txt_path": task.txt_path,
                            "elapsed": f"{task.elapsed_seconds:.0f}s",
                        }
                        result["remaining"] = action.get("remaining", 0)
                    else:
                        result["ok"] = False
                        result["error"] = task.error_message if task else "unknown"

            elif atype == "search":
                config.setup_proxy()
                result["data"] = search_notes(action["query"])

            elif atype == "chat":
                config.setup_proxy()
                result["data"] = chat_notes(action["question"])

            elif atype == "stats":
                result["data"] = self._build_stats_report(obs)

            elif atype == "guide":
                result["data"] = action["message"]

            elif atype == "proactive_checkin":
                result["data"] = "checkin"

        except Exception as e:
            result["ok"] = False
            result["error"] = str(e)
            logger.error(f"Agent action failed ({atype}): {e}")

        return result

    # ================================================================
    # 4. 学习 (Learn)
    # ================================================================

    def _learn(self, action: dict, result: dict, user_input: str):
        if not result["ok"]:
            pass
        profile_path = os.path.join(config.memory_dir, "user_profile.json")
        try:
            profile = {}
            if os.path.exists(profile_path):
                with open(profile_path, encoding="utf-8") as f:
                    profile = json.load(f)
            profile["last_active"] = datetime.now().isoformat()
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ================================================================
    # 5. 响应 (Respond)
    # ================================================================

    def _respond(self, action: dict, result: dict, obs: dict) -> str:
        atype = action["type"]

        if not result["ok"]:
            return f"抱歉，出了点问题: {result.get('error', '未知错误')}"

        if atype == "api_warning":
            return (
                "API Key 未设置。在当前 PowerShell 窗口执行：\n\n"
                '  $env:DEEPSEEK_API_KEY = "sk-你的key"\n'
                '  $env:QWEN_API_KEY = "sk-你的key"\n'
                '  $env:HTTP_PROXY = "http://127.0.0.1:7890"\n'
                '  $env:HTTPS_PROXY = "http://127.0.0.1:7890"\n\n'
                "设置后重新运行 weaver"
            )

        if atype == "process_video":
            d = result["data"]
            if d.get("batch"):
                return (
                    f"批量处理完成！共 {d['total']} 个视频，"
                    f"成功 {d['completed']} 个。"
                )
            lines = [
                f"搞定！笔记已生成，QA 评分 {d['qa_score']}",
                f"笔记: {d['note_path']}",
            ]
            kg = obs.get("knowledge_size", {})
            if kg.get("concepts", 0) > 0:
                lines.append("")
                lines.append(f"📚 知识库已积累 {kg['concepts']} 个概念。")
            return "\n".join(lines)

        if atype == "search":
            return result["data"]

        if atype == "chat":
            data = result["data"]
            kg = obs.get("knowledge_size", {})
            if kg.get("concepts", 0) > 0:
                data += f"\n\n---\n> 💡 知识库已收录 {kg['concepts']} 个半导体工艺概念。"
            return data

        if atype == "stats":
            return self._build_stats_report(obs)

        if atype == "guide" and result["data"] == "no_videos":
            return (
                "请告诉我视频在哪。你可以这样：\n\n"
                "  处理 E:\\路径\\视频.mp4        # 给文件路径\n"
                "  处理 E:\\视频文件夹\\          # 给文件夹路径\n"
                "  帮我处理 E:\\视频\\xxx.mp4     # 自然语言也行\n\n"
                "路径带中文就用引号包起来，例如：\n"
                "  处理 \"E:\\视频\\1.工艺速通\""
            )

        if atype == "proactive_checkin":
            kg = obs.get("knowledge_size", {})
            return (
                f"好久不见！你已经有 {obs['time_since_last_use']:.0f} 小时没来了。\n\n"
                f"当前知识库: {kg['concepts']} 个概念, {kg['relations']} 条关联。\n\n"
                f"我可以帮你:\n"
                f"- 处理新视频（告诉我文件或文件夹路径）\n"
                f"- 搜索已有笔记\n"
                f"- 复习薄弱知识点\n"
                f"- 回答半导体工艺问题\n\n"
                f"想做点什么？"
            )

        return self._build_idle_response(obs)

    def _build_stats_report(self, obs: dict) -> str:
        kg = obs.get("knowledge_size", {})
        note_dir = config.note_dir
        notes = []
        for root, dirs, files in os.walk(note_dir):
            for f in files:
                if f.endswith(".md"):
                    notes.append(os.path.join(root, f))

        cats = {}
        for n in notes:
            cat = os.path.basename(os.path.dirname(n))
            cats[cat] = cats.get(cat, 0) + 1

        lines = [
            "=" * 40,
            "  NoteWeaver 学习仪表盘",
            "=" * 40,
            f"  笔记总数:   {len(notes)} 篇",
            f"  知识图谱:   {kg['concepts']} 概念, {kg['relations']} 关系",
        ]

        mem = self.orchestrator.memory
        try:
            mem._ensure_loaded()
            history = mem.profile.get("learning_history", [])
            if history:
                scores = [h.get("qa_score", 0) for h in history]
                lines.append(f"  Agent处理:  {len(history)} 次, 平均 QA {sum(scores)/len(scores):.1f}")
        except Exception:
            pass

        lines.append("")
        lines.append("  分类分布:")
        for cat, count in sorted(cats.items()):
            lines.append(f"    {cat}: {count} 篇")

        lines.append("")
        lines.append("=" * 40)
        return "\n".join(lines)

    def _build_idle_response(self, obs: dict) -> str:
        kg = obs.get("knowledge_size", {})
        return (
            f"我在。当前知识库 {kg['concepts']} 概念。\n"
            f"告诉我视频路径（文件或文件夹）我就开始处理。"
        )


# ============================================================
# 全局单例
# ============================================================
_agent_instance: Optional[NoteWeaverAgent] = None


def get_agent() -> NoteWeaverAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = NoteWeaverAgent()
    return _agent_instance
