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
from note_weaver.utils.style import dashboard_panel, console
from note_weaver.agents.orchestrator import Orchestrator
from note_weaver.skills.search import search as search_notes
from note_weaver.skills.chat import chat as chat_notes


# ── 意图分类 System Prompt ──────────────────────────────────
_INTENT_SYSTEM = """你是一个意图分类器。根据用户的输入，判断他想做什么，只返回一个单词。

分类规则：
- process_video: 用户想处理视频/音频/PDF/网页，提到"转笔记""处理""下载"等
- read_note: 用户想读/讲解/解释某篇笔记，提到"讲解""读笔记""看一下这篇""帮我看看"
- summarize: 用户想总结某篇笔记或文章
- explain: 用户想问"为什么""怎么理解""是什么意思"
- compare: 用户想对比两个东西
- search: 用户想搜索知识库中的信息
- stats: 用户想看统计/概况/进度
- list: 用户想列出所有笔记
- chat: 以上都不符合，纯聊天/问答

只返回一个单词。"""


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
        """用 LLM 理解用户意图，替代关键词匹配"""
        if not text or not text.strip():
            return "autonomous"

        t = text.strip().lower()

        # ── 特殊命令（不走 LLM） ──
        if t in ("/quit", "/exit", "quit", "exit", "退出"):
            return "quit"
        if t in ("/stop", "/pause", "stop", "pause"):
            return "stop"

        # ── 快速关键词兜底 ──
        if any(kw in t for kw in
               ("处理视频", "转成笔记", "处理 ", ".mp4", ".mkv", ".avi")):
            return "process_video"

        # ── 用 LLM 理解意图 ──
        try:
            config.setup_proxy()
            from openai import OpenAI
            client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
            )
            resp = client.chat.completions.create(
                model=config.model_fast,
                messages=[{"role": "system", "content": _INTENT_SYSTEM},
                          {"role": "user", "content": text}],
                temperature=0.1,
                max_tokens=50,
            )
            intent = resp.choices[0].message.content.strip().lower()
            # 验证合法意图
            valid = {"process_video", "chat", "search", "read_note",
                     "stats", "summarize", "explain", "compare", "list"}
            for v in valid:
                if v in intent:
                    return v
            return "chat"
        except Exception as e:
            logger.warning(f"[Agent] LLM 分类失败，走关键词兜底: {e}")
            # ── LLM 不可用时的关键词兜底 ──
            process_kw = ["处理", "转笔记", "跑一下", "视频", ".mp4", ".mkv"]
            stats_kw = ["统计", "学了什么", "进度", "报告", "概况"]
            read_kw = ["讲解", "读笔记", "看看", "这篇笔记", "这个笔记",
                       "读一下", "讲一下", "解释一下", "帮我看看"]
            if any(k in t for k in process_kw):
                return "process_video"
            if any(k in t for k in stats_kw):
                return "stats"
            if any(k in t for k in read_kw):
                return "read_note"
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
            last = profile.get("last_visit", "")
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

        elif intent == "read_note":
            action["type"] = "read_note"
            # 尝试从输入中提取笔记路径或名称
            note_path_or_name = self._extract_note_ref(user_input)
            if note_path_or_name:
                action["note_ref"] = note_path_or_name
            else:
                action["note_ref"] = user_input
            action["question"] = user_input

        elif intent in ("explain", "summarize", "compare"):
            action["type"] = "chat"
            action["question"] = user_input

        elif intent == "list":
            action["type"] = "list_notes"

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

    @staticmethod
    def _extract_note_ref(text: str) -> Optional[str]:
        """尝试从用户输入中提取笔记名称或路径"""
        import pathlib as _pl

        # 1) 显式路径
        for q in ('"', '""', "'"):
            if q in text:
                parts = text.split(q)
                for i, part in enumerate(parts):
                    if i % 2 == 1 and part.strip():
                        p = _pl.Path(part.strip())
                        if p.exists() and p.suffix == ".md":
                            return str(p.resolve())

        for m in re.finditer(r'([A-Za-z]:[\\/][^\s,;)\]}"\']+\.md)', text):
            p = _pl.Path(m.group(1).strip())
            if p.exists():
                return str(p.resolve())

        # 2) 从常用笔记目录搜索匹配的 .md 文件名
        note_dir = config.note_dir
        if note_dir and _pl.Path(note_dir).exists():
            # 去掉可能的 .md 后缀和引号
            name = text.strip().strip('"\'')
            # 提取可能的文件名关键词
            import re as _re
            # 常见中文句式："讲解一下 xxx笔记" "读一下 xxx"
            patterns = [
                r'(?:讲解|读|看|总结|解释)(?:一下)?\s*[：:]\s*(.+?)(?:[，。！？]|$)',
                r'(?:讲解|读|看|总结|解释)(?:一下)?\s*(.+?)(?:\.md)?\s*$',
                r'["\'](.+?\.md)["\']',
            ]
            candidates = []
            for pat in patterns:
                m = _re.search(pat, text)
                if m:
                    candidates.append(m.group(1).strip())

            # 也搜索所有 .md 文件进行模糊匹配
            name_keywords = name.replace(".md", "").replace("\\", "/").split("/")[-1]
            for root, dirs, files in os.walk(str(note_dir)):
                for f in files:
                    if not f.endswith(".md"):
                        continue
                    stem = f[:-3]
                    # 精确匹配
                    if name_keywords == stem or name_keywords in stem:
                        return str(_pl.Path(root) / f)
                    # 对候选词匹配
                    for c in candidates:
                        if c == stem or c in stem or stem in c:
                            return str(_pl.Path(root) / f)

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

            elif atype == "read_note":
                config.setup_proxy()
                note_ref = action.get("note_ref", "")
                question = action.get("question", "")
                result["data"] = self._read_and_explain_note(note_ref, question)

            elif atype == "list_notes":
                result["data"] = self._list_all_notes()

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
            profile["last_visit"] = datetime.now().isoformat()
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ================================================================
    # 5. 响应 (Respond)
    # ================================================================

    def _read_and_explain_note(self, note_ref: str, question: str) -> str:
        """读取并讲解笔记"""
        import pathlib as _pl

        # 1) 先尝试直接作为路径
        note_path = None
        if _pl.Path(note_ref).exists() and note_ref.endswith(".md"):
            note_path = note_ref
        elif _pl.Path(note_ref).exists():
            # 可能是目录，找目录下的 .md
            p = _pl.Path(note_ref)
            if p.is_dir():
                mds = sorted(p.glob("*.md"))
                if mds:
                    note_path = str(mds[0])
        else:
            # 2) 在笔记目录中模糊搜索
            note_dir = _pl.Path(config.note_dir)
            if note_dir.exists():
                # 去掉路径元素，取最后一段作为关键词
                keywords = note_ref.replace(".md", "").replace("\\", "/").split("/")[-1]
                for root, dirs, files in os.walk(str(note_dir)):
                    for f in files:
                        if not f.endswith(".md"):
                            continue
                        stem = f[:-3]
                        if keywords in stem or stem in keywords:
                            note_path = str(_pl.Path(root) / f)
                            break
                    if note_path:
                        break

        if not note_path or not os.path.isfile(note_path):
            # 搜索结果提示
            note_dir = config.note_dir
            matches = []
            if note_dir:
                for root, dirs, files in os.walk(note_dir):
                    for f in files:
                        if f.endswith(".md"):
                            matches.append(os.path.join(root, f))
            hint = f"找到 {len(matches)} 篇笔记。试试说「讲解 03_工艺流程02」\n\n📚 笔记示例："
            for m in matches[:10]:
                rel = os.path.relpath(m, note_dir)
                hint += f"\n  · {rel[:-3]}"
            return hint

        # 3) 读取笔记内容
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 4) 用 LLM 讲解
        from openai import OpenAI
        client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        _EXPLAIN_SYSTEM = """你是一个专业的学习助手。你会收到一篇笔记和用户的问题。

请根据笔记内容回答用户的问题。要求：
1. 用通俗易懂的语言解释
2. 画 ASCII 结构图辅助理解
3. 举例子帮助理解
4. 如果笔记中有图片占位符(![](...))，说明此处有插图的用意
5. 如果用户没提具体问题，就先概括笔记核心内容，再逐段讲解
6. 保持自然对话语气，不要AI八股"""

        resp = client.chat.completions.create(
            model=config.model_fast,
            messages=[
                {"role": "system", "content": _EXPLAIN_SYSTEM},
                {"role": "user", "content": (
                    f"## 笔记文件\n{os.path.basename(note_path)}\n\n"
                    f"## 笔记内容\n{content[:12000]}\n\n"
                    f"## 用户问题\n{question if question and '讲解' not in question else '请帮我讲解这篇笔记'}"
                )},
            ],
            temperature=0.5,
        )
        return resp.choices[0].message.content or "（无响应）"

    @staticmethod
    def _list_all_notes() -> str:
        """列出所有笔记"""
        note_dir = config.note_dir
        if not note_dir or not os.path.isdir(note_dir):
            return "笔记库为空"

        categories = {}
        for root, dirs, files in os.walk(note_dir):
            cat = os.path.basename(root)
            if cat == os.path.basename(note_dir):
                continue
            mds = sorted([f[:-3] for f in files if f.endswith(".md")])
            if mds:
                categories[cat] = mds

        if not categories:
            return "笔记库为空，请先处理一些视频或 PDF。"

        total = sum(len(v) for v in categories.values())
        lines = [f"📚 **笔记库** — {total} 篇，{len(categories)} 个分类\n"]
        for cat in sorted(categories):
            files = categories[cat]
            lines.append(f"\n### {cat} ({len(files)}篇)")
            for f in files:
                lines.append(f"  · {f}")

        return "\n".join(lines)

    @staticmethod
    def _build_stats_report(obs: dict) -> str:
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

        history_count = 0
        avg_score = 0.0
        mem = None
        try:
            from note_weaver.agents.memory_agent import MemoryAgent
            mem = MemoryAgent()
            mem._ensure_loaded()
            history = mem.profile.get("learning_history", [])
            if history:
                history_count = len(history)
                scores = [h.get("qa_score", 0) for h in history]
                avg_score = sum(scores) / len(scores)
        except Exception:
            pass

        # Rich 仪表盘 Panel + Table
        dashboard_panel(
            notes_total=len(notes),
            concepts=kg.get("concepts", 0),
            relations=kg.get("relations", 0),
            history_count=history_count,
            avg_score=avg_score,
            categories=cats,
        )

        # 返回空字符串 — dashboard_panel 已直接打印
        return ""

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
            note_path = d.get("note_path", "")
            note_dir = os.path.dirname(note_path) if note_path else ""
            kg = obs.get("knowledge_size", {})

            lines = [
                f"[OK] 笔记生成完成！QA 评分 {d.get('qa_score', '?')}",
                "",
                f"  [FILE] Markdown:  {note_path}",
            ]
            # 图谱输出 — 判断是否已有或可生成
            graph_path = os.path.join(note_dir, "_knowledge_graph.html") if note_dir else ""
            if kg.get("concepts", 0) > 0:
                lines.append(f"  [GRAPH]  知识图谱:  {kg['concepts']} 概念, {kg['relations']} 条关系")
                lines.append(f"                  weaver graph 查看")
            # 问答模式
            lines.append(f"  [QA] 问答模式:  weaver ask \"你的问题\"")
            return "\n".join(lines)

        if atype == "search":
            return result["data"]

        if atype == "chat":
            data = result["data"]
            kg = obs.get("knowledge_size", {})
            # 追加相关概念推荐
            related = self.orchestrator.memory.search_concepts(
                action.get("question", ""), top_k=5
            ) if hasattr(self.orchestrator, 'memory') else []
            if related:
                names = [f"`{c.get('name', c.get('name_en', '?'))}`" for c in related[:5]]
                data += f"\n\n---\n💡 相关概念: {' · '.join(names)}"
            return data

        if atype == "read_note":
            return result.get("data", "未找到笔记内容")

        if atype == "list_notes":
            return result.get("data", "笔记库为空")

        if atype == "stats":
            return result.get("data", "")

        if atype == "guide" and result["data"] == "no_videos":
            return (
                "请告诉我视频在哪。你可以这样：\n\n"
                "  处理 D:\\videos\\demo.mp4         # 给文件路径\n"
                "  处理 D:\\videos\\                 # 给文件夹路径\n"
                "  帮我处理 D:\\videos\\xxx.mp4      # 自然语言也行\n\n"
                "路径带中文就用引号包起来，例如：\n"
                "  处理 \"D:\\videos\\教程视频.mp4\""
            )

        if atype == "proactive_checkin":
            kg = obs.get("knowledge_size", {})
            return (
                f"欢迎回来！\n\n"
                f"当前知识库 **{kg['concepts']}** 概念 · "
                f"**{kg['relations']}** 关系。\n\n"
                f"可以直接问我工艺问题，或拖视频/PDF 给我处理。"
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

        history_count = 0
        avg_score = 0.0
        mem = self.orchestrator.memory
        try:
            mem._ensure_loaded()
            history = mem.profile.get("learning_history", [])
            if history:
                history_count = len(history)
                scores = [h.get("qa_score", 0) for h in history]
                avg_score = sum(scores) / len(scores)
        except Exception:
            pass

        # Rich 仪表盘 Panel + Table
        dashboard_panel(
            notes_total=len(notes),
            concepts=kg.get("concepts", 0),
            relations=kg.get("relations", 0),
            history_count=history_count,
            avg_score=avg_score,
            categories=cats,
        )

        # 返回空字符串 — dashboard_panel 已直接打印
        return ""

    def _build_idle_response(self, obs: dict) -> str:
        kg = obs.get("knowledge_size", {})
        return (
            f"知识库 **{kg['concepts']}** 概念 · "
            f"**{kg['relations']}** 关系。"
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
