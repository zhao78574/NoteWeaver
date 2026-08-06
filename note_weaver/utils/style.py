"""NoteWeaver 终端样式封装 — 基于 Rich 的语义化 UI 辅助

集中管理所有终端输出样式。
所有 .py 模块统一 from note_weaver.utils.style import * 即可使用。
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box
import os
import re
import pathlib

# ── 全局 Console 实例（自动检测终端能力） ──
console = Console()

# ================================================================
# 语义化等级输出
# ================================================================

def ok(msg: str, detail: str = ""):
    """绿色成功消息"""
    if detail:
        console.print(f"  [bold green]✓ {msg}[/bold green]  [dim]{detail}[/dim]")
    else:
        console.print(f"  [bold green]✓ {msg}[/bold green]")


def err(msg: str):
    """红色错误消息"""
    console.print(f"  [bold red]✗ {msg}[/bold red]")


def warn(msg: str):
    """橙色警告消息"""
    console.print(f"  [bold orange3]⚠ {msg}[/bold orange3]")


def info(msg: str):
    """蓝色信息消息"""
    console.print(f"  [bold blue]● {msg}[/bold blue]")


def status(msg: str):
    """中性状态消息"""
    console.print(f"  [white]{msg}[/white]")


# ================================================================
# 专用输出函数
# ================================================================

def file_path(path: str):
    """文件路径输出"""
    console.print(f"    [bold yellow]\U0001f4c4 Markdown[/bold yellow]  [cyan]{path}[/cyan]")


def graph_stats(concepts: int, relations: int):
    """知识图谱统计"""
    console.print(f"    [bold cyan]\U0001f578️ 知识图谱[/bold cyan]  [yellow]{concepts}[/yellow]概念 · [yellow]{relations}[/yellow]关系")


def qa_hint(hint: str = 'weaver ask "你的问题"'):
    """问答模式提示"""
    console.print(f"    [bold purple]\U0001f4ac 问答可用[/bold purple]  [dim]{hint}[/dim]")


def step_done(step_name: str, elapsed: str = ""):
    """步骤已完成"""
    elapsed_str = f"  [dim]{elapsed}[/dim]" if elapsed else ""
    console.print(f"  [green]✓[/green] {step_name}{elapsed_str}")


def step_running(step_name: str, detail: str = ""):
    """步骤进行中（带 spin 效果）"""
    detail_str = f"  [dim]{detail}[/dim]" if detail else ""
    console.print(f"  [yellow]◐[/yellow] {step_name}{detail_str}")


def step_pending(step_name: str):
    """步骤待执行"""
    console.print(f"  [dim]◌[/dim] {step_name} [dim]等待中...[/dim]")


def section(title: str, border_style: str = "blue"):
    """分隔面板标题"""
    console.print(f"")
    console.print(Panel(f"[bold]{title}[/bold]", border_style=border_style, box=box.ROUNDED))


def command_prompt() -> str:
    """交互模式输入提示 — Tab 补全 + 持久化历史记录"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, PathCompleter, WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style as PtStyle
        import os

        history_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "data", ".cli_history"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        _COMMANDS = [
            "重排", "处理", "graph", "统计", "仪表盘",
            "/quit", "/exit", "/stop",
        ]

        class _SmartCompleter(Completer):
            def __init__(self):
                self.word_comp = WordCompleter(_COMMANDS, ignore_case=True, sentence=True)
                self.path_comp = PathCompleter(expanduser=True)

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if any(text.startswith(cmd + " ") for cmd in ("处理", "重排")):
                    yield from self.path_comp.get_completions(document, complete_event)
                else:
                    yield from self.word_comp.get_completions(document, complete_event)

        session = PromptSession(
            history=FileHistory(history_path),
            completer=_SmartCompleter(),
            style=PtStyle.from_dict({"prompt": "bold cyan"}),
        )
        return session.prompt("❯ ")

    except ImportError:
        return Prompt.ask("[bold cyan]❯[/bold cyan]")


def print_markdown(text: str):
    """渲染 Markdown 文本"""
    console.print(Markdown(text))


def print_separator(char: str = "─", width: int = 50, style: str = "dim"):
    """打印分隔线"""
    console.print(f"[{style}]{char * width}[/{style}]")


# ================================================================
# 仪表盘面板
# ================================================================

def dashboard_panel(notes_total: int, concepts: int, relations: int,
                    history_count: int = 0, avg_score: float = 0.0,
                    categories: dict = None):
    if categories is None:
        categories = {}

    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=box.ROUNDED,
        title="[bold]\U0001f4ca 分类分布[/bold]",
        title_style="bold blue",
    )
    table.add_column("分类", style="cyan", no_wrap=True)
    table.add_column("篇数", justify="right", style="bold yellow")
    table.add_column("分布", width=25)

    max_n = max(categories.values()) if categories else 1
    for cat, count in sorted(categories.items()):
        bar_len = int(count / max_n * 20) if max_n > 0 else 0
        bar = "[green]█[/green]" * bar_len + "[dim]░[/dim]" * (20 - bar_len)
        table.add_row(cat, str(count), bar)

    panel = Panel(
        f"[bold]笔记总数:[/bold] [yellow]{notes_total}[/yellow] 篇\n"
        f"[bold]知识图谱:[/bold] [cyan]{concepts}[/cyan] 概念 · [cyan]{relations}[/cyan] 关系"
        + (f"\n[bold]Agent处理:[/bold] [yellow]{history_count}[/yellow] 次, 平均 QA [yellow]{avg_score:.1f}[/yellow]"
           if history_count > 0 else "")
        + ("\n\n" + table.rendered if categories else ""),
        title="[bold]\U0001f4ca 学习仪表盘[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(panel)


# ================================================================
# 结果面板
# ================================================================

def result_panel(title: str, content_lines: list, border_style: str = "green"):
    panel = Panel(
        "\n".join(f"  {line}" for line in content_lines),
        title=f"[bold]{title}[/bold]",
        border_style=border_style,
        box=box.ROUNDED,
    )
    console.print(panel)


# ================================================================
# Markdown 笔记完成面板
# ================================================================

def note_complete_panel(note_path: str, concepts: int = 0, relations: int = 0, qa_score=None):
    lines = []
    lines.append(f"[bold yellow]\U0001f4c4 Markdown[/bold yellow]  [cyan]{note_path}[/cyan]")
    if concepts > 0:
        lines.append(f"[bold cyan]\U0001f578️ 知识图谱[/bold cyan]  [yellow]{concepts}[/yellow]概念 · [yellow]{relations}[/yellow]关系")
        lines.append(f"[dim]          weaver graph 查看[/dim]")
    lines.append(f"[bold purple]\U0001f4ac 问答可用[/bold purple]  [dim]weaver ask \"你的问题\"[/dim]")

    title = f"✅ 笔记生成完成"
    if qa_score is not None:
        title += f"  [dim]QA评分[/dim] [yellow]{qa_score}[/yellow] [dim]/ 10[/dim]"

    panel = Panel(
        "\n".join(lines),
        title=f"[bold]{title}[/bold]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(panel)


# ================================================================
# 启动仪表盘
# ================================================================

def startup_dashboard(concepts: int = 0, relations: int = 0, notes: int = 0):
    logo = (
        "[bold cyan]  \U0001f9e0 NoteWeaver[/bold cyan]\n"
        "[dim]  AI Note Assistant[/dim]"
    )
    stat_line = ""
    parts = []
    if notes:
        parts.append(f"[yellow]{notes}[/yellow] [dim]notes[/dim]")
    if concepts:
        parts.append(f"[cyan]{concepts}[/cyan] [dim]concepts[/dim]")
    if relations:
        parts.append(f"[cyan]{relations}[/cyan] [dim]relations[/dim]")
    if parts:
        stat_line = "\n  " + "  ·  ".join(parts)

    panel = Panel(
        logo + stat_line,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)


# ================================================================
# 创建 Progress 上下文管理器（进度条）
# ================================================================

def create_progress(transient: bool = True) -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )


# ================================================================
# 流式输出显示（ChatGPT 风格 — 直接打印，无闪烁）
# ================================================================

class StreamDisplay:
    """终端流式输出 — 像 ChatGPT 一样逐 token 直接打印"""

    def __init__(self):
        self._started = False
        self._phase_count = 0
        self.did_stream = False

    def make_callbacks(self) -> dict:
        return {
            "on_phase": self.on_phase,
            "on_token": self.on_token,
            "on_error": self.on_error,
            "on_complete": self.on_complete,
        }

    def on_phase(self, phase: str, status: str, detail: str = ""):
        if status == "start":
            self._phase_count += 1
            if self._phase_count > 1 or self._started:
                console.print()
            console.print(f"  [bold cyan]⏳ {phase}...[/bold cyan]")
        elif status == "done":
            console.print(f"  [bold green]✅ {phase} 完成[/bold green]")
        elif status == "error":
            console.print(f"  [bold red]❌ {phase} 失败: {detail}[/bold red]")

    def on_token(self, token: str):
        if not self._started:
            self._started = True
            if self._phase_count == 0:
                console.print()
        self.did_stream = True
        import sys
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_error(self, error: str):
        if self._started:
            console.print()
        console.print(f"  [bold red]❌ 错误: {error}[/bold red]")

    def on_complete(self, result: dict):
        pass

    def stop(self):
        if self._started:
            console.print()


# ================================================================
# 来源笔记高亮显示（含章节名 + 内容预览 + 可点击跳转）
# ================================================================

_note_cache: dict = None
_PROJECT_NOTE_DIR = None


def _get_note_dir() -> str:
    global _PROJECT_NOTE_DIR
    if _PROJECT_NOTE_DIR is None:
        try:
            from note_weaver.utils.config import config
            _PROJECT_NOTE_DIR = config.note_dir
            return _PROJECT_NOTE_DIR
        except Exception:
            pass
        here = pathlib.Path(__file__).resolve().parent
        project_root = here.parent.parent
        candidate = project_root / "data" / "Note"
        if candidate.exists():
            _PROJECT_NOTE_DIR = str(candidate)
        else:
            _PROJECT_NOTE_DIR = ""
    return _PROJECT_NOTE_DIR


def _build_note_cache():
    global _note_cache
    _note_cache = {}
    note_dir = _get_note_dir()
    if note_dir and os.path.isdir(note_dir):
        for root, dirs, files in os.walk(note_dir):
            for f in files:
                if f.endswith(".md"):
                    full = os.path.join(root, f)
                    name = f[:-3]
                    _note_cache[name] = full
                    _note_cache[f] = full


def _resolve_note_path(note_name: str) -> str:
    if _note_cache is None:
        _build_note_cache()
    note_name = note_name.strip()
    base = os.path.basename(note_name)
    if note_name in _note_cache:
        return _note_cache[note_name]
    if base in _note_cache:
        return _note_cache[base]
    if base.endswith(".md") and base[:-3] in _note_cache:
        return _note_cache[base[:-3]]
    return ""


def _get_source_details(src_name: str) -> dict:
    """从 chat.py 的检索缓存中获取章节名和预览"""
    try:
        from note_weaver.skills.chat import get_source_details
        return get_source_details(src_name)
    except ImportError:
        return {}


def print_source_footer(text: str):
    """打印 [来源：xxx] 为高亮可点击列表（含章节名 + 内容预览 + VS Code 行跳转）"""
    source_pattern = r'\[来源：([^\]]+)\]'
    sources = re.findall(source_pattern, text)
    if not sources:
        return

    seen = set()
    unique_sources = []
    for s in sources:
        s = s.strip()
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    console.print()
    console.print("[bold yellow]== 参考笔记 ==[/bold yellow]")
    for src in unique_sources:
        fpath = _resolve_note_path(src)
        details = _get_source_details(src)
        section = details.get("section", "") if details else ""
        snippet = details.get("snippet", "") if details else ""
        line_start = details.get("line_start", 0) if details else 0

        console.print(f'  [bold yellow]* {src}[/bold yellow]')
        if section:
            console.print(f'    [cyan]章节:[/cyan] {section}')
        if snippet:
            clipped = snippet[:120].replace("\n", " ")
            console.print(f'    [dim]预览:[/dim] {clipped}')
        if fpath:
            if line_start and line_start > 0:
                vscode_uri = f"vscode://file/{fpath.replace(chr(92), chr(47))}:{line_start}"
                console.print(f'    [bold cyan][link={vscode_uri}]VS Code 第{line_start}行[/link][/bold cyan]')
            uri = f"file:///{fpath.replace(os.sep, '/').lstrip('/')}"
            console.print(f'    [link={uri}]{uri}[/link]')
        else:
            console.print(f'    [dim](文件未找到)[/dim]')


def print_response(text: str):
    """打印回答，[来源：xxx] 标注显示为彩色章节名 + 预览 + 可点击链接"""
    source_pattern = r'\[来源：([^\]]+)\]'
    sources = re.findall(source_pattern, text)

    clean_text = re.sub(source_pattern, '', text).strip()
    if clean_text:
        console.print(Markdown(clean_text))

    if sources:
        seen = set()
        unique_sources = []
        for s in sources:
            s = s.strip()
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        console.print()
        console.print("[bold yellow]== 参考笔记 ==[/bold yellow]")
        for src in unique_sources:
            fpath = _resolve_note_path(src)
            details = _get_source_details(src)
            section = details.get("section", "") if details else ""
            snippet = details.get("snippet", "") if details else ""
            line_start = details.get("line_start", 0) if details else 0

            console.print(f'  [bold yellow]* {src}[/bold yellow]')
            if section:
                console.print(f'    [cyan]章节:[/cyan] {section}')
            if snippet:
                clipped = snippet[:120].replace("\n", " ")
                console.print(f'    [dim]预览:[/dim] {clipped}')
            if fpath:
                if line_start and line_start > 0:
                    vscode_uri = f"vscode://file/{fpath.replace(chr(92), chr(47))}:{line_start}"
                    console.print(f'    [bold cyan][link={vscode_uri}]VS Code 第{line_start}行[/link][/bold cyan]')
                uri = f"file:///{fpath.replace(os.sep, '/').lstrip('/')}"
                console.print(f'    [link={uri}]{uri}[/link]  [dim](Ctrl+打开)[/dim]')
            else:
                console.print(f'    [bold yellow]* {src}[/bold yellow]')
