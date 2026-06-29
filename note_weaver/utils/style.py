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
    TaskID,
)
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box
from rich.tree import Tree
from rich.columns import Columns
from rich.layout import Layout

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
    console.print(f"    [bold yellow]📄 Markdown[/bold yellow]  [cyan]{path}[/cyan]")


def graph_stats(concepts: int, relations: int):
    """知识图谱统计"""
    console.print(f"    [bold cyan]🕸️ 知识图谱[/bold cyan]  [yellow]{concepts}[/yellow]概念 · [yellow]{relations}[/yellow]关系")


def qa_hint(hint: str = 'weaver ask "你的问题"'):
    """问答模式提示"""
    console.print(f"    [bold purple]💬 问答可用[/bold purple]  [dim]{hint}[/dim]")


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
        from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style as PtStyle
        import os

        # 持久化历史文件（在 data/ 下，已被 .gitignore 排除）
        history_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "data", ".cli_history"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        # 常用命令列表
        _COMMANDS = [
            "重排", "处理", "graph", "统计", "仪表盘",
            "/quit", "/exit", "/stop",
        ]

        class _SmartCompleter(Completer):
            """根据上下文切换命令补全 vs 路径补全"""
            def __init__(self):
                self.word_comp = WordCompleter(_COMMANDS, ignore_case=True, sentence=True)
                self.path_comp = PathCompleter(expanduser=True)

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                # 如果已经输入了"处理"或"重排"，后面跟路径
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
        # fallback: prompt_toolkit 未安装时用 Rich
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
    """生成 Rich 仪表盘 Panel + Table

    Args:
        notes_total: 笔记总数
        concepts: 知识图谱概念数
        relations: 知识图谱关系数
        history_count: 处理历史次数
        avg_score: 平均 QA 评分
        categories: {分类名: 篇数} 字典
    """
    if categories is None:
        categories = {}

    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=box.ROUNDED,
        title="[bold]📊 分类分布[/bold]",
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
        + ("\n\n" + table.rendered if categories else ""),  # type: ignore[attr-defined]
        title="[bold]📊 学习仪表盘[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(panel)


# ================================================================
# 结果面板
# ================================================================

def result_panel(title: str, content_lines: list, border_style: str = "green"):
    """通用结果面板

    Args:
        title: 面板标题
        content_lines: 内容行列表
        border_style: 边框颜色
    """
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
    """笔记生成完成后的总结面板"""
    lines = []
    lines.append(f"[bold yellow]📄 Markdown[/bold yellow]  [cyan]{note_path}[/cyan]")
    if concepts > 0:
        lines.append(f"[bold cyan]🕸️ 知识图谱[/bold cyan]  [yellow]{concepts}[/yellow]概念 · [yellow]{relations}[/yellow]关系")
        lines.append(f"[dim]          weaver graph 查看[/dim]")
    lines.append(f"[bold purple]💬 问答可用[/bold purple]  [dim]weaver ask \"你的问题\"[/dim]")

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
    """极简启动界面 — Logo + 核心指标"""
    logo = (
        "[bold cyan]  🧠 NoteWeaver[/bold cyan]\n"
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
    """创建一个 NoteWeaver 风格进度条实例"""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )
