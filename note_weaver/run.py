#!/usr/bin/env python3
"""NoteWeaver Agent — 自主 AI 笔记数字助理

安装: pip install -e /path/to/NoteWeaver
用法:
    weaver                               # Agent 交互模式
    weaver "PIE和TD PIE有什么区别？"      # 单次问答
    weaver ./lecture.mp4                 # 直接处理视频（自动识别路径）
    weaver ./videos/                     # 直接处理整个文件夹
    weaver --batch ./videos/             # 批量处理
"""

import os, sys, argparse, json, pathlib, re
from datetime import datetime
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from note_weaver.utils.style import (
    console, ok, err, warn, info, status, file_path, graph_stats, qa_hint,
    step_done, step_running, step_pending, section, command_prompt,
    print_markdown, print_separator, dashboard_panel, result_panel,
    note_complete_panel, create_progress,
)

__version__ = "1.0.0"

# 作为 pip 包导入时无需手动加 sys.path
if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.agent import NoteWeaverAgent
from note_weaver.core.job import Job, Modality, PipelineType


def cmd_agent(user_input: str = "", template_name: str = None):
    agent = NoteWeaverAgent()
    if template_name:
        agent.orchestrator.set_template(template_name)
    response = agent.run(user_input)
    print_markdown(response)


def cmd_process(video_path: str, template_name: str = None):
    agent = NoteWeaverAgent()
    if template_name:
        agent.orchestrator.set_template(template_name)
    status(f"开始处理: [cyan]{video_path}[/cyan]")
    response = agent.run(f"处理视频: {video_path}")
    print_markdown(response)
    _regenerate_graph()
    _rebuild_embedding_index()


def _extract_page_from_filename(image_id: str) -> int:
    """从 image_id 中解析页码（e.g. paper_p3_0_hash.jpg → 3）"""
    import re as _re
    m = _re.search(r'_p(\d+)_', image_id)
    return int(m.group(1)) if m else 999




def _extract_hash_from_filename(image_id: str) -> str:
    """从 image_id 中解析内容hash（e.g. paper_p3_0_f770bc.jpg → f770bc）"""
    import re as _re
    m = _re.search(r'_([a-f0-9]{8,16})\.[a-z]+$', image_id)
    return m.group(1) if m else ""


def _extract_keywords(text: str) -> set:
    """提取文本中的中文词+英文词，过滤单字和常见停用词"""
    import re as _re
    STOP_WORDS = {"this", "that", "the", "and", "for", "with", "from", "figure", "fig",
                  "image", "show", "shows", "shown", "using", "based", "number"}
    words = set(
        w for w in _re.sub(r'[^一-鿿\w]', ' ', text).split()
        if len(w) > 1 and w.lower() not in STOP_WORDS
    )
    return words


def insert_images_by_content(note_text, vision_results, file_base):
    import re as _re
    from note_weaver.agents.composer import _replace_placeholders, _fix_broken_markdown_images
    images = [r for r in vision_results if r.get("should_include", True)]
    if not images:
        return note_text

    # ── 第1步：替换 [图片: path] 占位符 ──────────────────────
    note_text, placeholder_count = _replace_placeholders(note_text, file_base, images)
    if placeholder_count:
        info("Images 占位符替换: {} 张".format(placeholder_count))

    # ── 第2步：修复 Composer 生成的断裂/嵌套 ![]() ──────────
    repaired = _fix_broken_markdown_images(note_text)
    if repaired != note_text:
        note_text = repaired
        info("Images 修复断裂 markdown 图片引用")

    # ── 第3步：检查是否需要关键字匹配补插 ────────────────────
    # 获取所有已存在的图片引用路径
    existing_refs = _re.findall(r'!\[\]\(([^)]+)\)', note_text)
    if existing_refs:
        info("Images 已有 {} 张图引用，不再重复匹配".format(len(existing_refs)))
        return note_text

    # 按页码排序，让图片按文档顺序插入
    images.sort(key=lambda r: _extract_page_from_filename(r.get("image_id", "")))

    # ── 清理可能残留的旧图片引用 ────────────────────────────
    cleaned = _re.sub(r'\n?!\[\]\([^)]+\)\n?\*[^*]*\*\n?', '', note_text)
    cleaned = _re.sub(r'\n?!\[\]\([^)]+\)\n?\*\*[^*]*\*\*[^\n]*\n?', '', cleaned)
    cleaned = _re.sub(r'\n?!\[\]\([^)]+\)\n?', '', cleaned)
    cleaned = _re.sub(r'\n?#\s*[📎📌]\s*图注.*?\n', '\n', cleaned)
    note_text = cleaned.strip()

    # ── 按 ## 标题切分章节 ──────────────────────────────────
    sections = _re.split(r'(?=^## )', note_text, flags=_re.MULTILINE)
    sections = [s for s in sections if s.strip()]
    if not sections:
        sections = [note_text]

    # ── 跨页去重：相同 hash 的图片只插一次 ──────────────────
    seen_hashes = set()
    deduped = []
    for img in images:
        h = _extract_hash_from_filename(img.get("image_id", ""))
        if h and h in seen_hashes:
            continue
        if h:
            seen_hashes.add(h)
        deduped.append(img)
    images = deduped

    # ── 逐图匹配 ────────────────────────────────────────────
    section_image_count = {}
    matched_count = 0
    for img in images:
        img_id = img.get("image_id", "unknown.jpg")
        desc = img.get("content_description", "")
        key_terms = img.get("key_terms", [])
        caption = img.get("suggested_caption", "") or desc[:60]
        img_ref = "\n![]({}/{})\n*{}*\n".format(file_base, img_id, caption)

        # 可追溯性：HTML trace 注释标记图片来源
        trace_source = "{}/{}".format(file_base, img_id)
        if img.get("timestamp"):
            trace_source += " @ {}".format(img["timestamp"])
        trace_comment = "\n<!-- frame: {} -->\n".format(trace_source)
        img_ref = trace_comment + img_ref

        keyword_source = desc + ' ' + ' '.join(key_terms) + ' ' + img_id
        img_words = _extract_keywords(keyword_source)

        best_idx, best_score = -1, 0
        for i, sec in enumerate(sections):
            sec_words = _extract_keywords(sec[:300].lower())
            score = len(img_words & sec_words)
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx >= 0 and best_score > 0:
            matched_count += 1
            sec = sections[best_idx]
            insert_pos = len(sec)
            for tag in ('\n</font>', '\n---\n## '):
                pos = sec.rfind(tag)
                if pos > 0:
                    insert_pos = pos
                    break
            sections[best_idx] = sec[:insert_pos] + img_ref + sec[insert_pos:]
            section_image_count[best_idx] = section_image_count.get(best_idx, 0) + 1
        else:
            insert_idx = min(
                range(len(sections)),
                key=lambda i: (section_image_count.get(i, 0), i)
            )
            sec = sections[insert_idx]
            insert_pos = len(sec)
            for tag in ('\n</font>', '\n---\n## '):
                pos = sec.rfind(tag)
                if pos > 0:
                    insert_pos = pos
                    break
            sections[insert_idx] = sec[:insert_pos] + img_ref + sec[insert_pos:]
            section_image_count[insert_idx] = section_image_count.get(insert_idx, 0) + 1

    total = sum(section_image_count.values())
    distributed = total - matched_count
    info("Images 关键匹配插图: {} 张 (关键词 {} / 均衡分布 {})".format(
        total, matched_count, distributed))
    return "\n".join(sections)


def count_inserted_images(original, modified):
    import re as _re
    return len(_re.findall(r'!\[\]\(', modified)) - len(_re.findall(r'!\[\]\(', original))


def cmd_regenerate_note(note_path, instruction=""):
    import re as _re
    config.setup_proxy()
    p = pathlib.Path(note_path)
    file_base = p.stem.replace(' ', '_')
    category = p.parent.name
    base_dir = pathlib.Path(config.base_dir)
    txt_path = base_dir / "data" / "TXT" / category / f"{file_base}.txt"
    note_img_dir = p.parent / file_base

    if not txt_path.exists():
        err(f"转录不存在: {txt_path}")
        return

    json_path = txt_path.with_suffix(".transcript.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            td = json.load(f)
        timestamped_text = td.get("timestamped", "")
        segments = td.get("segments", [])  # V2: 读取转录段落用于时间戳对齐
        ok("转录 JSON: {} 字, {} 段落".format(len(timestamped_text), len(segments)))
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        sents = [s.strip() for s in _re.split(r'([。！？\n])', raw) if s.strip()]
        timestamped_text = ""
        ft = 0
        i = 0
        while i < len(sents):
            chunk = sents[i]
            if i + 1 < len(sents) and sents[i+1] in ('。', '！', '？'):
                chunk += sents[i+1]
                i += 1
            timestamped_text += "[{:02d}:{:02d}] {}\n".format(ft//60, ft%60, chunk)
            ft += 15
            i += 1
        segments = []  # 无 JSON 时无法获取段落
        ok("转录 TXT: {} 字（无段落信息）".format(len(timestamped_text)))

    screenshot_files = sorted([str(f) for f in note_img_dir.glob("*.jpg")]) if note_img_dir.exists() else []
    ok("截图: {} 张".format(len(screenshot_files)))

    from note_weaver.agents.vision import VisionAgent
    vision_results = []
    if screenshot_files:
        info("Vision 分析截图...")
        vision_results = VisionAgent().execute(screenshot_files)

    from note_weaver.agents.composer import ComposerAgent
    composer = ComposerAgent()
    revision_feedback = ""
    hints = []
    if "详细" in instruction:
        hints.append("每个知识点至少展开 4-5 句")
    if "通俗" in instruction:
        hints.append("用最通俗的大白话解释")
    if "专业" in instruction:
        hints.append("深入技术细节")
    if hints:
        revision_feedback = "；".join(hints)

    info("Composer 生成笔记...")
    note_content = composer.execute(
        file_base=file_base,
        timestamped_text=timestamped_text,
        vision_results=vision_results,
        strategy={"note_style": "detailed", "focus_areas": []},
        revision_feedback=revision_feedback,
        segments=segments,  # V2: 传入段落用于时间戳对齐
    )

    note_with_images = insert_images_by_content(note_content, vision_results, file_base)
    from note_weaver.agents.composer import _fix_broken_markdown_images
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    ok("图片: {} 张插入/保留".format(inserted))
    note_path = composer.save_note(file_base, note_with_images, str(p.parent))
    console.print()
    ok("重排完成！")
    file_path(note_path)


def cmd_batch_regenerate(category_dir, instruction=""):
    import pathlib as _pl
    note_dir = _pl.Path(category_dir)
    if not note_dir.is_dir():
        err("目录不存在: {}".format(category_dir))
        return
    md_files = sorted(note_dir.glob("*.md"))
    if not md_files:
        info("没有 .md 文件")
        return
    status("批量重排: {} ({} 篇)".format(note_dir.name, len(md_files)))
    for i, f in enumerate(md_files, 1):
        txt = _pl.Path(config.txt_dir) / note_dir.name / "{}.txt".format(f.stem)
        if not txt.exists():
            warn("跳过 {} (缺转录)".format(f.stem))
            continue
        console.print()
        status("[{}/{}] {}".format(i, len(md_files), f.stem))
        cmd_regenerate_note(str(f), instruction)


def _search_and_regenerate(search, full_input):
    import pathlib as _pl
    note_dir = _pl.Path(config.note_dir)
    if not note_dir.exists():
        err("笔记目录不存在")
        return
    instruction = ""
    for kw in ["写详细点", "通俗易懂", "深入专业", "加案例", "加表格"]:
        if kw in full_input:
            instruction = kw
            break
    for cat_dir in sorted(note_dir.iterdir()):
        if cat_dir.is_dir() and cat_dir.name == search:
            cmd_batch_regenerate(str(cat_dir), instruction)
            return
    matches = []
    for cat_dir in sorted(note_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            if search.lower() in f.stem.lower() or search.lower() in cat_dir.name.lower():
                txt = _pl.Path(config.txt_dir) / cat_dir.name / "{}.txt".format(f.stem)
                matches.append((f, txt.exists()))
    if not matches:
        err("未找到匹配的笔记: {}".format(search))
    elif len(matches) == 1:
        f, has_txt = matches[0]
        if not has_txt:
            err("找到笔记但缺少转录: {}".format(f.name))
            return
        info("重排: {} / {}".format(f.parent.name, f.stem))
        cmd_regenerate_note(str(f), instruction)
    else:
        info("找到 {} 个匹配：".format(len(matches)))
        for i, (f, has_txt) in enumerate(matches, 1):
            tag = "[bold green]可重排[/bold green]" if has_txt else "[dim]缺转录[/dim]"
            console.print("  [{}] {} / {}  {}".format(i, f.parent.name, f.stem, tag))


def _scan_videos(directory: str) -> list:
    """扫描目录下的视频文件"""
    VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.m4a', '.mp3', '.wav', '.aac'}
    videos = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(os.path.join(root, f))
    logger.info(f"[Batch] 扫描到 {len(videos)} 个视频文件")
    return videos


def cmd_batch(directory):
    if not os.path.isdir(directory):
        logger.error("目录不存在:", directory)
        return
    videos = _scan_videos(directory)
    if not videos:
        info("没有视频文件")
        return
    for i, vp in enumerate(videos, 1):
        status("[{}/{}]".format(i, len(videos)))
        cmd_process(vp)


def cmd_config_init(args):
    """weaver --init → 交互式生成 config.yaml"""
    config_source = pathlib.Path(__file__).resolve().parent / "config.yaml"
    if config_source.exists():
        import yaml
        from note_weaver.utils.style import Prompt as _Prompt
        ans = _Prompt.ask("config.yaml 已存在，覆盖？", choices=["y", "N"], default="N")
        if ans != "y":
            info("已取消。")
            return
        section("NoteWeaver 配置初始化")
        deepseek_key = _Prompt.ask("DeepSeek API Key", default="")
        qwen_key = _Prompt.ask("Qwen VL API Key", default="")
        whisper_size = _Prompt.ask("Whisper 模型大小", choices=["small", "base", "large"], default="small")
        proxy_host = _Prompt.ask("代理地址 (回车跳过)", default="")
        proxy_port = _Prompt.ask("代理端口 (回车跳过)", default="")
        cfg = {
            "api": {"deepseek": {"api_key": deepseek_key},
                    "qwen": {"api_key": qwen_key}},
            "whisper": {"model_size": whisper_size},
            "proxy": {"enabled": bool(proxy_host), "host": proxy_host or "127.0.0.1",
                      "port": int(proxy_port) if proxy_port else 7890},
        }
        with open(config_source, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        ok("配置文件已生成: {}".format(config_source))
    else:
        err("默认配置文件不存在: {}".format(config_source))


def cmd_graph():
    import subprocess as _sp
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weaver_graph.py"
    memory_db = pathlib.Path(config.memory_dir)
    kg_path = memory_db / "knowledge_graph.json"

    if not kg_path.exists():
        warn("知识图谱尚未生成 — 先处理视频生成笔记，图谱会自动构建")
        return

    # 确保输出目录存在
    memory_db.mkdir(parents=True, exist_ok=True)

    # 先生成 HTML
    _regenerate_graph()

    if not script.exists():
        err("图谱脚本不存在: {}".format(script))
        return

    # 打开浏览器
    html_path = memory_db / "knowledge_graph.html"
    if html_path.exists():
        import webbrowser
        webbrowser.open(str(html_path.resolve()))
        ok("知识图谱已打开: [cyan]{}[/cyan]".format(html_path))
    else:
        err("图谱 HTML 未生成，尝试直接运行脚本...")
        try:
            _sp.run([sys.executable, str(script)], check=True)
        except _sp.CalledProcessError as _e:
            err("图谱脚本运行失败 (exit {})".format(_e.returncode))


def _regenerate_graph():
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weaver_graph.py"
    memory_db = pathlib.Path(config.memory_dir)
    memory_db.mkdir(parents=True, exist_ok=True)
    output_path = memory_db / "knowledge_graph.html"
    if script.exists():
        import subprocess as _sp
        try:
            _sp.run([sys.executable, str(script), "--output", str(output_path)],
                     capture_output=True, timeout=30)
        except _sp.TimeoutExpired:
            warn("图谱生成超时")
        except Exception as _e:
            warn("图谱生成异常: {}".format(_e))


def _rebuild_embedding_index():
    try:
        from note_weaver.utils.embeddings import EmbeddingIndex
        count = EmbeddingIndex().build(force=True)
        if count:
            logger.info("[Embedding] 索引重建完成: {} 条".format(count))
    except Exception as e:
        logger.warning("[Embedding] 索引重建失败（非致命）: {}".format(e))


def cmd_from_url(url):
    config.setup_proxy()
    from note_weaver.agents.downloader import VideoDownloader
    info("解析链接: {}".format(url))

    dl = VideoDownloader()

    # ── 先检测是否为合集（B站 /sp?spm_id=...episodes 等隐式合集） ──
    playlist_info = dl.detect_playlist(url)
    if playlist_info and playlist_info["playlist_count"] > 1:
        console.print()
        ok(f"检测到合集: 「{playlist_info['title']}」({playlist_info['playlist_count']} 集)")
        from_hint = console.input(f"  是否进入合集模式？(y/是/N) ").strip().lower()
        if from_hint in ("y", "yes", "是", "对", "ok"):
            cmd_merge_playlist(url)
            return
        info("按单视频处理...")
        console.print()

    # ── 单视频下载 ──
    try:
        meta = dl.get_info(url)
        ok("{} ({}分{}秒)".format(meta.get("title","?"), meta.get("duration",0)//60, meta.get("duration",0)%60))
    except Exception as e:
        err("获取元数据失败: {}".format(e))
        return
    info("下载视频...")
    try:
        result = dl.download(url)
    except Exception as e:
        err("下载失败: {}".format(e))
        return
    ok("下载完成: [cyan]{}[/cyan]".format(result["local_path"]))
    cmd_process(result["local_path"])


def cmd_from_pdf(pdf_path):
    config.setup_proxy()
    from note_weaver.utils.extractors import extract_from_pdf
    from note_weaver.agents.vision import VisionAgent
    from note_weaver.agents.composer import ComposerAgent
    p = pathlib.Path(pdf_path)
    file_base = p.stem.replace(' ', '_')
    base_dir = pathlib.Path(config.base_dir)
    note_category = "pdf"
    img_dir = base_dir / "data" / "Note" / note_category / file_base
    note_dir = base_dir / "data" / "Note" / note_category

    info("PDF 提取: [cyan]{}[/cyan]".format(p.name))
    result = extract_from_pdf(pdf_path, output_dir=str(img_dir))
    step_done("文本: {} 字 | 图片: {} 张".format(len(result["text"]), len(result["images"])))

    vision_results = []
    if result["images"]:
        info("PDF 提取图片 {} 张 → Vision 分析...".format(len(result["images"])))
        vision_results = VisionAgent().execute(result["images"])
        included = sum(1 for r in vision_results if r.get("should_include", True))
        ok("Vision: {} 张采纳 / {} 张过滤".format(included, len(vision_results) - included))
    else:
        info("PDF 无图片")

    info("Composer 生成笔记...")
    composer = ComposerAgent()
    note_content = composer.execute(
        file_base=file_base,
        timestamped_text=result["text"][:15000],
        vision_results=vision_results,
        strategy={"note_style": "detailed", "focus_areas": []},
        revision_feedback="这是从PDF提取的内容，整理成结构化学习笔记。",
    )
    note_with_images = insert_images_by_content(note_content, vision_results, file_base)
    # ── 最终后处理：修复 Composer 可能生成的断裂 ![]() markdown ──
    from note_weaver.agents.composer import _fix_broken_markdown_images
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    ok("笔记图片: 共 {} 张".format(inserted))
    note_path = composer.save_note(file_base, note_with_images, str(note_dir))
    _regenerate_graph()
    _rebuild_embedding_index()
    console.print()
    note_complete_panel(note_path)


def cmd_from_web(url):
    config.setup_proxy()
    from note_weaver.utils.extractors import extract_from_url
    from note_weaver.agents.vision import VisionAgent
    from note_weaver.agents.composer import ComposerAgent
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    file_base = re.sub(r'[^\w一-鿿]+', '_', domain)[:30]
    base_dir = pathlib.Path(config.base_dir)
    note_category = "web"
    img_dir = base_dir / "data" / "Note" / note_category / file_base
    note_dir = base_dir / "data" / "Note" / note_category

    info("网页提取: [cyan]{}[/cyan]".format(url))
    result = extract_from_url(url, output_dir=str(img_dir))
    ok("标题: {}".format(result["title"][:60]))
    step_done("文本: {} 字 | 图片: {} 张".format(len(result["text"]), len(result["images"])))

    vision_results = []
    if result["images"]:
        info("Vision 分析图片...")
        vision_results = VisionAgent().execute(result["images"])

    info("Composer 生成笔记...")
    composer = ComposerAgent()
    note_content = composer.execute(
        file_base=file_base,
        timestamped_text=result["text"][:15000],
        vision_results=vision_results,
        strategy={"note_style": "detailed", "focus_areas": []},
        revision_feedback="来源网页: {}\n整理成结构化学习笔记。".format(url),
    )
    note_with_images = insert_images_by_content(note_content, vision_results, file_base)
    from note_weaver.agents.composer import _fix_broken_markdown_images
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    ok("图片: {} 张插入/保留".format(inserted))
    note_path = composer.save_note(file_base, note_with_images, str(note_dir))
    _regenerate_graph()
    _rebuild_embedding_index()
    console.print()
    note_complete_panel(note_path)


# ════════════════════════════════════════════════════════════════
# 合集合并笔记
# ════════════════════════════════════════════════════════════════

def cmd_merge_playlist(url: str):
    """处理合集/播放列表链接 — 交互式选择合并范围

    流程:
      1. detect_playlist → 显示合集信息+视频列表
      2. 询问用户"哪些视频要合并到一起"
      3. 选中的视频逐条转写→笔记→合并为一篇
      4. 未选中的（如有）忽略不处理

    Args:
        url: B站合集 / YouTube播放列表链接
    """
    config.setup_proxy()
    from note_weaver.agents.downloader import VideoDownloader

    info(f"检测合集: [cyan]{url}[/cyan]")
    dl = VideoDownloader()
    playlist_info = dl.detect_playlist(url)

    if not playlist_info:
        err("未能识别为合集，或不支持的平台")
        return

    total = playlist_info["playlist_count"]
    title = playlist_info["title"]
    videos = playlist_info["videos"]

    # ── 显示合集信息 ──
    console.print()
    ok(f"📁 合集: 「[bold]{title}[/bold]」({total} 集)")
    total_dur = sum(v.get("duration", 0) for v in videos)
    step_done(f"总时长: {total_dur//60} 分 {total_dur%60} 秒")
    console.print()

    # 显示视频列表（缩略）
    _display_video_list(videos, max_show=20)
    console.print()

    # ── 询问合并范围 ──
    hint = (
        "  范围? [dim]1-10[/dim] 一组 / [dim]1-10;11-18;19-25;26-36[/dim] 多批\n"
        "  [dim]示例: all 全部 / 1,3,5-8 / 11-18;19-25;26-36 跳过前10集 / 回车[/dim]=跳过"
    )
    selection = console.input(hint + "\n  ❯ ").strip()

    if not selection:
        info("已跳过合并，不处理")
        return

    # ── 解析用户选择（支持多组） ──
    groups = _parse_selection_groups(selection, total)
    if not groups:
        err("无效的选择，请使用例如: 1-10 / 1-10;11-18;19-25 / all")
        return

    # ── 多组预览 ──
    if len(groups) > 1:
        console.print()
        ok(f"共 {len(groups)} 批，每批独立出一篇笔记:")
        for i, g in enumerate(groups, 1):
            total_dur_g = sum(
                v.get("duration", 0)
                for v in _filter_videos_by_indices(videos, g)
            )
            console.print(
                f"  [bold]第{i}批:[/bold] {_summarize_range(g)} "
                f"({len(g)}集 / {total_dur_g//60}分{total_dur_g%60}秒)"
            )
        console.print()

    # ── 确认 ──
    total_selected = sum(len(g) for g in groups)
    confirm_text = f"  是否开始处理这 {total_selected} 个视频（{len(groups)} 批）？[Y/n] "
    confirm = console.input(confirm_text).strip().lower()
    if confirm in ("n", "no"):
        info("已取消")
        return

    # ── 逐批处理 ──
    from note_weaver.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    all_results = []

    for i, group_indices in enumerate(groups, 1):
        group_videos = _filter_videos_by_indices(videos, group_indices)
        # 组标题加范围后缀，方便区分
        range_str = _summarize_range(group_indices)
        group_title = f"{title}（{range_str}）"

        console.print()
        status(f"[第{i}批/{len(groups)}] {group_title} ({len(group_videos)}集)")

        result = orchestrator._run_merge_from_videos(
            group_title, group_videos,
            original_url=url, selected_indices=group_indices,
        )

        if result.get("ok"):
            ok(f"第{i}批完成: 合并 {result.get('merged_count', 0)} 个视频")
            for path in result.get("output_paths", []):
                file_path(path)
        else:
            err(f"第{i}批失败: {result.get('error', '未知错误')}")

        all_results.append(result)

    # ── 汇总 ──
    succeeded = [r for r in all_results if r.get("ok")]
    failed = [r for r in all_results if not r.get("ok")]
    console.print()
    if len(groups) == 1 and succeeded:
        merged = succeeded[0].get("merged_count", 0)
        total = succeeded[0].get("total_videos", 0)
        if merged > 0:
            ok("合集处理完成!")
        else:
            err(f"❌ 合并失败：{merged}/{total} 个视频处理成功，请检查日志或网络连接")
    elif succeeded:
        ok(f"合集处理完成: {len(succeeded)}/{len(groups)} 批成功")
    if failed:
        err(f"{len(failed)} 批失败")

    if succeeded and any(r.get("merged_count", 0) > 0 for r in succeeded):
        _regenerate_graph()
        _rebuild_embedding_index()


def _display_video_list(videos: list, max_show: int = 20):
    """显示合集中的视频列表"""
    for i, v in enumerate(videos[:max_show], 1):
        title_text = v.get("title", f"第{i}集")
        if len(title_text) > 50:
            title_text = title_text[:50] + "…"
        console.print(f"  [{i:03d}] {title_text}")

    remaining = len(videos) - max_show
    if remaining > 0:
        console.print(f"  [dim]… 还有 {remaining} 个视频未显示[/dim]")


def _parse_selection(text: str, total: int) -> list:
    """解析用户输入的合并范围

    支持格式:
      - "all" / "全部" → 全部
      - "1-5" → 1,2,3,4,5
      - "1,3,5-8" → 1,3,5,6,7,8
      - "1-5,7,9-12" → 混合

    Args:
        text: 用户输入
        total: 视频总数

    Returns:
        排序后的视频 index 列表（1-based），空列表=无效输入
    """
    text = text.strip().lower()
    if text in ("all", "全部", "全选", "a"):
        return list(range(1, total + 1))

    indices = set()
    parts = [p.strip() for p in text.split(",")]
    for part in parts:
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                s, e = int(start.strip()), int(end.strip())
                if 1 <= s <= total and 1 <= e <= total and s <= e:
                    indices.update(range(s, e + 1))
            except ValueError:
                continue
        else:
            try:
                n = int(part)
                if 1 <= n <= total:
                    indices.add(n)
            except ValueError:
                continue

    return sorted(indices)


def _parse_selection_groups(text: str, total: int) -> list:
    """解析用户输入的合并范围，支持分号分隔的「多组」

    相比 _parse_selection 的增强:
      - "1-10;11-18;19-25;26-36" → 四组合并，每组出一篇笔记
      - 单组时退化为 _parse_selection 行为

    支持格式（每组内语法与 _parse_selection 一致）:
      - "all" / "全部" / "全选" / "a"
      - "1-10" → 一组: 1-10
      - "1-10;11-18;19-25;26-36" → 四组
      - "1,3,5-8;11-18" → 两组

    Args:
        text: 用户输入
        total: 视频总数

    Returns:
        每组一个 index 列表（组内 1-based，排序）
        空列表 = 无效输入
    """
    text = text.strip().lower()
    if text in ("all", "全部", "全选", "a"):
        return [list(range(1, total + 1))]

    # 按分号分隔多组
    group_texts = [g.strip() for g in text.split(";")]
    groups = []

    for group_text in group_texts:
        if not group_text:
            continue
        indices = set()
        parts = [p.strip() for p in group_text.split(",")]
        for part in parts:
            if not part:
                continue
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    s, e = int(start.strip()), int(end.strip())
                    if 1 <= s <= total and 1 <= e <= total and s <= e:
                        indices.update(range(s, e + 1))
                except ValueError:
                    continue
            else:
                try:
                    n = int(part)
                    if 1 <= n <= total:
                        indices.add(n)
                except ValueError:
                    continue

        if indices:
            groups.append(sorted(indices))

    return groups


def _filter_videos_by_indices(videos: list, indices: list) -> list:
    """按 1-based index 从视频列表中筛选"""
    index_set = set(indices)
    return [v for i, v in enumerate(videos, 1) if i in index_set]


def _summarize_range(indices: list) -> str:
    """将 index 列表转成简洁的范围字符串，如 [1,2,3,4,5] → "1-5"
    用于多批笔记的文件名和标题。
    """
    if not indices:
        return ""
    indices = sorted(set(indices))
    if len(indices) == 1:
        return str(indices[0])
    if indices[-1] - indices[0] + 1 == len(indices):
        return f"{indices[0]}-{indices[-1]}"
    # 不连续 → 取首尾
    return f"{indices[0]}-{indices[-1]}"


def _summarize_durations(videos: list) -> str:
    """生成合集中视频时长的分布摘要"""
    if not videos:
        return "无数据"
    total = sum(v.get("duration", 0) for v in videos)
    total_m = total // 60
    short = sum(1 for v in videos if v.get("duration", 0) <= 300)
    long_ = sum(1 for v in videos if v.get("duration", 0) > 300)
    return (
        f"共 {len(videos)} 集, "
        f"总时长 {total_m} 分钟"
        f"（≤5min: {short} 集 / >5min: {long_} 集）"
    )


# ════════════════════════════════════════════════════════════════
# 删除笔记
# ════════════════════════════════════════════════════════════════

def _find_notes(query: str) -> list:
    """按名称搜索笔记，返回 [(路径, 分类, 文件名不含扩展名), ...]"""
    note_dir = config.note_dir
    if not os.path.isdir(note_dir):
        return []
    query_lower = query.lower().replace(" ", "_")
    results = []
    for root, dirs, files in os.walk(note_dir):
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            stem = f[:-3]  # 去掉 .md
            category = os.path.basename(root)
            if query_lower in stem.lower().replace(" ", "_"):
                results.append((os.path.join(root, f), category, stem))
    return results


def _delete_note_files(note_path: str, mode: int) -> dict:
    """级联删除笔记相关文件

    Args:
        note_path: 笔记 .md 文件路径
        mode: 1=仅笔记, 2=+截图+TXT+缓存, 3=全部(含视频+KG引用)

    Returns:
        {"deleted": [路径...], "skipped": [原因...], "message": str}
    """
    deleted = []
    skipped = []
    note_dir = os.path.dirname(note_path)
    note_stem = os.path.basename(note_path)[:-3]  # 去掉 .md
    category = os.path.basename(note_dir)
    base_dir = config.base_dir

    # ── 1) 删除 .md ──
    if os.path.isfile(note_path):
        os.remove(note_path)
        deleted.append(f"📄 {note_path}")

    # ── 2) 删除 _images 目录 ──
    images_dir = os.path.join(note_dir, f"{note_stem}_images")
    # 也可能是同名的单文件夹
    alt_images_dir = os.path.join(note_dir, note_stem)
    if mode >= 2:
        for d in (images_dir, alt_images_dir):
            if os.path.isdir(d) and d != note_dir:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                deleted.append(f"🖼️  {d}/")
                break  # 只删一个

    # ── 3) 删除 TXT / transcript JSON ──
    if mode >= 2:
        txt_dir = os.path.join(base_dir, config.txt_dir if not os.path.isabs(config.txt_dir) else config.txt_dir)
        # 处理 txt_dir 可能是相对路径
        txt_base = config.txt_dir if os.path.isabs(config.txt_dir) else os.path.join(base_dir, config.txt_dir)
        txt_cat_dir = os.path.join(txt_base, category)
        for ext in (".txt", "_transcript.json"):
            p = os.path.join(txt_cat_dir, f"{note_stem}{ext}")
            if os.path.isfile(p):
                os.remove(p)
                deleted.append(f"📄 {p}")

    # ── 4) 清理缓存 ──
    if mode >= 2:
        cache_dir = os.path.join(base_dir, "data", "cache")
        if os.path.isdir(cache_dir):
            cleared = 0
            for entry_dir in os.listdir(cache_dir):
                entry_path = os.path.join(cache_dir, entry_dir)
                if not os.path.isdir(entry_path):
                    continue
                # 检查 meta.json 中的 input_name
                for stage_dir in os.listdir(entry_path):
                    meta_path = os.path.join(entry_path, stage_dir, "meta.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if meta.get("input_name", "").startswith(note_stem):
                                import shutil
                                shutil.rmtree(os.path.join(entry_path, stage_dir), ignore_errors=True)
                                cleared += 1
                        except Exception:
                            pass
            if cleared:
                deleted.append(f"💾 缓存: {cleared} 条")

    # ── 5) 查找并删除视频文件 ──
    video_paths = []
    if mode >= 3:
        video_dir = os.path.join(base_dir, "Video")
        if os.path.isdir(video_dir):
            for source_dir in os.listdir(video_dir):
                src_path = os.path.join(video_dir, source_dir)
                if not os.path.isdir(src_path):
                    continue
                for f in os.listdir(src_path):
                    if f.endswith((".mp4", ".mkv", ".webm")):
                        # 视频文件名可能包含原视频标题，检查 stem 是否包含 note_stem
                        f_stem = os.path.splitext(f)[0]
                        if note_stem in f_stem or f_stem in note_stem:
                            fp = os.path.join(src_path, f)
                            video_paths.append(fp)

        for vp in video_paths:
            os.remove(vp)
            deleted.append(f"🎬 {vp}")
            # 同时删除对应的 .meta.json
            meta_file = vp.rsplit(".", 1)[0] + ".meta.json"
            if os.path.isfile(meta_file):
                os.remove(meta_file)
                deleted.append(f"📋 {meta_file}")

    return {"deleted": deleted, "skipped": skipped}


def _has_read_intent(text: str) -> bool:
    """判断用户是否想读笔记/讲解笔记（而不是重排）"""
    low = text.lower()
    # 先排除重排/排版 意图
    if any(kw in low for kw in ["重排", "排版", "regenerate"]):
        return False
    read_kw = ["讲解", "总结", "读一下", "读这篇", "解释", "看看",
               "讲一下", "这笔记", "这个笔记", "这篇笔记",
               "说什么的", "讲的什么", "内容是什么"]
    return any(kw in low for kw in read_kw)


def _agent_read_note(text: str):
    """用 Agent 讲解笔记（不走管线）"""
    import re as _re
    # 先从输入中提取 .md 路径
    m = _re.search(r'([A-Za-z]:[\\/][^\s,;)\]}"\'"]+\.md)', text)
    if m:
        md_path = m.group(1).strip()
        if os.path.isfile(md_path):
            from note_weaver.agent import NoteWeaverAgent
            agent = NoteWeaverAgent()
            response = agent._read_and_explain_note(md_path, text)
            from note_weaver.utils.style import print_markdown
            print_markdown(response)
            return
    # 兜底
    from note_weaver.agent import NoteWeaverAgent
    agent = NoteWeaverAgent()
    response = agent.run(text)
    from note_weaver.utils.style import print_markdown
    print_markdown(response)


def _clean_knowledge_graph(source_note: str):
    """从知识图谱移除指定笔记来源的概念和关系"""
    kg_path = os.path.join(config.memory_dir, "knowledge_graph.json")
    if not os.path.isfile(kg_path):
        return

    try:
        with open(kg_path, "r", encoding="utf-8") as f:
            kg = json.load(f)
    except Exception:
        return

    before_concepts = len(kg.get("concepts", []))
    before_relations = len(kg.get("relations", []))

    # 移除仅引用此笔记的概念
    kept_concepts = []
    removed_names = set()
    for c in kg.get("concepts", []):
        sources = c.get("source_notes", [])
        # 还有别的来源则保留，否则删
        remaining = [s for s in sources if source_note not in s]
        if remaining:
            c["source_notes"] = remaining
            kept_concepts.append(c)
        else:
            removed_names.add(c["name"])

    # 移除与被删概念相关的边
    kept_relations = []
    for r in kg.get("relations", []):
        if r.get("from") not in removed_names and r.get("to") not in removed_names:
            # 也移除 source 标记为此笔记的关系
            if r.get("source", "") != source_note:
                kept_relations.append(r)

    kg["concepts"] = kept_concepts
    kg["relations"] = kept_relations

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    removed_c = before_concepts - len(kept_concepts)
    removed_r = before_relations - len(kept_relations)
    if removed_c or removed_r:
        info(f"知识图谱: 移除 {removed_c} 概念, {removed_r} 关系")


def _clean_user_profile(note_name: str):
    """从用户画像学习历史移除指定笔记"""
    profile_path = os.path.join(config.memory_dir, "user_profile.json")
    if not os.path.isfile(profile_path):
        return
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        return

    history = profile.get("learning_history", [])
    before = len(history)
    profile["learning_history"] = [
        h for h in history if note_name not in h.get("note", "")
    ]
    after = len(profile["learning_history"])
    if after < before:
        profile["last_updated"] = datetime.now().isoformat()
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        info(f"用户画像: 移除 {before - after} 条学习记录")


def cmd_delete(query: str):
    """删除笔记 — 交互式搜索 + 级联清理

    Args:
        query: 笔记名称关键字 或 .md 文件路径
    """
    config.setup_proxy()

    # ── 支持直接传 .md 文件路径 ──
    if query.endswith(".md") and os.path.isfile(query):
        note_path = os.path.abspath(query)
        category = os.path.basename(os.path.dirname(note_path))
        stem = os.path.basename(note_path)[:-3]
        _show_delete_prompt(note_path, category, stem)
        return

    # ── 搜索笔记 ──
    results = _find_notes(query)
    if not results:
        # 尝试更宽松：去掉下划线再搜
        alt_query = query.replace("_", " ").replace("-", " ")
        results = _find_notes(alt_query)
    if not results:
        err(f"未找到包含「{query}」的笔记")
        info(f"笔记目录: {config.note_dir}")
        return

    # ── 选择笔记 ──
    selected = None
    if len(results) == 1:
        selected = results[0]
    else:
        console.print()
        ok(f"找到 {len(results)} 篇匹配的笔记:")
        for i, (path, cat, stem) in enumerate(results, 1):
            rel = os.path.relpath(path, config.note_dir)
            console.print(f"  [{i}] [cyan]{rel}[/cyan]")
        console.print()
        choice = console.input(f"  选择要删除的笔记 (1-{len(results)}) 或 [dim]回车[/dim]=取消: ").strip()
        if not choice:
            info("已取消")
            return
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(results):
                raise ValueError
            selected = results[idx]
        except (ValueError, IndexError):
            err("无效的选择")
            return

    note_path, category, stem = selected
    _show_delete_prompt(note_path, category, stem)


def _show_delete_prompt(note_path: str, category: str, stem: str):
    """交互式删除确认和执行的共享逻辑"""
    rel_path = os.path.relpath(note_path, config.note_dir)

    # ── 显示信息 ──
    console.print()
    warn(f"即将删除笔记: [bold]{rel_path}[/bold]")

    # 估计关联文件
    base_dir = config.base_dir
    txt_base = config.txt_dir if os.path.isabs(config.txt_dir) else os.path.join(base_dir, config.txt_dir)
    note_parts = [
        ("📄 笔记", note_path, os.path.isfile(note_path)),
        ("🖼️  截图目录", os.path.join(os.path.dirname(note_path), f"{stem}_images"),
         os.path.isdir(os.path.join(os.path.dirname(note_path), f"{stem}_images"))),
        ("📄 转录文本", os.path.join(txt_base, category, f"{stem}.txt"),
         os.path.isfile(os.path.join(txt_base, category, f"{stem}.txt"))),
        ("📄 转录JSON", os.path.join(txt_base, category, f"{stem}_transcript.json"),
         os.path.isfile(os.path.join(txt_base, category, f"{stem}_transcript.json"))),
    ]

    # 找视频
    video_dir = os.path.join(base_dir, "Video")
    video_files = []
    if os.path.isdir(video_dir):
        for src_dir in os.listdir(video_dir):
            src_path = os.path.join(video_dir, src_dir)
            if not os.path.isdir(src_path):
                continue
            for f in os.listdir(src_path):
                if f.endswith((".mp4", ".mkv", ".webm")):
                    f_stem = os.path.splitext(f)[0]
                    if stem in f_stem or f_stem in stem:
                        video_files.append(os.path.join(src_path, f))

    console.print()
    for label, path, exists in note_parts:
        status_icon = "[green]●[/green]" if exists else "[dim]○[/dim]"
        display = os.path.relpath(path, base_dir) if os.path.exists(path) else path
        console.print(f"  {status_icon} {label}: {display if exists else '[dim]不存在[/dim]'}")

    if video_files:
        for vf in video_files:
            size = os.path.getsize(vf) // (1024 * 1024)
            vrel = os.path.relpath(vf, base_dir)
            console.print(f"  [green]●[/green] 🎬 视频: {vrel} ({size}MB)")
    else:
        console.print(f"  [dim]○[/dim] 🎬 视频: [dim]未找到匹配[/dim]")

    # 知识图谱引用
    kg_path = os.path.join(config.memory_dir, "knowledge_graph.json")
    kg_refs = 0
    if os.path.isfile(kg_path):
        try:
            with open(kg_path, "r", encoding="utf-8") as f:
                kg = json.load(f)
            for c in kg.get("concepts", []):
                if any(stem in s for s in c.get("source_notes", [])):
                    kg_refs += 1
        except Exception:
            pass
    if kg_refs:
        console.print(f"  [green]●[/green] 🧠 知识图谱: {kg_refs} 个概念引用此笔记")
    else:
        console.print(f"  [dim]○[/dim] 🧠 知识图谱: [dim]无引用[/dim]")

    # ── 选择删除模式 ──
    console.print()
    console.print("  [bold]删除选项:[/bold]")
    console.print("    [1] 仅删笔记文件")
    console.print("    [2] 笔记+截图+TXT+缓存 [bold green](推荐)[/bold green]")
    console.print("    [3] 全部清理（含视频+知识图谱引用+重建索引）")
    console.print("    [n/N] 取消")

    choice = console.input("\n  请选择 [1/2/3/n], 回车默认 [2]: ").strip().lower()

    if choice in ("n", "no", "取消"):
        info("已取消")
        return

    mode_map = {"": 2, "1": 1, "2": 2, "3": 3}
    mode = mode_map.get(choice, 2)

    # ── 确认 ──
    mode_names = {1: "仅笔记", 2: "推荐清理", 3: "全部清理"}
    confirm = console.input(
        f"  确认执行「{mode_names[mode]}」? 输入 [bold red]yes[/bold red] 确认: "
    ).strip().lower()
    if confirm != "yes":
        info("已取消")
        return

    # ── 执行 ──
    console.print()
    status("正在删除...")

    result = _delete_note_files(note_path, mode)
    for p in result.get("deleted", []):
        console.print(f"  [dim]✕[/dim] {p}")

    if mode >= 3:
        _clean_knowledge_graph(stem)
        _clean_user_profile(stem)

    # ── 重建索引 ──
    if mode >= 2:
        info("重建嵌入索引...")
        _rebuild_embedding_index()
        _regenerate_graph()

    ok("删除完成!")
    console.print()

def cmd_list():
    """列出所有笔记 — 按分类分组显示"""
    note_dir = config.note_dir
    if not os.path.isdir(note_dir):
        info("笔记目录不存在")
        return

    # 按分类收集
    categories = {}
    for root, dirs, files in os.walk(note_dir):
        cat = os.path.basename(root)
        if cat == os.path.basename(note_dir):
            continue  # 跳过根目录
        md_files = sorted([f for f in files if f.endswith(".md")])
        if md_files:
            categories[cat] = md_files

    if not categories:
        info("笔记库为空")
        return

    total = sum(len(v) for v in categories.values())
    console.print()
    ok(f"📚 笔记库: [bold]{total}[/bold] 篇, [bold]{len(categories)}[/bold] 个分类")
    console.print()

    for cat in sorted(categories):
        files = categories[cat]
        console.print(f"  [bold]{cat}[/bold] [dim]({len(files)}篇)[/dim]")
        for f in files:
            note_stem = f[:-3]
            # 检查是否有图片目录
            has_img = os.path.isdir(os.path.join(note_dir, cat, f"{note_stem}_images"))
            img_mark = " 🖼️" if has_img else ""
            console.print(f"    · {note_stem}{img_mark}")


# ════════════════════════════════════════════════════════════════
# 统一命令分发
# ════════════════════════════════════════════════════════════════

def cmd_unified(user_input: str):
    """统一入口 — Job.from_input → 自动分发到对应 cmd_*

    替代原来 200 行的 if-elif 链。
    将所有用户输入统一抽象为 Job，然后按 job.pipeline 分发。
    """
    # ── 特殊命令（不进入 Job 路由） ──
    text = user_input.strip()
    low = text.lower()

    # delete/删除 命令 — 兼容 "删除 xxx", "删除xxx", "delete xxx" 等格式
    is_delete = any(low.startswith(kw) for kw in ("delete ", "删除 ", "删除"))
    if is_delete:
        name = text
        if low.startswith("delete "):
            name = text[len("delete "):]
        elif low.startswith("删除 "):
            name = text[len("删除 "):]
        elif low.startswith("删除"):
            name = text[len("删除"):]
        name = name.strip().strip('"').strip("'").strip()
        if name:
            _run_with_cancel("删除笔记", cmd_delete, name)
            return

    if low in ("/list", "list", "ls", "笔记列表", "所有笔记"):
        _run_with_cancel("列出笔记", cmd_list)
        return

    # ── 自然语言中的 URL 提前拦截（无论什么句式，有视频 URL 就转视频处理） ──
    import re as _re_url
    _url_match = _re_url.search(
        r'https?://[^\s,;)\]}"\''
        r'一-鿿　-〿＀-￯'
        r'，。、；：？！）】」』》【】]+',
        text,
    )
    if _url_match:
        raw_url = _url_match.group(0).rstrip(".,;")
        # 确认是视频平台链接
        _low_url = raw_url.lower()
        if any(d in _low_url for d in ("youtube.com", "youtu.be", "bilibili.com", "b23.tv")):
            job = Job.from_input(raw_url)
            _dispatch_job(job)
            return

    # ── .md 文件 + 讲解/总结/读 → 走 Agent 讲解（不重排） ──
    _md_in_text = bool(_re_url.search(r'[A-Za-z]:[\\/][^\s,;)\]}"\'"]+\.md', text))
    if _md_in_text and _has_read_intent(text):
        _run_with_cancel("讲解笔记", _agent_read_note, text)
        return

    job = Job.from_input(user_input)
    _dispatch_job(job)


def _dispatch_job(job: Job):
    """按 Job 类型分发到对应的 cmd_* 函数"""
    desc_map = {
        PipelineType.FULL_NOTE: "视频处理",
        PipelineType.PDF_NOTE: "PDF处理",
        PipelineType.WEB_NOTE: "网页处理",
        PipelineType.REGENERATE: "重排笔记",
        PipelineType.BATCH_VIDEO: "批量处理",
        PipelineType.BATCH_REGENERATE: "批量重排",
        PipelineType.MERGE_NOTES: "合集处理",
    }

    # ── 特殊命令（不进入管线） ──
    cmd = job.metadata.get("command", "")
    if cmd == "graph":
        _run_with_cancel("知识图谱", cmd_graph)
        return
    if cmd == "stats":
        agent = NoteWeaverAgent()
        print_markdown(agent.run("学习统计"))
        return

    # ── 问答 ──
    if job.pipeline == PipelineType.QA_ONLY:
        agent = NoteWeaverAgent()
        console.print()
        print_markdown(agent.run(job.input))
        return

    # ── 管线任务 ──
    pipeline = job.pipeline
    label = desc_map.get(pipeline, str(pipeline.value))

    if pipeline == PipelineType.FULL_NOTE:
        # 注意：先检查 source=video_url（否则 modality=VIDEO 率先拦截 URL）
        if job.metadata.get("source") == "video_url":
            _run_with_cancel(label, cmd_from_url, job.input)
        elif job.modality in (Modality.VIDEO, Modality.AUDIO):
            _run_with_cancel(label, cmd_process, job.input)
        return

    if pipeline == PipelineType.PDF_NOTE:
        _run_with_cancel(label, cmd_from_pdf, job.input)
        return

    if pipeline == PipelineType.WEB_NOTE:
        _run_with_cancel(label, cmd_from_web, job.input)
        return

    if pipeline == PipelineType.REGENERATE:
        if job.metadata.get("search_mode"):
            _run_with_cancel("搜索重排", _search_and_regenerate, job.input,
                             job.metadata.get("raw_input", job.input))
        else:
            _run_with_cancel(label, cmd_regenerate_note, job.input)
        return

    if pipeline == PipelineType.BATCH_VIDEO:
        if job.metadata.get("source") == "playlist_url":
            # 合集链接 → 交互式选择哪些视频合并
            _run_with_cancel("合集检测", cmd_merge_playlist, job.input)
        else:
            _run_with_cancel(label, cmd_batch, job.input)
        return

    if pipeline == PipelineType.BATCH_REGENERATE:
        _run_with_cancel(label, cmd_batch_regenerate, job.input)
        return

    if pipeline == PipelineType.MERGE_NOTES:
        _run_with_cancel(label, cmd_merge_playlist, job.input)
        return


# ════════════════════════════════════════════════════════════════
# 交互模式 & CLI
# ════════════════════════════════════════════════════════════════

def _run_with_cancel(description: str, fn, *args, **kwargs):
    """执行一个可能长时间运行的操作，支持 Ctrl+C 取消"""
    import time as _time
    console.print(f"  [bold blue]●[/bold blue] {description}...  [dim](Ctrl+C 取消)[/dim]")
    try:
        fn(*args, **kwargs)
    except KeyboardInterrupt:
        warn("操作已取消")
        console.print()


def cmd_interactive():
    from note_weaver.utils.style import startup_dashboard
    agent = NoteWeaverAgent()

    # 获取知识库统计
    kg = agent.orchestrator.memory.get_learning_stats() if hasattr(agent.orchestrator, 'memory') else {}

    # 统计笔记数
    note_dir = config.note_dir
    note_count = 0
    if os.path.isdir(note_dir):
        for root, dirs, files in os.walk(note_dir):
            note_count += sum(1 for f in files if f.endswith(".md"))

    # 极简启动 Dashboard
    console.print()
    startup_dashboard(
        concepts=kg.get('kg_concepts', 0),
        relations=kg.get('kg_relations', 0),
        notes=note_count,
    )

    # 首次进入的简短问候 — 不再刷 Panel，直接显示对话
    initial = agent.run("")
    if initial.strip():
        console.print()
        print_markdown(initial)

    console.print()

    while True:
        try:
            user_input = command_prompt()
        except (EOFError, KeyboardInterrupt):
            warn("按 Ctrl+C 取消当前操作，输入 /quit 退出")
            continue
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            info("再见！")
            break
        if user_input.lower() in ("/stop", "/pause", "stop", "pause"):
            warn("没有正在运行的任务")
            continue
        if user_input.lower().startswith("/search "):
            user_input = user_input[8:]

        # ── 统一分发（替代 200 行 if-elif） ──
        try:
            cmd_unified(user_input)
        except KeyboardInterrupt:
            warn("操作已取消")

        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="NoteWeaver Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  weaver                                    # 交互模式
  weaver "PIE和TD PIE有什么区别？"          # 问答
  weaver lecture.mp4                        # 处理视频
  weaver paper.pdf                          # 处理 PDF
  weaver https://example.com                # 处理网页
  weaver --graph                            # 知识图谱
        """,
    )
    parser.add_argument("message", nargs="?", default="",
                        help="自然语言 或 文件/URL 路径")
    parser.add_argument("--video", help="处理单个视频")
    parser.add_argument("-t", "--template", default=None,
                        help="模板: semiconductor, academic, meeting, tutorial, general")
    parser.add_argument("--batch", help="批量处理目录")
    parser.add_argument("--graph", action="store_true", help="知识图谱可视化")
    parser.add_argument("--config", "-c", help="指定配置文件路径")
    parser.add_argument("--init", action="store_true", help="初始化配置")
    parser.add_argument("--version", action="store_true", help="显示版本号")

    args = parser.parse_args()

    if args.version:
        console.print("[bold purple]NoteWeaver[/bold purple] [dim]v{}[/dim]".format(__version__))
        return

    if args.config:
        from note_weaver.utils.config import Config
        Config.load(args.config)

    if args.init:
        cmd_config_init(args)
        return

    for d in [config.txt_dir, config.note_dir, config.memory_dir, config.log_dir]:
        if not d or not d.strip():
            logger.warning(f"路径为空，跳过: {d!r}")
        else:
            os.makedirs(d, exist_ok=True)

    if args.video:
        cmd_process(args.video)
    elif args.batch:
        cmd_batch(args.batch)
    elif args.graph:
        cmd_graph()
    elif args.message:
        cmd_unified(args.message.strip())
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
