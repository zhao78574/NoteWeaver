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
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

__version__ = "1.0.0"

# 作为 pip 包导入时无需手动加 sys.path
if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.agent import NoteWeaverAgent


# ── 支持的文件格式 ──────────────────────────────────────────────
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4a", ".mp3", ".wav", ".flac", ".aac"}
_NOTE_EXTS = {".md"}  # 笔记文件，用于重排
_PDF_EXTS = {".pdf"}  # PDF 文档


def _is_media_file(text: str) -> bool:
    """判断输入是否为媒体文件路径"""
    p = pathlib.Path(text.strip("\"' "))
    return p.suffix.lower() in _VIDEO_EXTS and p.exists()


def _is_media_dir(text: str) -> bool:
    """判断输入是否为包含媒体文件的目录"""
    p = pathlib.Path(text.strip("\"' "))
    return p.is_dir() and any(
        f.suffix.lower() in _VIDEO_EXTS for f in p.iterdir()
    )


def _extract_path_from_text(text: str):
    """从自然语言中提取文件/文件夹路径

    返回: (path, is_file) 或 None
    """
    # 1. 尝试取引号内的完整路径
    for q in ('"', '"', "'"):
        if q in text:
            parts = text.split(q)
            for i, part in enumerate(parts):
                if i % 2 == 1 and part.strip():
                    p = pathlib.Path(part.strip())
                    if p.exists():
                        return str(p.resolve()), p.is_file()

    # 2. 尝试匹配盘符路径如 E:\xxx 或 E:/xxx
    for m in re.finditer(r'([A-Za-z]:[\\/][^\s,;)\]}"\']+)', text):
        p = pathlib.Path(m.group(1).strip())
        if p.exists():
            return str(p.resolve()), p.is_file()

    # 3. 尝试匹配相对路径 ./ 或 ../
    for m in re.finditer(r'([.][.\/\\][^\s,;)\]}"\']+)', text):
        p = pathlib.Path(m.group(1).strip())
        if p.exists():
            return str(p.resolve()), p.is_file()

    return None


def _resolve_media_path(text: str) -> str:
    """将可能含引号的路径清理后返回绝对路径"""
    p = pathlib.Path(text.strip("\"' "))
    return str(p.resolve())


def _scan_videos(directory: str) -> list:
    """扫描目录中所有支持的媒体文件，返回绝对路径列表"""
    supported = list(_VIDEO_EXTS)
    videos = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in [s.lower() for s in supported]:
            videos.append(os.path.join(directory, f))
    return videos


# ── 命令处理 ────────────────────────────────────────────────────

def _is_pdf_file(text: str) -> bool:
    p = pathlib.Path(text.strip("\"' "))
    return p.suffix.lower() in _PDF_EXTS and p.exists()


def _is_note_file(text: str) -> bool:
    p = pathlib.Path(text.strip("\"' "))
    return p.suffix.lower() in _NOTE_EXTS and p.exists()


def _is_note_dir(text: str) -> bool:
    p = pathlib.Path(text.strip("\"' "))
    return p.is_dir() and any(f.suffix.lower() in _NOTE_EXTS for f in p.iterdir())


def _is_url(text: str) -> bool:
    t = text.strip().lower()
    return (("youtube.com" in t or "youtu.be" in t or "bilibili.com" in t or "b23.tv" in t)
            and t.startswith("http"))


def _is_web_url(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("http") and not _is_url(text)


def _extract_url_from_text(text: str) -> str:
    import re
    m = re.search(r'https?://[^\s,;)\]}"\']+', text)
    if m:
        return m.group(0).rstrip(".,;")
    return ""


def cmd_agent(user_input: str = ""):
    agent = NoteWeaverAgent()
    response = agent.run(user_input)
    print(response)


def cmd_process(video_path: str):
    agent = NoteWeaverAgent()
    print(f"开始处理: {video_path}")
    response = agent.run(f"处理视频: {video_path}")
    print(response)
    _regenerate_graph()
    _rebuild_embedding_index()


def _extract_page_from_filename(image_id: str) -> int:
    """从 image_id 中解析页码（e.g. paper_p3_0_hash.jpg → 3）"""
    import re as _re
    m = _re.search(r'_p(\d+)_', image_id)
    return int(m.group(1)) if m else 999


# 从 composer 模块导入共用的图片占位符处理和断裂修复函数
from note_weaver.agents.composer import _replace_placeholders, _fix_broken_markdown_images


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
    images = [r for r in vision_results if r.get("should_include", True)]
    if not images:
        return note_text

    # ── 第1步：替换 [图片: path] 占位符 ──────────────────────
    note_text, placeholder_count = _replace_placeholders(note_text, file_base, images)
    if placeholder_count:
        print("[Images] 占位符替换: {} 张".format(placeholder_count))

    # ── 第2步：修复 Composer 生成的断裂/嵌套 ![]() ──────────
    repaired = _fix_broken_markdown_images(note_text)
    if repaired != note_text:
        note_text = repaired
        print("[Images] 修复断裂 markdown 图片引用")

    # ── 第3步：检查是否需要关键字匹配补插 ────────────────────
    # 获取所有已存在的图片引用路径
    existing_refs = _re.findall(r'!\[\]\(([^)]+)\)', note_text)
    if existing_refs:
        print("[Images] 已有 {} 张图引用，不再重复匹配".format(len(existing_refs)))
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
    print("[Images] 关键匹配插图: {} 张 (关键词 {} / 均衡分布 {})".format(
        total, matched_count, distributed))
    return "\n".join(sections)


def count_inserted_images(original, modified):
    import re as _re
    return len(_re.findall(r'!\[\]\(', modified)) - len(_re.findall(r'!\[\]\(', original))


def cmd_regenerate_note(note_path, instruction=""):
    import re as _re
    config.setup_proxy()
    p = pathlib.Path(note_path)
    file_base = p.stem
    category = p.parent.name
    base_dir = pathlib.Path(config.base_dir)
    txt_path = base_dir / "data" / "TXT" / category / f"{file_base}.txt"
    note_img_dir = p.parent / file_base

    if not txt_path.exists():
        print("[X] 转录不存在:", txt_path)
        return

    json_path = txt_path.with_suffix(".transcript.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            td = json.load(f)
        timestamped_text = td.get("timestamped", "")
        print("[OK] 转录 JSON:", len(timestamped_text), "字")
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
        print("[OK] 转录 TXT:", len(timestamped_text), "字")

    screenshot_files = sorted([str(f) for f in note_img_dir.glob("*.jpg")]) if note_img_dir.exists() else []
    print("[OK] 截图:", len(screenshot_files), "张")

    from note_weaver.agents.vision import VisionAgent
    vision_results = []
    if screenshot_files:
        print("Vision 分析截图...")
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

    print("Composer 生成笔记...")
    note_content = composer.execute(
        file_base=file_base,
        timestamped_text=timestamped_text,
        vision_results=vision_results,
        strategy={"note_style": "detailed", "focus_areas": []},
        revision_feedback=revision_feedback,
    )

    note_with_images = insert_images_by_content(note_content, vision_results, file_base)
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    print("[OK] 图片:", inserted, "张插入/保留")
    note_path = composer.save_note(file_base, note_with_images, str(p.parent))
    print("\n[OK] 已保存:", note_path, "({} 字符)".format(len(note_with_images)))


def cmd_batch_regenerate(category_dir, instruction=""):
    import pathlib as _pl
    note_dir = _pl.Path(category_dir)
    if not note_dir.is_dir():
        print("[X] 目录不存在:", category_dir)
        return
    md_files = sorted(note_dir.glob("*.md"))
    if not md_files:
        print("没有 .md 文件")
        return
    print("批量重排:", note_dir.name, "({} 篇)".format(len(md_files)))
    for i, f in enumerate(md_files, 1):
        txt = _pl.Path(config.txt_dir) / note_dir.name / "{}.txt".format(f.stem)
        if not txt.exists():
            print("  [{}/{}] 跳过 {} (缺转录)".format(i, len(md_files), f.stem))
            continue
        print("\n  [{}/{}] {}".format(i, len(md_files), f.stem))
        cmd_regenerate_note(str(f), instruction)


def _search_and_regenerate(search, full_input):
    import pathlib as _pl
    note_dir = _pl.Path(config.note_dir)
    if not note_dir.exists():
        print("笔记目录不存在")
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
        print("未找到匹配的笔记:", search)
    elif len(matches) == 1:
        f, has_txt = matches[0]
        if not has_txt:
            print("找到笔记但缺少转录:", f.name)
            return
        print("重排:", f.parent.name, "/", f.stem)
        cmd_regenerate_note(str(f), instruction)
    else:
        print("找到 {} 个匹配：".format(len(matches)))
        for i, (f, has_txt) in enumerate(matches, 1):
            status = " [可重排]" if has_txt else " [缺转录]"
            print("  [{}] {}/{}".format(i, f.parent.name, f.stem) + status)


def cmd_batch(directory):
    if not os.path.isdir(directory):
        logger.error("目录不存在:", directory)
        return
    videos = _scan_videos(directory)
    if not videos:
        print("没有视频文件")
        return
    for i, vp in enumerate(videos, 1):
        print("[{}/{}]".format(i, len(videos)))
        cmd_process(vp)


def cmd_graph():
    _regenerate_graph()
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weaver_graph.py"
    if script.exists():
        os.system('"{}" "{}"'.format(sys.executable, script))


def _regenerate_graph():
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weaver_graph.py"
    if script.exists():
        import subprocess
        try:
            subprocess.run([sys.executable, str(script), "--output",
                str(pathlib.Path(__file__).resolve().parent.parent / "data" / "memory_db" / "knowledge_graph.html")],
                capture_output=True, timeout=30)
        except Exception:
            pass


def _rebuild_embedding_index():
    try:
        from note_weaver.utils.embeddings import EmbeddingIndex
        count = EmbeddingIndex().build(force=True)
        if count:
            logger.info("[Embedding] 索引重建完成: {} 条".format(count))
    except Exception:
        pass


def cmd_from_url(url):
    config.setup_proxy()
    from note_weaver.agents.downloader import VideoDownloader
    print("解析链接:", url)
    dl = VideoDownloader()
    try:
        info = dl.get_info(url)
        print("{} ({}分{}秒)".format(info.get("title","?"), info.get("duration",0)//60, info.get("duration",0)%60))
    except Exception as e:
        print("获取元数据失败:", e)
        return
    print("下载视频...")
    try:
        result = dl.download(url)
    except Exception as e:
        print("下载失败:", e)
        return
    print("下载完成:", result["local_path"])
    cmd_process(result["local_path"])


def cmd_from_pdf(pdf_path):
    config.setup_proxy()
    from note_weaver.utils.extractors import extract_from_pdf
    from note_weaver.agents.vision import VisionAgent
    from note_weaver.agents.composer import ComposerAgent
    p = pathlib.Path(pdf_path)
    file_base = p.stem
    base_dir = pathlib.Path(config.base_dir)
    note_category = "pdf"
    img_dir = base_dir / "data" / "Note" / note_category / file_base
    note_dir = base_dir / "data" / "Note" / note_category

    print("PDF 提取:", p.name)
    result = extract_from_pdf(pdf_path, output_dir=str(img_dir))
    print("  文本: {} 字 | 图片: {} 张".format(len(result["text"]), len(result["images"])))

    vision_results = []
    if result["images"]:
        print("[PDF] 提取图片 {} 张 → Vision 分析...".format(len(result["images"])))
        vision_results = VisionAgent().execute(result["images"])
        included = sum(1 for r in vision_results if r.get("should_include", True))
        print("[Vision] {} 张采纳 / {} 张过滤".format(included, len(vision_results) - included))
    else:
        print("[PDF] 无图片")

    print("Composer 生成笔记...")
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
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    print("[OK] 笔记图片: 共 {} 张".format(inserted))
    note_path = composer.save_note(file_base, note_with_images, str(note_dir))
    print("\n[OK] 已保存:", note_path)
    _regenerate_graph()
    _rebuild_embedding_index()


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

    print("网页提取:", url)
    result = extract_from_url(url, output_dir=str(img_dir))
    print("  标题: {}".format(result["title"][:60]))
    print("  文本: {} 字 | 图片: {} 张".format(len(result["text"]), len(result["images"])))

    vision_results = []
    if result["images"]:
        print("Vision 分析图片...")
        vision_results = VisionAgent().execute(result["images"])

    print("Composer 生成笔记...")
    composer = ComposerAgent()
    note_content = composer.execute(
        file_base=file_base,
        timestamped_text=result["text"][:15000],
        vision_results=vision_results,
        strategy={"note_style": "detailed", "focus_areas": []},
        revision_feedback="来源网页: {}\n整理成结构化学习笔记。".format(url),
    )
    note_with_images = insert_images_by_content(note_content, vision_results, file_base)
    note_with_images = _fix_broken_markdown_images(note_with_images)
    inserted = count_inserted_images(note_content, note_with_images)
    print("[OK] 图片: %s 张插入/保留" % inserted)
    note_path = composer.save_note(file_base, note_with_images, str(note_dir))
    print("\n[OK] 已保存:", note_path)
    _regenerate_graph()
    _rebuild_embedding_index()


# ════════════════════════════════════════════════════════════════
# 交互模式 & CLI
# ════════════════════════════════════════════════════════════════

def _run_with_cancel(description: str, fn, *args, **kwargs):
    """执行一个可能长时间运行的操作，支持 Ctrl+C 取消"""
    import time as _time
    print("[{}] {}... (Ctrl+C 取消)".format(
        _time.strftime("%H:%M:%S"), description))
    try:
        fn(*args, **kwargs)
    except KeyboardInterrupt:
        print("\n[!] 操作已取消\n")


def cmd_interactive():
    agent = NoteWeaverAgent()
    initial = agent.run("")
    print(initial)
    print()
    print("输入：问题(问答) | 视频/PDF/URL(处理) | 重排 笔记名(重排)")
    print("      graph(图谱) | /stop(取消当前任务) | /quit(退出)")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C 在输入提示符时，不退出，继续
            print("\n[!] 按 Ctrl+C 取消当前操作，输入 /quit 退出")
            continue
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("再见！")
            break
        if user_input.lower() in ("/stop", "/pause", "stop", "pause"):
            print("[!] 没有正在运行的任务")
            continue

        # ── 辅助命令 ──
        if user_input.lower() in ("/stats",):
            user_input = "学习统计"
        if user_input.lower() in ("graph", "/graph", "/kg", "知识图谱"):
            cmd_graph()
            print()
            continue
        if user_input.lower().startswith("/search "):
            user_input = user_input[8:]

        is_regenerate = any(user_input.startswith(kw) for kw in ("重排", "排版", "regenerate"))

        # ── 命令分发（每条命令都包在 _run_with_cancel 里） ──
        try:
            if _is_note_file(user_input):
                _run_with_cancel("重排笔记", cmd_regenerate_note, _resolve_media_path(user_input))
            elif _is_note_dir(user_input):
                _run_with_cancel("批量重排", cmd_batch_regenerate, _resolve_media_path(user_input))
            elif _is_pdf_file(user_input):
                _run_with_cancel("PDF处理", cmd_from_pdf, _resolve_media_path(user_input))
            elif is_regenerate:
                search = user_input
                for kw in ("重排", "排版", "regenerate"):
                    search = search.replace(kw, "", 1).strip()
                if not search:
                    print("要重排哪个笔记？例如：重排 04.BCD工艺流程")
                elif _is_note_file(search):
                    _run_with_cancel("重排笔记", cmd_regenerate_note, _resolve_media_path(search))
                else:
                    extracted = _extract_path_from_text(user_input)
                    if extracted:
                        path, is_file = extracted
                        if is_file and _is_note_file(path):
                            _run_with_cancel("重排笔记", cmd_regenerate_note, path)
                        elif not is_file and _is_note_dir(path):
                            _run_with_cancel("批量重排", cmd_batch_regenerate, path)
                        else:
                            _run_with_cancel("搜索重排", _search_and_regenerate, search, user_input)
                    else:
                        _run_with_cancel("搜索重排", _search_and_regenerate, search, user_input)
            elif _is_media_file(user_input):
                _run_with_cancel("视频处理", cmd_process, _resolve_media_path(user_input))
            elif _is_media_dir(user_input):
                _run_with_cancel("批量处理", cmd_batch, _resolve_media_path(user_input))
            elif _is_url(user_input):
                _run_with_cancel("视频下载", cmd_from_url, _extract_url_from_text(user_input))
            elif _is_web_url(user_input):
                _run_with_cancel("网页处理", cmd_from_web, _extract_url_from_text(user_input))
            elif user_input.lower().startswith("from ") or user_input.lower().startswith("下载 "):
                target = user_input.split(" ", 1)[1].strip().strip("\"'")
                if _is_url(target):
                    _run_with_cancel("视频下载", cmd_from_url, target)
                elif _is_web_url(target):
                    _run_with_cancel("网页处理", cmd_from_web, target)
                elif _is_pdf_file(target):
                    _run_with_cancel("PDF处理", cmd_from_pdf, _resolve_media_path(target))
                else:
                    extracted = _extract_url_from_text(user_input)
                    if extracted:
                        if _is_url(extracted):
                            _run_with_cancel("视频下载", cmd_from_url, extracted)
                        else:
                            _run_with_cancel("网页处理", cmd_from_web, extracted)
                    else:
                        print("请提供有效的链接或 PDF 路径")
            else:
                extracted = _extract_path_from_text(user_input)
                if extracted:
                    path, is_file = extracted
                    if is_file:
                        if _is_note_file(path):
                            _run_with_cancel("重排笔记", cmd_regenerate_note, path)
                        elif _is_pdf_file(path):
                            _run_with_cancel("PDF处理", cmd_from_pdf, path)
                        else:
                            _run_with_cancel("视频处理", cmd_process, path)
                    else:
                        if _is_note_dir(path):
                            _run_with_cancel("批量重排", cmd_batch_regenerate, path)
                        else:
                            _run_with_cancel("批量处理", cmd_batch, path)
                else:
                    url = _extract_url_from_text(user_input)
                    if url:
                        if _is_url(url):
                            _run_with_cancel("视频下载", cmd_from_url, url)
                        else:
                            _run_with_cancel("网页处理", cmd_from_web, url)
                    else:
                        response = agent.run(user_input)
                        print()
                        print(response)
        except KeyboardInterrupt:
            # 意外穿透的 Ctrl+C 兜底
            print("\n[!] 操作已取消")

        print()


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
    parser.add_argument("--batch", help="批量处理目录")
    parser.add_argument("--graph", action="store_true", help="知识图谱可视化")
    parser.add_argument("--version", action="store_true", help="显示版本号")

    args = parser.parse_args()

    if args.version:
        print("NoteWeaver v{}".format(__version__))
        return

    for d in [config.txt_dir, config.note_dir, config.memory_dir, config.log_dir]:
        os.makedirs(d, exist_ok=True)

    if args.video:
        cmd_process(args.video)
    elif args.batch:
        cmd_batch(args.batch)
    elif args.graph:
        cmd_graph()
    elif args.message:
        msg = args.message.strip()
        if _is_note_file(msg):
            cmd_regenerate_note(_resolve_media_path(msg))
        elif _is_note_dir(msg):
            cmd_batch_regenerate(_resolve_media_path(msg))
        elif _is_pdf_file(msg):
            cmd_from_pdf(_resolve_media_path(msg))
        elif msg.lower() in ("graph", "/graph", "知识图谱"):
            cmd_graph()
        elif any(msg.startswith(kw) for kw in ("重排", "排版", "regenerate")):
            search = msg
            for kw in ("重排", "排版", "regenerate"):
                search = search.replace(kw, "", 1).strip()
            if not search:
                print("要重排哪个笔记？例如：重排 04.BCD工艺流程")
            elif _is_note_file(search):
                cmd_regenerate_note(_resolve_media_path(search))
            elif search:
                extracted = _extract_path_from_text(msg)
                if extracted:
                    path, is_file = extracted
                    if is_file and _is_note_file(path):
                        cmd_regenerate_note(path)
                    elif not is_file and _is_note_dir(path):
                        cmd_batch_regenerate(path)
                    else:
                        _search_and_regenerate(search, msg)
                else:
                    _search_and_regenerate(search, msg)
        elif _is_media_file(msg):
            cmd_process(_resolve_media_path(msg))
        elif _is_media_dir(msg):
            cmd_batch(_resolve_media_path(msg))
        elif _is_url(msg):
            cmd_from_url(_extract_url_from_text(msg))
        elif _is_web_url(msg):
            cmd_from_web(_extract_url_from_text(msg))
        elif msg.lower().startswith("from ") or msg.lower().startswith("下载 "):
            target = msg.split(" ", 1)[1].strip().strip("\"'")
            if _is_url(target):
                cmd_from_url(target)
            elif _is_web_url(target):
                cmd_from_web(target)
            elif _is_pdf_file(target):
                cmd_from_pdf(_resolve_media_path(target))
            else:
                raw = _extract_url_from_text(msg)
                if raw:
                    cmd_from_web(raw) if not _is_url(raw) else cmd_from_url(raw)
                else:
                    print("请提供有效的链接或 PDF 路径")
        else:
            raw = _extract_url_from_text(msg)
            if raw:
                cmd_from_web(raw) if not _is_url(raw) else cmd_from_url(raw)
            else:
                cmd_agent(msg)
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
