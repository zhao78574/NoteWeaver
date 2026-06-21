"""内容提取器 — 从 PDF / 网页中提取文本和图片，供笔记管线使用

用法:
    from note_weaver.utils.extractors import extract_from_pdf, extract_from_url

    # PDF
    result = extract_from_pdf("paper.pdf")
    print(result["title"], len(result["text"]), len(result["images"]))

    # 网页
    result = extract_from_url("https://example.com/article")
    print(result["title"], len(result["text"]))
"""

import os
import re
import io
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from note_weaver.utils.logger import logger


# ════════════════════════════════════════════════════════════════
# PDF 提取
# ════════════════════════════════════════════════════════════════

def extract_from_pdf(pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """从 PDF 提取文本和图片

    Args:
        pdf_path: PDF 文件路径
        output_dir: 图片输出目录（None = 不保存图片）

    Returns:
        {
            "title": str,           # PDF 标题（取文件名或首段）
            "text": str,            # 全部文本
            "images": [str],        # 提取的图片路径列表
            "pages": int,           # 总页数
        }
    """
    import fitz  # PyMuPDF

    pdf_path = str(Path(pdf_path).resolve())
    file_base = Path(pdf_path).stem
    img_dir = Path(output_dir) if output_dir else None

    doc = fitz.open(pdf_path)
    title = doc.metadata.get("title", "") or file_base
    all_text = []
    image_paths = []

    logger.info(f"[Extractor] PDF 提取: {file_base}.pdf ({doc.page_count} 页)")

    # 跨页图片去重（相同 MD5 hash 只保留第一次出现的）
    seen_hashes: set = set()

    for page_num in range(doc.page_count):
        page = doc[page_num]
        # 提取文本
        text = page.get_text()
        if text.strip():
            all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text.strip()}")

        # 提取图片
        if img_dir:
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]

                # 跳过小图标（< 5KB）
                if len(img_bytes) < 5120:
                    continue

                # 跨页去重（按内容 hash）
                img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                if img_hash in seen_hashes:
                    logger.info(f"[Extractor]   ⏭ 跳过重复图 p{page_num+1}_{img_idx} (hash={img_hash})")
                    continue
                seen_hashes.add(img_hash)

                img_name = f"{file_base}_p{page_num+1}_{img_idx}_{img_hash}.{ext}"
                img_path = img_dir / img_name

                img_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                image_paths.append(str(img_path))

    page_count = doc.page_count
    doc.close()

    result = {
        "title": title,
        "text": "\n\n".join(all_text),
        "images": image_paths,
        "pages": page_count,
    }
    logger.info(f"[Extractor] PDF 完成: {len(result['text'])} 字, {len(image_paths)} 张图")
    return result


# ════════════════════════════════════════════════════════════════
# 网页提取
# ════════════════════════════════════════════════════════════════

def extract_from_url(url: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """从网页提取正文文本和图片

    Args:
        url: 网页链接
        output_dir: 图片下载目录（None = 不下载图片）

    Returns:
        {
            "title": str,           # 网页标题
            "text": str,            # 正文文本
            "images": [str],        # 下载的图片路径列表
            "source_url": str,      # 原始 URL
        }
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    }

    logger.info(f"[Extractor] 网页提取: {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.encoding = resp.apparent_encoding
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # 提取标题
    title = ""
    for tag in ["h1", "title"]:
        el = soup.find(tag)
        if el and el.get_text(strip=True):
            title = el.get_text(strip=True)
            break
    # 清理标题
    title = re.sub(r'\s+', ' ', title).strip()

    # 移除无用元素
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "noscript"]):
        tag.decompose()

    # 提取正文（优先 article，否则 body）
    body = soup.find("article") or soup.find("body") or soup
    text = body.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 截取太长内容（最大 30000 字）
    if len(text) > 30000:
        text = text[:30000] + "\n\n...(内容过长已截断)"

    # 提取图片（可选下载）
    image_paths = []
    if output_dir:
        img_out = Path(output_dir)
        img_out.mkdir(parents=True, exist_ok=True)

        img_tags = soup.find_all("img")
        for i, img in enumerate(img_tags):
            src = img.get("src") or img.get("data-src") or ""
            if not src or src.startswith("data:"):
                continue
            if not src.startswith("http"):
                # 补全相对路径
                parsed = urlparse(url)
                if src.startswith("//"):
                    src = f"{parsed.scheme}:{src}"
                elif src.startswith("/"):
                    src = f"{parsed.scheme}://{parsed.netloc}{src}"
                else:
                    src = f"{parsed.scheme}://{parsed.netloc}/{src}"

            try:
                img_resp = requests.get(src, headers=headers, timeout=10)
                if len(img_resp.content) < 10240:
                    continue
                ext = Path(urlparse(src).path).suffix or ".jpg"
                img_name = f"web_img_{i:03d}{ext}"
                img_path = img_out / img_name
                with open(img_path, "wb") as f:
                    f.write(img_resp.content)
                image_paths.append(str(img_path))
            except Exception:
                continue

    result = {
        "title": title or url,
        "text": text,
        "images": image_paths,
        "source_url": url,
    }
    logger.info(f"[Extractor] 网页完成: {title[:50]} ({len(text)} 字, {len(image_paths)} 张图)")
    return result
