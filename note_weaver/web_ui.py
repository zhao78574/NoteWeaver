"""NoteWeaver Web UI — 独立浏览器界面

启动: python web_ui.py
然后打开 http://localhost:7860
"""

import gradio as gr
import os, sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from note_weaver.agent import NoteWeaverAgent

# 全局 Agent 实例（保持状态）
agent = NoteWeaverAgent()

# 自定义 CSS
CSS = """
.container { max-width: 900px; margin: auto; }
.header { text-align: center; padding: 20px; }
.header h1 { color: #f59e0b; font-size: 2em; margin: 0; }
.header p { color: #6b7280; }
.stats-bar {
    display: flex; gap: 15px; justify-content: center;
    padding: 10px; background: #1f2937; border-radius: 10px;
    margin-bottom: 15px; color: #d1d5db; font-size: 0.9em;
}
.stats-bar span { color: #f59e0b; font-weight: bold; }
"""


def get_stats_html():
    """获取当前知识库统计"""
    cfg_path = os.path.join(PARENT_DIR, "data", "memory_db", "knowledge_graph.json")
    concepts = 0
    relations = 0
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, encoding="utf-8") as f:
            kg = json.load(f)
        concepts = len(kg.get("concepts", []))
        relations = len(kg.get("relations", []))

    note_dir = os.path.join(PARENT_DIR, "data", "Note")
    note_count = 0
    if os.path.isdir(note_dir):
        for root, dirs, files in os.walk(note_dir):
            note_count += len([f for f in files if f.endswith(".md")])

    return f"""
    <div class="stats-bar">
        <div>📝 <span>{note_count}</span> 篇笔记</div>
        <div>🧠 <span>{concepts}</span> 概念</div>
        <div>🔗 <span>{relations}</span> 关联</div>
    </div>
    """


def respond(message, history):
    """处理用户消息"""
    if not message or not message.strip():
        return ""
    response = agent.run(message)
    return response


# ============================================================
# 构建界面
# ============================================================

with gr.Blocks(title="NoteWeaver") as demo:
    # 头部
    gr.HTML("""
        <div class="header">
            <h1>🤖 NoteWeaver</h1>
            <p>AI 笔记数字助理 — 告诉我路径，我来处理</p>
        </div>
    """)

    # 统计栏
    stats_display = gr.HTML(get_stats_html())

    # 聊天区
    chat = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="输入你的指令...", container=False),
    )

    # 定期刷新统计
    demo.load(get_stats_html, None, stats_display)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NoteWeaver Web UI")
    print("  打开浏览器访问: http://localhost:7860")
    print("=" * 50 + "\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )
