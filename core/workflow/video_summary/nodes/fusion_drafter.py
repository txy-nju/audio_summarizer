import os
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState

def fusion_drafter_node(state: VideoSummaryState) -> dict:
    """
    核心的“组装节点” (Synthesizer)。
    接收并汇聚来自并行层的 text_insights 和 visual_insights，根据时间戳或逻辑相关性进行“图文对齐”。
    生成结构化的综合 draft_summary。如果存在 critique，则必须结合修改意见重新生成修正版草稿。
    
    :param state: VideoSummaryState
    :return: dict 包含更新的 draft_summary 和增加的 revision_count
    """
    current_count = state.get("revision_count", 0)
    text_insights = state.get("text_insights", "")
    visual_insights = state.get("visual_insights", "")
    user_prompt = state.get("user_prompt", "")
    critique = state.get("critique", "")

    # 1. 获取环境变量凭证
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("在执行融合组装节点时，未能找到 OPENAI_API_KEY 环境变量。")
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 2. 构造 System Prompt (基于架构设计：强调图文对齐)
    system_prompt = (
        "你是一个顶级的多模态视频内容综合编辑与深度报告撰写专家。\n"
        "由于视频的语音和画面是被并行分离提取的，你的核心任务是将【纯文本提炼分析（听觉侧）】与【关键帧多模态分析（视觉侧）】进行完美的“图文对齐”与“逻辑融合”，"
        "最终生成一份高质量、连贯且逻辑自洽的视频深度总结报告。\n\n"
        "【架构级约束指令】：\n"
        "1. 🔗 强制图文对齐：请敏锐地寻找听觉分析与视觉分析在时间线（如 [00:15]）或逻辑上的交叉点。必须采用“图文结合”的叙述方式，例如：“在讲解 XXX 概念时，画面同步展示了 YYY 走势图或操作界面”，绝不能孤立地将文字和画面分两块罗列。\n"
        "2. 🚫 消除认知矛盾：如果发现文本分析和视觉分析存在细微出入（如发音识别错误但画面正确），请以符合客观常理和画面事实的方式进行中和纠正。\n"
        "3. 📝 专业排版规范：输出必须使用易于阅读的 Markdown 语法。建议包含：【内容导读】、【核心图文融合解析（按大纲展开）】、【关键金句摘录】、【总结与升华】等模块。"
    )

    # 3. 反思机制 (Reflector Feedback) 介入
    if critique and critique.strip():
        system_prompt += (
            f"\n\n⚠️ 【重要警告：这是第 {current_count + 1} 次重写草稿】\n"
            "在上一版的草稿中，审查员 (Consistency Checker) 指出了以下幻觉、逻辑断层或重点遗漏。请务必在本次生成中着重修正并体现出来：\n"
            f"====== 审查修改意见 (Critique) ======\n"
            f"{critique}\n"
            f"====================================="
        )

    # 4. 组装 User Content
    user_content = (
        f"【用户期望的总结侧重点】：\n{user_prompt}\n\n"
        f"【听觉侧 - 文本提炼分析】：\n{text_insights}\n\n"
        f"【视觉侧 - 关键帧多模态分析】：\n{visual_insights}"
    )

    print(f"  -> [Fusion Drafter Node] Synthesizing parallel insights into Draft (Revision {current_count + 1})...")

    # 5. 执行 API 调用
    try:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            # 融合节点需要将碎片化信息组织为流畅文章，因此适度提高 temperature 以获取更好的文笔和行文组织能力
            temperature=0.5, 
        )
        draft = response.choices[0].message.content
        print("  -> [Fusion Drafter Node] Draft synthesized successfully.")
        
        return {
            "draft_summary": draft,
            "revision_count": current_count + 1
        }
    except Exception as e:
        print(f"  -> [Fusion Drafter Node] Error during synthesis: {str(e)}")
        # 将异常上抛，由后续路由或最终结果展现
        return {
            "draft_summary": f"[系统自动提示]：综合图文大纲失败，LLM 调用发生异常：{str(e)}",
            "revision_count": current_count + 1
        }