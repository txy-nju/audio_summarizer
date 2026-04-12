import os
import json
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.tools.search_tools import execute_tavily_search

# [重构优化] 引入 Tool Handler Mapping (策略模式)，彻底消除僵硬的 if-else 分支
# 未来如果需要新增更多的 Tool，只需在此字典中注册对应的处理函数即可，符合开闭原则 (OCP)
AVAILABLE_TOOLS = {
    "tavily_search": lambda args: execute_tavily_search(args.get("query", ""))
}

def text_analyzer_node(state: VideoSummaryState) -> dict:
    """
    纯文本处理节点 (Audio Agent) + ReAct 主动搜索能力。
    负责读取 transcript，提取核心观点、章节主题和金句。如果遇到生僻词汇或梗，主动发起 Web Search 获取背景解释。
    """
    transcript = state.get("transcript", "")
    user_prompt = state.get("user_prompt", "")
    
    if not transcript or not transcript.strip():
        return {"text_insights": "未提供有效的语音转录文本。"}

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("在执行文本分析节点时，未能找到 OPENAI_API_KEY 环境变量。")
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    system_prompt = (
        "你是一位极其资深的视频内容分析专家 (Audio Agent)。你的核心任务是：仔细阅读提供的视频语音识别文本（Transcript），"
        "过滤掉一切口语化的废话，提取出高度结构化的核心信息。\n\n"
        "【主动求知工具授权 (Tool Calling)】：\n"
        "对于文本中出现的任何专业术语、网络热梗、新造词汇、人名、角色名或品牌名，"
        "只要你**无法立刻自信地给出其准确含义或来源**，你就必须立刻暂停，"
        "调用 `tavily_search` 去互联网查询真实背景知识，并将查明的内容体现在最终总结中，绝不能凭空捏造词义。\n\n"
        "【输出强制规范】：\n"
        "在查明所有疑惑之后，请务必输出以下三个维度的分析结果，并保持 Markdown 结构：\n"
        "1. 🌟 核心观点 (Core Insights)：提炼视频传达的最重要的观点。\n"
        "2. 📑 章节主题大纲 (Chapter Outlines)：按照逻辑流向，梳理骨架。\n"
        "3. 💡 关键金句与术语释义 (Key Quotes & Jargons)：摘录金句，若曾使用搜索工具查明了生僻梗或术语，请在此处附上简短释义。"
    )
    
    user_content = f"【用户总结侧重点要求】\n{user_prompt}\n\n【视频语音识别完整文本】\n{transcript}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "当你在语音转录文本中遇到无法理解的网络热词、新造梗或极其生僻的专业术语时，立刻调用此工具去互联网搜索其最新的真实含义。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The exact search query to find the explanation, e.g. 'mhc 网络用语意思' or 'DeepSeek V3 论文亮点'"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    print("  -> [Text Analyzer Node] Calling LLM API for text insights extraction with ReAct tools...")
    
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    max_tool_calls = 2 # 文本分析所需的深挖深度较浅，限制2次即可防止死循环
    loop_count = 0
    
    while loop_count < max_tool_calls:
        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                temperature=0.3, # 较低的温度以保证分析结果的客观性与稳定性
            )
            response_message = response.choices[0].message
            
            messages.append(response_message) # type: ignore
            
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  -> [Text Analyzer] 🧠 Agent initiated Tool Call: {function_name} with {function_args}")
                    
                    # 拦截并在本地执行对应的 Python 工具函数 (基于 Mapping 优雅调度)
                    handler = AVAILABLE_TOOLS.get(function_name)
                    if handler:
                        tool_result = handler(function_args)
                    else:
                        tool_result = f"Tool Execution Failed: Unknown tool {function_name}"
                        
                    messages.append({ # type: ignore
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
                
                loop_count += 1
                continue
            else:
                insights = response_message.content
                print(f"  -> [Text Analyzer Node] Extraction finished after {loop_count} tool calls.")
                return {"text_insights": insights}
                
        except Exception as e:
            print(f"  -> [Text Analyzer Node] Error during API call: {str(e)}")
            return {"text_insights": f"[系统自动提示]：文本分析提取失败，LLM 或工具调用异常：{str(e)}"}
            
    fallback_content = "[系统自动提示]：文本分析超时，由于模型陷入频繁查询死循环被强制熔断。"
    if 'response_message' in locals() and hasattr(response_message, "content") and response_message.content:
        fallback_content = response_message.content
    return {"text_insights": fallback_content}