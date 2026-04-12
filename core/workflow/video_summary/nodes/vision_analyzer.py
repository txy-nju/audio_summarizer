import os
import json
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.tools.search_tools import execute_tavily_search

# [重构优化] 引入 Tool Handler Mapping (策略模式)，彻底消除僵硬的 if-else 分支
AVAILABLE_TOOLS = {
    "tavily_search": lambda args: execute_tavily_search(args.get("query", ""))
}

def vision_analyzer_node(state: VideoSummaryState) -> dict:
    """
    视觉处理节点 (Vision Agent) + ReAct 主动搜索能力。
    调用多模态大模型，传入 keyframes，并赋予模型“以图生文再搜索”的工具调用能力。
    """
    keyframes = state.get("keyframes", [])
    user_prompt = state.get("user_prompt", "")
    
    if not keyframes:
        return {"visual_insights": "未提取到任何视频关键帧，无法进行视觉分析。"}

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("在执行视觉分析节点时，未能找到 OPENAI_API_KEY 环境变量。")
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    system_prompt = (
        "你是一个极其专业的多模态视频视觉分析专家 (Vision Agent)。我将为你提供按时间顺序排列的视频关键帧截图。\n"
        "你的任务是：仔细观察每一帧画面，提取出对于理解视频内容至关重要的视觉信息（屏幕/PPT内容、动作与场景切换）。\n\n"
        "【主动求知工具授权 (Tool Calling - 以图生文再搜索)】：\n"
        "对于画面中出现的任何角色、虚构形象、表情包 (meme)、品牌 logo 或文化符号，"
        "只要你**无法立刻自信地说出其确切名称、IP 归属或流行背景**，你就必须立刻停下，"
        "调用 `tavily_search` 工具去搜索查明——**即便你能描述其外观特征**（如'黄色生物'、'狗坐在着火的屋里'），"
        "仅凭外貌描述就输出结论是被明令禁止的，必须先搜索确认身份后再作结论，绝不可凭空捏造名称。\n"
        "执行策略：在脑海中生成该形象的详尽外观特征文字描述（如：'yellow anthropomorphic cartoon character pushing a shopping cart, looking at shelf plushies of same character'），"
        "将这串描述直接作为 query 参数，调用 `tavily_search` 搜索其真实身份或出处。\n\n"
        "【约束】：\n"
        "- 保持绝对客观，【绝不能】发生幻觉。\n"
        "- 最终的视觉分析报告格式必须清晰易读，并【必须】在每条结论前附上对应的时间戳（如 [00:15]）。"
    )
    
    content_list = [
        {"type": "text", "text": f"【用户总结侧重点要求】：\n{user_prompt}\n\n以下是按时间顺序截取的视频关键帧："}
    ]
    
    for frame in keyframes:
        time_str = frame.get("time", "未知时间")
        base64_img = frame.get("image", "")
        if not base64_img:
            continue
        content_list.append({"type": "text", "text": f"--- 当前画面时间戳: [{time_str}] ---"})
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "auto"}})
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "If you encounter an unknown image meme, UI, or character, carefully describe its visual features and text, then use this tool to search the internet for its meaning or origin.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The detailed text description of the image to search for, e.g., 'This is fine dog sitting in fire meme meaning'"}},
                    "required": ["query"]
                }
            }
        }
    ]
    
    print(f"  -> [Vision Analyzer Node] Calling Vision LLM API with {len(keyframes)} frames and ReAct tools...")
    
    model_name = os.getenv("OPENAI_VISION_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))
    max_tool_calls = 3 # 防止进入工具调用的死循环
    loop_count = 0
    
    while loop_count < max_tool_calls:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages, # type: ignore
                tools=tools, # type: ignore
                temperature=0.2, # 较低的温度，要求客观识图与决策
                max_tokens=2048 
            )
            response_message = response.choices[0].message
            
            # [ReAct] 将模型的回答（哪怕是工具调用意图）忠实地追加进上下文历史
            messages.append(response_message) # type: ignore
            
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  -> [Vision Analyzer] 🧠 Agent initiated Tool Call: {function_name} with {function_args}")
                    
                    # 拦截并在本地执行对应的 Python 工具函数 (基于 Mapping 优雅调度)
                    handler = AVAILABLE_TOOLS.get(function_name)
                    if handler:
                        tool_result = handler(function_args)
                    else:
                        tool_result = f"Tool Execution Failed: Unknown tool {function_name}"
                        
                    # 将外网获取的真实知识（或失败警告）作为 Tool Message 塞回给大模型
                    messages.append({ # type: ignore
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
                
                loop_count += 1
                # 带着这些新获得的“外挂知识”，继续循环请求大模型，直至它认为资料已充分，可以输出最终的 Markdown 报告
                continue 
            else:
                # 如果没有 tool_calls，说明大模型已经“大彻大悟”，输出了最终的文字报告
                visual_insights = response_message.content
                print(f"  -> [Vision Analyzer Node] Visual extraction finished after {loop_count} tool calls.")
                return {"visual_insights": visual_insights}
                
        except Exception as e:
            print(f"  -> [Vision Analyzer Node] Error during Vision API call: {str(e)}")
            return {"visual_insights": f"[系统自动提示]：视觉分析提取失败或中途工具调用异常：{str(e)}"}
            
    # 如果超过了最大的 ReAct 循环次数，强行兜底退出
    fallback_content = "[系统自动提示]：视觉分析超时，由于模型陷入频繁查询死循环被强制熔断。请手动简化需求。"
    # 修复：提取最后一次 response_message (Assistant 角色) 的 content
    if 'response_message' in locals() and hasattr(response_message, "content") and response_message.content:
        fallback_content = response_message.content

    return {"visual_insights": fallback_content}