import os
import json
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState

# [消除魔法数字]: 将最大重写次数提炼为模块级常量，与幻觉评分器保持一致
MAX_REVISIONS = 2

def usefulness_grader_node(state: VideoSummaryState) -> dict:
    """
    [Self-RAG 架构升级] 有用性评分器 (Usefulness Grader)。
    第二道质量防线：在确认草稿无幻觉的前提下，评估其是否完美回答了用户的特定的总结侧重点 (user_prompt)。
    必须输出严格的 JSON 结构。
    
    :param state: VideoSummaryState
    :return: dict 更新 usefulness_score 和 feedback_instructions
    """
    draft = state.get("draft_summary", "")
    user_prompt = state.get("user_prompt", "")
    revision_count = state.get("revision_count", 0)

    # 1. 熔断与防死循环
    # 如果草稿为空、达到重写上限，或者用户根本没有提供额外的特定要求，直接绿灯放行，绝不浪费 Token
    if not draft or not user_prompt or not user_prompt.strip() or revision_count >= MAX_REVISIONS:
        return {"usefulness_score": "yes", "feedback_instructions": ""}

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("在执行有用性评分节点时，未能找到 OPENAI_API_KEY 环境变量。")
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 2. 构造 System Prompt，要求强制 JSON 输出
    system_prompt = (
        "你是一名严格的用户体验评估官 (Usefulness Grader)。\n"
        "你的唯一任务是评估【总结草稿】是否充分、准确地回应了用户的【特定总结侧重点】。\n\n"
        "【严格 JSON 格式输出要求】：\n"
        "你必须输出一个合法且格式化良好的 JSON 对象，包含以下两个字段：\n"
        "1. \"score\": 字符串，只能是 \"yes\"（草稿很好地满足了用户需求）或 \"no\"（草稿偏题、遗漏了用户的核心要求）。\n"
        "2. \"reason\": 字符串，如果 score 为 \"no\"，请给出极其明确的修改指令（例如：'用户要求侧重讲解微服务架构，但草稿完全没有提到，请在第二段大幅补充架构相关的技术细节'）；如果 score 为 \"yes\"，置为空字符串 \"\"。"
    )

    user_content = (
        f"【用户的特定总结侧重点】：\n{user_prompt}\n\n"
        f"【待评估的总结草稿】：\n{draft}"
    )

    print(f"  -> [Usefulness Grader] Checking if draft meets user prompt (Revision {revision_count})...")

    # 3. 执行评估 API 调用
    try:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}, # 开启 JSON Mode 获取确定性结果
            temperature=0.0, # 评估节点必须保持绝对客观冷静
        )
        
        result_json_str = response.choices[0].message.content.strip()
        result = json.loads(result_json_str)
        
        # 兼容处理，默认放行
        score = result.get("score", "yes").lower()
        reason = result.get("reason", "")
        
        if score == "no":
            print("  -> [Usefulness Grader] Result: NO (Draft missed user intent). Routing back to Drafter.")
            feedback = f"【偏题拦截 - 需求未满足】：\n{reason}"
            return {"usefulness_score": "no", "feedback_instructions": feedback}
        else:
            print("  -> [Usefulness Grader] Result: YES (Draft is useful). Final Approval.")
            return {"usefulness_score": "yes", "feedback_instructions": ""}
            
    except Exception as e:
        # [增强可观察性] 异常降级兜底：记录日志并放行
        print(f"  -> [Usefulness Grader] Error or Invalid JSON: {str(e)}. Fallback to YES usefulness.")
        return {"usefulness_score": "yes", "feedback_instructions": ""}