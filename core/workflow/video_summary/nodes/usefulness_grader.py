import os
import json
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState

# [消除魔法数字]: 将最大重写次数提炼为模块级常量，与幻觉评分器保持一致
MAX_REVISIONS = 2

def usefulness_grader_node(state: VideoSummaryState) -> dict:
    """
    有用性审查节点。

    地位:
    - 位于 hallucination_grader_node 之后，是质量闭环的第二道防线。
    - 在事实基本成立的前提下，检查草稿是否真正回应了用户诉求（含第二阶段人类指导）。

    任务:
    - 对比 draft_summary 与 user_prompt/human_guidance。
    - 以 JSON Mode 返回是否满足用户需求。
    - 若偏题或遗漏重点，则输出定向修改指令并回流到成文节点。
    - 当无审核要求或达到重写上限时直接放行。

    主要输入:
    - state["draft_summary"]
    - state["user_prompt"]
    - state["human_guidance"]
    - state["revision_count"]

    主要输出:
    - usefulness_score: "yes" 或 "no"。
    - feedback_instructions: 供 fusion_drafter_node 使用的补充修改指令。

    :param state: VideoSummaryState
    :return: dict 更新 usefulness_score 和 feedback_instructions
    """
    draft = state.get("draft_summary", "")
    user_prompt = state.get("user_prompt", "")
    human_guidance = state.get("human_guidance", "")
    revision_count = state.get("revision_count", 0)

    # user_prompt 与 human_guidance 共同构成有用性评分的审核要求。
    review_requirements = []
    if isinstance(user_prompt, str) and user_prompt.strip():
        review_requirements.append(f"【用户原始总结侧重点】\n{user_prompt.strip()}")
    if isinstance(human_guidance, str) and human_guidance.strip():
        review_requirements.append(f"【人类审批补充指导】\n{human_guidance.strip()}")
    review_requirements_text = "\n\n".join(review_requirements)

    # 1. 熔断与防死循环
    # 如果草稿为空、达到重写上限，或没有任何审核要求，直接放行，避免无效调用。
    if not draft or revision_count >= MAX_REVISIONS or not review_requirements_text:
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
        f"【审核要求（原始需求 + 人类审批意见）】：\n{review_requirements_text}\n\n"
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