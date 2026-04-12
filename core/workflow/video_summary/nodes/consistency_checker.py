import os
from openai import OpenAI
from core.workflow.video_summary.state import VideoSummaryState

def consistency_checker_node(state: VideoSummaryState) -> dict:
    """
    幻觉与一致性审查员 (Reflector/Evaluator)。
    核对 draft_summary 中的内容：
    1. 是否有“文本里没提到且画面里也没有”的幻觉？
    2. 是否遗漏了重要的 PPT 画面解析或用户要求的侧重点？
    如果不合格，输出具体的修改点至 critique；如果合格，输出空字符串以触发结束路由。
    
    :param state: VideoSummaryState
    :return: dict 包含更新的 critique 字段
    """
    draft = state.get("draft_summary", "")
    text_insights = state.get("text_insights", "")
    visual_insights = state.get("visual_insights", "")
    user_prompt = state.get("user_prompt", "")
    revision_count = state.get("revision_count", 0)

    # 1. 熔断机制：如果草稿为空，或者已经达到最大循环次数，不再浪费 Token 进行审查，直接放行
    # 根据 router 的逻辑，revision_count >= 2 会直接终止，这里做双重保险
    if not draft or revision_count >= 2:
        return {"critique": ""}

    # 2. 获取环境变量凭证
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("在执行一致性审查节点时，未能找到 OPENAI_API_KEY 环境变量。")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 3. 构造 System Prompt (严厉的审查官人设)
    system_prompt = (
        "你是一个极其严苛且无情的视频总结质量审查官 (Consistency Checker)。\n"
        "你的唯一任务是对另一位AI生成的【视频总结草稿】进行事实核查与多模态一致性审查。\n\n"
        "【审查铁律】：\n"
        "你必须且只能基于我提供的【听觉侧提炼】与【视觉侧提炼】来评判草稿，绝对不能引入你自身的外部知识进行脑补。\n\n"
        "【审查执行标准】：\n"
        "1. ❌ 幻觉检测：草稿中是否出现了听觉侧和视觉侧中【均未提及】的内容？（重点打击无中生有的数据、功能或概念）\n"
        "2. ⚠️ 遗漏检测：草稿是否遗漏了听觉侧或视觉侧中的重大信息（如视觉侧中明确提到的PPT核心数据、关键操作步骤）？\n"
        "3. 🎯 需求偏离：草稿的行文重点是否偏离了用户的【总结侧重点要求】？\n\n"
        "【强制输出规范】：\n"
        "- 如果你发现上述任何严重问题，请直接输出清晰、具体、可执行的【修改建议 (Critique)】。例如：'草稿第2段捏造了80%的增长率，原始提炼中并未提及，请删除；遗漏了视觉侧[00:15]处的钛金属材质描述，请在对应段落补充。'\n"
        "- 如果你认为草稿质量非常高，事实准确且完美融合了图文，请【只输出】七个字母：'APPROVE'。不要输出任何其他的标点符号或客套话。"
    )

    # 4. 组装 User Content
    user_content = (
        f"【用户的总结侧重点要求】：\n{user_prompt}\n\n"
        f"====== 事实核查源 (Truth Sources) ======\n"
        f"【听觉侧原始提炼】：\n{text_insights}\n\n"
        f"【视觉侧原始提炼】：\n{visual_insights}\n"
        f"========================================\n\n"
        f"【待严格审查的总结草稿】：\n{draft}"
    )

    print(f"  -> [Consistency Checker Node] Evaluating Draft (Revision {revision_count})...")

    # 5. 执行审查 API 调用
    try:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            # 审查官必须像机器一样绝对严谨客观，温度降至 0 以剥离幻觉
            temperature=0.0,
        )

        critique_response = response.choices[0].message.content.strip()

        # 6. 解析判定结果，与 Router 路由层进行状态握手
        if critique_response.upper() == "APPROVE" or ("APPROVE" in critique_response.upper() and len(critique_response) < 15):
            print("  -> [Consistency Checker Node] Result: APPROVED. (Quality Gate Passed)")
            # 输出空字符串，router.py 会将其判定为 "Approve" 分支，流向 END
            return {"critique": ""}
        else:
            print("  -> [Consistency Checker Node] Result: REJECTED with Critique. (Routing back to Drafter)")
            # 返回修改意见，router.py 会将其判定为 "Needs Revision" 分支，流回 fusion_drafter_node
            return {"critique": critique_response}

    except Exception as e:
        print(f"  -> [Consistency Checker Node] Error during evaluation: {str(e)}")
        # 若审查系统自身发生 API 异常，为防止状态机卡死或无限报错死循环，强制放行 (Approve)
        return {"critique": ""}