from typing import TypedDict, List, Dict

class VideoSummaryState(TypedDict):
    # 输入层数据
    transcript: str                 # 视频语音识别文本 (ASR/Whisper 输出)
    keyframes: List[Dict]           # 关键帧列表，包含 base64 数据及时间戳：[{"time": "00:15", "image": "base64_str"}]
    user_prompt: str                # 用户具体的总结侧重点
    
    # 中间态数据
    text_insights: str              # 语音文本的提炼结果
    visual_insights: str            # 关键帧画面的动作/图表解析结果
    
    # 输出与循环控制
    draft_summary: str              # 当前生成的融合总结草稿
    
    # [Self-RAG 架构升级新增字段]
    hallucination_score: str        # 幻觉评分器的裁定结果 (取值: "yes" 表示有幻觉, "no" 表示无幻觉)
    usefulness_score: str           # 有用性评分器的裁定结果 (取值: "yes" 表示有用, "no" 表示无用/偏题)
    feedback_instructions: str      # 精确的反馈修改指导，替代原先单一模糊的 critique

    revision_count: int             # 重写次数