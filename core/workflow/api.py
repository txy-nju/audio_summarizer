from typing import List, Dict
from core.workflow.video_summary.graph import build_video_summary_graph

def summarize_video(transcript: str, keyframes: List[Dict], user_prompt: str = "请结合画面与语音，给出一个全面、客观的高质量视频总结。") -> str:
    """
    外部调用接口：启动多模态视频总结工作流。
    该接口严格遵守架构契约的分层设计，对外屏蔽了 State 和 Graph 的复杂性。
    
    :param transcript: 视频语音识别出的完整文本
    :param keyframes: 视频关键帧列表，格式必须为 [{"time": "...", "image": "..."}]
    :param user_prompt: 用户的特殊总结侧重点要求
    :return: 最终生成的总结内容 (即工作流末态的 draft_summary)
    """
    # 1. 获取已编译的工作流引擎
    workflow_app = build_video_summary_graph()
    
    # 2. 构造符合 VideoSummaryState 契约的初始状态
    initial_state = {
        "transcript": transcript,
        "keyframes": keyframes,
        "user_prompt": user_prompt,
        "text_insights": "",
        "visual_insights": "",
        "draft_summary": "",
        # [Self-RAG 升级] 初始化新增的质量控制字段
        "hallucination_score": "",
        "usefulness_score": "",
        "feedback_instructions": "",
        "revision_count": 0
    }
    
    # 3. 阻塞式执行工作流并获取终态数据
    final_state = workflow_app.invoke(initial_state)
    
    # 4. 抽取对外唯一关心的产物进行返回
    return final_state.get("draft_summary", "")