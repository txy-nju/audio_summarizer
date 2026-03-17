from core.workflow.video_summary.state import VideoSummaryState

def vision_analyzer_node(state: VideoSummaryState) -> dict:
    """
    视觉处理节点。调用多模态大模型（Vision LLM），传入 keyframes 和对应的时间戳，
    要求模型描述画面中的关键动作、PPT 文本或场景变化。
    
    :param state: VideoSummaryState
    :return: dict 包含更新的 visual_insights 字段
    """
    # 存根：架构占位符
    return {"visual_insights": ""}