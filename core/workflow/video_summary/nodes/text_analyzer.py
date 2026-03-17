from core.workflow.video_summary.state import VideoSummaryState

def text_analyzer_node(state: VideoSummaryState) -> dict:
    """
    纯文本处理节点。负责读取 transcript，提取核心观点、章节主题和金句。这一步旨在过滤口语化的废话。
    
    :param state: VideoSummaryState
    :return: dict 包含更新的 text_insights 字段
    """
    # 存根：架构占位符
    return {"text_insights": ""}