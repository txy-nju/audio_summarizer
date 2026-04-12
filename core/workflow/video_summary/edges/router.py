from typing import Literal
from core.workflow.video_summary.state import VideoSummaryState

def route_after_hallucination(state: VideoSummaryState) -> Literal["Has Hallucination", "No Hallucination"]:
    """
    [Self-RAG 第一道防线路由]：
    根据幻觉评分器 (hallucination_grader) 的输出决定流向。
    若存在幻觉 -> 回流起草节点重写 (Needs Revision)。
    若无幻觉 -> 流向下一道防线：有用性评分器。
    
    :param state: VideoSummaryState
    :return: 下一步路由指令
    """
    score = state.get("hallucination_score", "no").lower()
    
    if score == "yes":
        return "Has Hallucination"
    else:
        return "No Hallucination"

def route_after_usefulness(state: VideoSummaryState) -> Literal["Not Useful", "Useful"]:
    """
    [Self-RAG 第二道防线路由]：
    根据有用性评分器 (usefulness_grader) 的输出决定流向。
    若偏离需求 -> 回流起草节点重写 (Needs Revision)。
    若满足需求 -> 大功告成，流向终点 END。
    
    :param state: VideoSummaryState
    :return: 下一步路由指令
    """
    score = state.get("usefulness_score", "yes").lower()
    
    if score == "no":
        return "Not Useful"
    else:
        return "Useful"