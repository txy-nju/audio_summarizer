from typing import Literal
from core.workflow.video_summary.state import VideoSummaryState

def route_after_review(state: VideoSummaryState) -> Literal["Approve", "Needs Revision", "Max Loops Reached"]:
    """
    条件路由函数：根据反思节点的 critique 和当前的 revision_count 决定工作流的下一步走向。
    
    - "Approve": 无明显遗漏且逻辑自洽（无 critique 或表示通过），流向 END
    - "Needs Revision": 发现图文融合生硬或有遗漏（有 critique）且循环未满 2 次，流回 fusion_drafter_node
    - "Max Loops Reached": 循环达到或超过 2 次，强制终止并输出当前最佳版本至 END
    
    :param state: VideoSummaryState
    :return: 下一节点的路由指令
    """
    revision_count = state.get("revision_count", 0)
    critique = state.get("critique", "")

    # 判断是否通过
    if not critique or critique.strip() == "":
        return "Approve"
    
    # 未通过则判断循环次数
    if revision_count < 2:
        return "Needs Revision"
    else:
        return "Max Loops Reached"