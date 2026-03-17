from langgraph.graph import StateGraph, START, END
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.nodes.text_analyzer import text_analyzer_node
from core.workflow.video_summary.nodes.vision_analyzer import vision_analyzer_node
from core.workflow.video_summary.nodes.fusion_drafter import fusion_drafter_node
from core.workflow.video_summary.nodes.consistency_checker import consistency_checker_node
from core.workflow.video_summary.edges.router import route_after_review

def build_video_summary_graph():
    """
    基于 《多模态视频内容总结 AI 工作流架构设计书》 构建 LangGraph 执行拓扑。
    采用“并行双轨分析 -> 图文结合综合 -> 审查循环反馈”模式。
    """
    # 1. 初始化 StateGraph，绑定强制状态类型
    workflow = StateGraph(VideoSummaryState)

    # 2. 注册模块（Nodes）
    workflow.add_node("text_analyzer_node", text_analyzer_node)
    workflow.add_node("vision_analyzer_node", vision_analyzer_node)
    workflow.add_node("fusion_drafter_node", fusion_drafter_node)
    workflow.add_node("consistency_checker_node", consistency_checker_node)

    # 3. 构建拓扑关系（Edges）
    # 3.1 START -> 并行分发
    workflow.add_edge(START, "text_analyzer_node")
    workflow.add_edge(START, "vision_analyzer_node")

    # 3.2 汇聚并行流 -> 综合组装层
    workflow.add_edge("text_analyzer_node", "fusion_drafter_node")
    workflow.add_edge("vision_analyzer_node", "fusion_drafter_node")

    # 3.3 综合组装层 -> 审查反思层
    workflow.add_edge("fusion_drafter_node", "consistency_checker_node")

    # 3.4 反思层 -> 路由决策区（按规则循环或终止）
    workflow.add_conditional_edges(
        "consistency_checker_node",
        route_after_review,
        {
            "Approve": END,
            "Needs Revision": "fusion_drafter_node",
            "Max Loops Reached": END
        }
    )

    # 4. 编译校验图实例
    return workflow.compile()