from typing import Any
from langgraph.graph import StateGraph, START, END
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.planner.chunk_planner import chunk_planner_node
from core.workflow.video_summary.nodes.map_dispatcher import (
    map_dispatch_node,
    synthesis_barrier_node,
    route_audio_send_tasks,
    route_synthesis_send_tasks,
    route_vision_send_tasks,
)
from core.workflow.video_summary.nodes.chunk_audio_analyzer import chunk_audio_analyzer_node, chunk_audio_worker_node
from core.workflow.video_summary.nodes.chunk_vision_analyzer import chunk_vision_analyzer_node, chunk_vision_worker_node
from core.workflow.video_summary.nodes.chunk_synthesizer import chunk_synthesizer_node, chunk_synthesizer_worker_node
from core.workflow.video_summary.nodes.chunk_aggregator import chunk_aggregator_node
from core.workflow.video_summary.nodes.fusion_drafter import fusion_drafter_node

# 质量审查节点
from core.workflow.video_summary.nodes.hallucination_grader import hallucination_grader_node
from core.workflow.video_summary.nodes.usefulness_grader import usefulness_grader_node

# 质量审查路由常量与路由函数
from core.workflow.video_summary.edges.router import (
    route_after_hallucination, 
    route_after_usefulness,
    ROUTE_HAS_HALLUCINATION,
    ROUTE_NO_HALLUCINATION,
    ROUTE_NOT_USEFUL,
    ROUTE_USEFUL
)

CONCURRENCY_MODE_THREADPOOL = "threadpool"
CONCURRENCY_MODE_SEND_API = "send_api"


def _add_chunk_pipeline_for_threadpool(workflow: StateGraph) -> None:
    """
    当前稳定路径：图级并行 + 节点内 ThreadPoolExecutor。
    """
    workflow.add_edge(START, "chunk_planner_node")
    workflow.add_edge("chunk_planner_node", "map_dispatch_node")

    workflow.add_edge("map_dispatch_node", "chunk_audio_node")
    workflow.add_edge("map_dispatch_node", "chunk_vision_node")

    workflow.add_edge("chunk_audio_node", "chunk_synthesizer_node")
    workflow.add_edge("chunk_vision_node", "chunk_synthesizer_node")


def _add_chunk_pipeline_for_send_api_scaffold(workflow: StateGraph) -> None:
    """
    Send API 模式：音频和视觉分支采用图级 fan-out，融合阶段在 barrier 后二次分发。
    """
    workflow.add_edge(START, "chunk_planner_node")
    workflow.add_edge("chunk_planner_node", "map_dispatch_node")

    workflow.add_conditional_edges("map_dispatch_node", route_audio_send_tasks)
    workflow.add_conditional_edges("map_dispatch_node", route_vision_send_tasks)

    # 先汇聚到 barrier，再触发 synthesis fan-out。
    workflow.add_edge("chunk_audio_worker_node", "synthesis_barrier_node")
    workflow.add_edge("chunk_vision_worker_node", "synthesis_barrier_node")
    workflow.add_conditional_edges("synthesis_barrier_node", route_synthesis_send_tasks)

    workflow.add_edge("chunk_synthesizer_worker_node", "chunk_synthesizer_node")


def build_video_summary_graph(checkpointer: Any = None, concurrency_mode: str = CONCURRENCY_MODE_THREADPOOL) -> Any:
    """
    构建视频总结工作流图。
    主干流程为：分片规划 -> 音频/视觉分析 -> 分片融合 -> 聚合成文 -> 质量审查闭环。
    """
    # 1. 初始化 StateGraph，绑定状态结构
    workflow = StateGraph(VideoSummaryState) # type: ignore

    # 2. 注册节点
    workflow.add_node("chunk_planner_node", chunk_planner_node) # type: ignore
    workflow.add_node("map_dispatch_node", map_dispatch_node) # type: ignore
    workflow.add_node("synthesis_barrier_node", synthesis_barrier_node) # type: ignore
    workflow.add_node("chunk_audio_node", chunk_audio_analyzer_node) # type: ignore
    workflow.add_node("chunk_audio_worker_node", chunk_audio_worker_node) # type: ignore
    workflow.add_node("chunk_vision_node", chunk_vision_analyzer_node) # type: ignore
    workflow.add_node("chunk_vision_worker_node", chunk_vision_worker_node) # type: ignore
    workflow.add_node("chunk_synthesizer_worker_node", chunk_synthesizer_worker_node) # type: ignore
    workflow.add_node("chunk_synthesizer_node", chunk_synthesizer_node) # type: ignore
    workflow.add_node("chunk_aggregator_node", chunk_aggregator_node) # type: ignore
    workflow.add_node("fusion_drafter_node", fusion_drafter_node) # type: ignore
    
    # 注册质量审查节点
    workflow.add_node("hallucination_grader_node", hallucination_grader_node) # type: ignore
    workflow.add_node("usefulness_grader_node", usefulness_grader_node) # type: ignore

    # 3. 编排拓扑连线
    normalized_mode = (concurrency_mode or CONCURRENCY_MODE_THREADPOOL).strip().lower()
    if normalized_mode == CONCURRENCY_MODE_SEND_API:
        _add_chunk_pipeline_for_send_api_scaffold(workflow)
    else:
        _add_chunk_pipeline_for_threadpool(workflow)

    # 分片结果先聚合，再交由成文节点生成草稿
    workflow.add_edge("chunk_synthesizer_node", "chunk_aggregator_node")
    workflow.add_edge("chunk_aggregator_node", "fusion_drafter_node")

    # 草稿生成后依次经过幻觉审查和有用性审查
    workflow.add_edge("fusion_drafter_node", "hallucination_grader_node")

    # 幻觉审查：若失败则回到成文节点重写
    workflow.add_conditional_edges(
        "hallucination_grader_node",
        route_after_hallucination,
        {
            ROUTE_HAS_HALLUCINATION: "fusion_drafter_node",
            ROUTE_NO_HALLUCINATION: "usefulness_grader_node"
        }
    )

    # 有用性审查：若失败则回到成文节点重写
    workflow.add_conditional_edges(
        "usefulness_grader_node",
        route_after_usefulness,
        {
            ROUTE_NOT_USEFUL: "fusion_drafter_node",
            ROUTE_USEFUL: END
        }
    )

    # 4. 编译并返回可执行工作流
    return workflow.compile(checkpointer=checkpointer)