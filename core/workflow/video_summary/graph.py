from typing import Any
from langgraph.graph import StateGraph, START, END
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.planner.chunk_planner import chunk_planner_node
from core.workflow.video_summary.nodes.map_dispatcher import map_dispatch_node, route_audio_send_tasks
from core.workflow.video_summary.nodes.chunk_audio_analyzer import chunk_audio_analyzer_node, chunk_audio_worker_node
from core.workflow.video_summary.nodes.chunk_vision_analyzer import chunk_vision_analyzer_node
from core.workflow.video_summary.nodes.chunk_synthesizer import chunk_synthesizer_node
from core.workflow.video_summary.nodes.text_analyzer import text_analyzer_node
from core.workflow.video_summary.nodes.vision_analyzer import vision_analyzer_node
from core.workflow.video_summary.nodes.fusion_drafter import fusion_drafter_node

# [Self-RAG 架构升级] 导入新拆分的双重独立审查节点
from core.workflow.video_summary.nodes.hallucination_grader import hallucination_grader_node
from core.workflow.video_summary.nodes.usefulness_grader import usefulness_grader_node

# [Self-RAG 架构升级] 导入新的多级路控体系与消除魔法字符串的常量
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
    方案B阶段4试点：音频分支切换为 Send API fan-out，视觉分支保留现状。
    """
    workflow.add_edge(START, "chunk_planner_node")
    workflow.add_edge("chunk_planner_node", "map_dispatch_node")

    workflow.add_conditional_edges("map_dispatch_node", route_audio_send_tasks)
    workflow.add_edge("map_dispatch_node", "chunk_vision_node")

    workflow.add_edge("chunk_audio_worker_node", "chunk_synthesizer_node")
    workflow.add_edge("chunk_vision_node", "chunk_synthesizer_node")


def build_video_summary_graph(checkpointer: Any = None, concurrency_mode: str = CONCURRENCY_MODE_THREADPOOL) -> Any:
    """
    基于 《多模态视频内容总结 AI 工作流架构设计书》 升级构建的 LangGraph 执行拓扑。
    彻底抛弃了脆弱的单线流转，正式升级为带有 Self-RAG 反思闭环的 Multi-Agent 架构。
    """
    # 1. 初始化 StateGraph，绑定强类型的状态模式 (Schema)
    workflow = StateGraph(VideoSummaryState) # type: ignore

    # 2. 注册智能体 Worker 节点 (Nodes)
    # 使用 type: ignore 来抑制因 LangGraph 底层复杂泛型而产生的静态推断警告
    workflow.add_node("chunk_planner_node", chunk_planner_node) # type: ignore
    workflow.add_node("map_dispatch_node", map_dispatch_node) # type: ignore
    workflow.add_node("chunk_audio_node", chunk_audio_analyzer_node) # type: ignore
    workflow.add_node("chunk_audio_worker_node", chunk_audio_worker_node) # type: ignore
    workflow.add_node("chunk_vision_node", chunk_vision_analyzer_node) # type: ignore
    workflow.add_node("chunk_synthesizer_node", chunk_synthesizer_node) # type: ignore
    workflow.add_node("text_analyzer_node", text_analyzer_node) # type: ignore
    workflow.add_node("vision_analyzer_node", vision_analyzer_node) # type: ignore
    workflow.add_node("fusion_drafter_node", fusion_drafter_node) # type: ignore
    
    # 注册双重防护栏评估节点 (Evaluators)
    workflow.add_node("hallucination_grader_node", hallucination_grader_node) # type: ignore
    workflow.add_node("usefulness_grader_node", usefulness_grader_node) # type: ignore

    # 3. 编排拓扑连线 (Edges)
    # 3.1 方案B阶段1：根据并发模式选择图骨架（默认 threadpool）
    normalized_mode = (concurrency_mode or CONCURRENCY_MODE_THREADPOOL).strip().lower()
    if normalized_mode == CONCURRENCY_MODE_SEND_API:
        _add_chunk_pipeline_for_send_api_scaffold(workflow)
    else:
        _add_chunk_pipeline_for_threadpool(workflow)

    # 3.3 迭代 C 预留：全局分析与融合阶段开始
    # chunk_synthesizer 汇聚了并行音视频分片分析的成果，准备进入全局分析链路
    workflow.add_edge("chunk_synthesizer_node", "text_analyzer_node")
    workflow.add_edge("chunk_synthesizer_node", "vision_analyzer_node")

    # 3.4 汇聚 (Fan-in)：并发处理结束后统一流入 Drafter 进行时空组合
    workflow.add_edge("text_analyzer_node", "fusion_drafter_node")
    workflow.add_edge("vision_analyzer_node", "fusion_drafter_node")

    # 3.5 组装完毕后，立刻进入第一道质量防线：检查是否无中生有 (幻觉)
    workflow.add_edge("fusion_drafter_node", "hallucination_grader_node")

    # 3.6 路由分流 1 (防幻觉路由)：若有幻觉则打回重写，若无则推进到第二关
    workflow.add_conditional_edges(
        "hallucination_grader_node",
        route_after_hallucination,
        {
            ROUTE_HAS_HALLUCINATION: "fusion_drafter_node",  # 存在幻觉，驳回重组
            ROUTE_NO_HALLUCINATION: "usefulness_grader_node" # 事实正确，放行至第二关
        }
    )

    # 3.7 路由分流 2 (防偏题路由)：若偏题则打回重写，若完美则大功告成
    workflow.add_conditional_edges(
        "usefulness_grader_node",
        route_after_usefulness,
        {
            ROUTE_NOT_USEFUL: "fusion_drafter_node",  # 偏离用户需求，驳回重组
            ROUTE_USEFUL: END                         # 满足一切高压线，正式交付！
        }
    )

    # 4. 编译校验图实例并返回可执行工作流
    # compile() 返回的是 CompiledStateGraph，具有 .invoke 和 .stream 方法
    return workflow.compile(checkpointer=checkpointer)