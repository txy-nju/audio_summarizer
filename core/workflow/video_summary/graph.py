from typing import Any
from langgraph.graph import StateGraph, START, END
from core.workflow.video_summary.state import VideoSummaryState
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

def build_video_summary_graph(checkpointer: Any = None) -> Any:
    """
    基于 《多模态视频内容总结 AI 工作流架构设计书》 升级构建的 LangGraph 执行拓扑。
    彻底抛弃了脆弱的单线流转，正式升级为带有 Self-RAG 反思闭环的 Multi-Agent 架构。
    """
    # 1. 初始化 StateGraph，绑定强类型的状态模式 (Schema)
    workflow = StateGraph(VideoSummaryState) # type: ignore

    # 2. 注册智能体 Worker 节点 (Nodes)
    # 使用 type: ignore 来抑制因 LangGraph 底层复杂泛型而产生的静态推断警告
    workflow.add_node("text_analyzer_node", text_analyzer_node) # type: ignore
    workflow.add_node("vision_analyzer_node", vision_analyzer_node) # type: ignore
    workflow.add_node("fusion_drafter_node", fusion_drafter_node) # type: ignore
    
    # 注册双重防护栏评估节点 (Evaluators)
    workflow.add_node("hallucination_grader_node", hallucination_grader_node) # type: ignore
    workflow.add_node("usefulness_grader_node", usefulness_grader_node) # type: ignore

    # 3. 编排拓扑连线 (Edges)
    # 3.1 启动并发提取：START -> (text, vision)
    workflow.add_edge(START, "text_analyzer_node")
    workflow.add_edge(START, "vision_analyzer_node")

    # 3.2 汇聚 (Fan-in)：并发处理结束后统一流入 Drafter 进行时空组合
    workflow.add_edge("text_analyzer_node", "fusion_drafter_node")
    workflow.add_edge("vision_analyzer_node", "fusion_drafter_node")

    # 3.3 组装完毕后，立刻进入第一道质量防线：检查是否无中生有 (幻觉)
    workflow.add_edge("fusion_drafter_node", "hallucination_grader_node")

    # 3.4 路由分流 1 (防幻觉路由)：若有幻觉则打回重写，若无则推进到第二关
    workflow.add_conditional_edges(
        "hallucination_grader_node",
        route_after_hallucination,
        {
            ROUTE_HAS_HALLUCINATION: "fusion_drafter_node",  # 存在幻觉，驳回重组
            ROUTE_NO_HALLUCINATION: "usefulness_grader_node" # 事实正确，放行至第二关
        }
    )

    # 3.5 路由分流 2 (防偏题路由)：若偏题则打回重写，若完美则大功告成
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