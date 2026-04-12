from typing import List, Dict, Callable, Optional
from core.workflow.video_summary.graph import build_video_summary_graph
from core.workflow.checkpoint_factory import create_checkpointer
from core.workflow.session import ensure_thread_id
from config.settings import CHECKPOINT_BACKEND, CHECKPOINT_DB_URL

def summarize_video(
    transcript: str, 
    keyframes: List[Dict], 
    user_prompt: str = "请结合画面与语音，给出一个全面、客观的高质量视频总结。",
    status_callback: Optional[Callable[[str], None]] = None,
    thread_id: str = ""
) -> str:
    """
    外部调用接口：启动多模态视频总结工作流。
    该接口严格遵守架构契约的分层设计，对外屏蔽了 State 和 Graph 的复杂性。
    引入了基于 stream() 的流式状态透传，用于向前端 UI 进行全链路极客风的进度流水播报。
    
    :param transcript: 视频语音识别出的完整文本
    :param keyframes: 视频关键帧列表，格式必须为 [{"time": "...", "image": "..."}]
    :param user_prompt: 用户的特殊总结侧重点要求
    :param status_callback: 允许回调注入状态字符串以供前端展示
    :return: 最终生成的总结内容 (即工作流末态的 draft_summary)
    """
    if status_callback:
        status_callback("⚙️ [LangGraph 初始化] 正在编排多智能体认知状态机网络...")

    # 1. 获取已编译的工作流引擎（第一阶段：接入 checkpointer）
    checkpointer = create_checkpointer(CHECKPOINT_BACKEND, CHECKPOINT_DB_URL)
    workflow_app = build_video_summary_graph(checkpointer=checkpointer)
    resolved_thread_id = ensure_thread_id(thread_id)

    if status_callback:
        status_callback(f"🧵 [Session] 当前会话 thread_id: {resolved_thread_id}")
    
    # 2. 构造符合 VideoSummaryState 契约的初始状态
    initial_state = {
        "transcript": transcript,
        "keyframes": keyframes,
        "user_prompt": user_prompt,
        "text_insights": "",
        "visual_insights": "",
        "draft_summary": "",
        "hallucination_score": "",
        "usefulness_score": "",
        "feedback_instructions": "",
        "revision_count": 0
    }
    
    # 3. [架构级革命] 将阻塞式的 invoke 升级为分布式的 stream，并截获中间局部全量状态
    if status_callback:
        status_callback("⚡ [引擎点火] 工作流正式启动，并行并发拆解多模态任务中...")

    # Node 名称与前端友好播报信息的极客风映射表
    node_msg_map = {
        "text_analyzer_node": "🧠 [Audio Agent] 并发线程：正在利用大模型深入梳理数千字的语音逐字稿...",
        "vision_analyzer_node": "👁️ [Vision Agent] 并发线程：正将数百张关键帧特征灌入视觉模型以提取图文细节...",
        "fusion_drafter_node": "🧩 [Synthesizer Agent] 正在根据精准的时空锚点，将音视频特征融合缝合并起草首版报告...",
        "hallucination_grader_node": "⚖️ [Hallucination Guard] 启动 SSCD 时空对抗防御网，正在对草稿中的每一个数据源进行反向核查...",
        "usefulness_grader_node": "🎯 [Usefulness Guard] 正在从挑剔的 C-level 视角评估当前的总结草稿是否真正命中了您的原始痛点诉求..."
    }

    # 维护一个局部字典缓冲区，用来组装流星碎片式返回的状态片段
    current_state = initial_state.copy()
    
    # stream_mode="updates" 会在每一个图节点执行完毕后，将它吐出的局部更新 `dict` yield 出来
    for output in workflow_app.stream(
        initial_state,
        {"configurable": {"thread_id": resolved_thread_id}},
        stream_mode="updates"
    ):
        for node_name, state_update in output.items():
            # 实时更新缓冲区状态
            current_state.update(state_update)
            
            # 若前端注册了回调函数，且该节点是我们关注的核心智能体，则向外抛出事件信号
            if status_callback and node_name in node_msg_map:
                msg = node_msg_map[node_name]
                
                # 为核心草稿生成节点动态加入轮次重写标识
                if node_name == "fusion_drafter_node":
                    rev = current_state.get("revision_count", 1)
                    if rev > 1:
                        msg = f"🔄 [Reflective Synthesizer] 收到来自质量检查门神的强制驳回指令，正在进行第 {rev} 次深度重组修改..."
                
                status_callback(msg)
                
                # [状态透传高光时刻]：若探测到防线被击穿并下达了打回重写指令，在此专门向前端界面抛出夺目的红色报警日志
                if node_name == "hallucination_grader_node" and current_state.get("hallucination_score") == "yes":
                    status_callback(f"🚨 [系统熔断] 幻觉评分器刚刚挫败了一次模型捏造事实的行为！正在带参打回重建...")
                elif node_name == "usefulness_grader_node" and current_state.get("usefulness_score") == "no":
                    status_callback(f"🚨 [系统驳回] 草稿偏离了您的核心输入诉求。已生成定点修改指令打回重做...")

    # 4. 抽取对外唯一关心的最终形态产物
    return current_state.get("draft_summary", "")