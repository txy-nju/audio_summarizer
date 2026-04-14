import os
import time
import random
import json
from typing import Any, List, Dict, Callable, Optional
from openai import OpenAI
from core.workflow.video_summary.graph import build_video_summary_graph
from core.workflow.checkpoint_factory import create_checkpointer
from core.workflow.session import ensure_thread_id
from core.workflow.time_travel import (
    parse_timestamp_to_seconds,
    find_nearest_keyframe,
    extract_transcript_window,
)
from config.settings import (
    CHECKPOINT_BACKEND,
    CHECKPOINT_DB_URL,
    CONCURRENCY_MODE,
    resolve_concurrency_mode,
    ENABLE_METRICS_LOGGING,
    METRICS_SAMPLE_RATE,
    TEMP_FRAMES_DIR,
)
from utils.logger import setup_logger, log_metric_event
from core.workflow.video_summary.utils.frame_utils import resolve_frame_image_base64


def _build_time_travel_fallback_response(
    timestamp: str,
    frame_time: str,
    transcript_window: str,
    reason: str,
) -> str:
    return (
        "[系统降级回答]\n"
        f"- 目标时间戳: {timestamp}\n"
        f"- 最近关键帧: {frame_time}\n"
        f"- 语音证据:\n{transcript_window}\n\n"
        f"降级原因: {reason}"
    )

def summarize_video(
    transcript: str, 
    keyframes: List[Dict], 
    user_prompt: str = "请结合画面与语音，给出一个全面、客观的高质量视频总结。",
    status_callback: Optional[Callable[[str], None]] = None,
    thread_id: str = "",
    concurrency_mode: str = "",
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

    metrics_enabled = ENABLE_METRICS_LOGGING and random.random() <= METRICS_SAMPLE_RATE
    metrics_logger = setup_logger("video_summarizer.metrics")
    run_started_at = time.perf_counter()

    # 1. 获取已编译的工作流引擎（第一阶段：接入 checkpointer）
    checkpointer = create_checkpointer(CHECKPOINT_BACKEND, CHECKPOINT_DB_URL)
    resolved_mode = resolve_concurrency_mode(concurrency_mode or CONCURRENCY_MODE)
    workflow_app = build_video_summary_graph(
        checkpointer=checkpointer,
        concurrency_mode=resolved_mode,
    )
    resolved_thread_id = ensure_thread_id(thread_id)

    if status_callback:
        status_callback(f"🧵 [Session] 当前会话 thread_id: {resolved_thread_id}")
        status_callback(f"🛠️ [Concurrency] 当前并发模式: {resolved_mode}")

    if metrics_enabled:
        keyframes_estimate_bytes = 0
        try:
            keyframes_estimate_bytes = len(json.dumps(keyframes, ensure_ascii=False).encode("utf-8"))
        except Exception:
            keyframes_estimate_bytes = 0

        log_metric_event(
            metrics_logger,
            "workflow_started",
            thread_id=resolved_thread_id,
            concurrency_mode=resolved_mode,
            transcript_chars=len(transcript),
            keyframes_count=len(keyframes),
            keyframes_estimate_bytes=keyframes_estimate_bytes,
            user_prompt_chars=len(user_prompt),
        )
    
    # 2. 构造符合 VideoSummaryState 契约的初始状态
    initial_state = {
        "transcript": transcript,
        "keyframes": keyframes,
        "keyframes_base_path": str(TEMP_FRAMES_DIR),
        "concurrency_mode": resolved_mode,
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
        # 迭代 A & B：分片计划与微智能体群 (Plan Checker + Chunk Micro-Agents)
        "chunk_planner_node": "📋 [Plan Checker] 正在以时间戳为锚点，将视频逻辑分割成多个 120 秒粒度的分片任务...",
        "map_dispatch_node": "🗺️ [Dispatcher] 正在为微智能体群编排分片执行配方，准备发起并行实时处理...",
        
        # 迭代 B 并行微智能体群 (Micro-Agent Swarm)
        "chunk_audio_node": "🎧 [Chunk Audio Micro-Agent] 微线程 1&2&3 并行：正在对每个 120s 分片逐一进行语音深度梳理与查证搜索...",
        "chunk_audio_worker_node": "🎧 [Chunk Audio Send Worker] 图级 fan-out：正在处理单分片音频洞察...",
        "chunk_vision_node": "📸 [Chunk Vision Micro-Agent] 并行通道：正同步对应时间片的关键帧进行视觉特征提取与图表解析...",
        "chunk_synthesizer_node": "⚡ [Chunk Synthesizer] 并行汇聚：将分片级音视频洞察实时融合为中间层 chunk_summary...",
        
        # 全局分析层 (Global Analysis)
        "text_analyzer_node": "🧠 [Audio Agent] 并发线程：正在利用大模型深入梳理数千字的全局语音逐字稿...",
        "vision_analyzer_node": "👁️ [Vision Agent] 并发线程：正将全部关键帧特征灌入视觉模型以提取全局图文细节...",
        "fusion_drafter_node": "🧩 [Synthesizer Agent] 正在根据分片 chunk_summary 与全局洞察，融合缝合并起草最终报告...",
        "hallucination_grader_node": "⚖️ [Hallucination Guard] 启动 SSCD 时空对抗防御网，正在对草稿中的每一个数据源进行反向核查...",
        "usefulness_grader_node": "🎯 [Usefulness Guard] 正在从挑剔的 C-level 视角评估当前的总结草稿是否真正命中了您的原始痛点诉求..."
    }

    # 维护一个局部字典缓冲区，用来组装流星碎片式返回的状态片段
    current_state = initial_state.copy()
    previous_event_at = run_started_at
    node_event_counts: Dict[str, int] = {}
    
    # stream_mode="updates" 会在每一个图节点执行完毕后，将它吐出的局部更新 `dict` yield 出来
    for output in workflow_app.stream(
        initial_state,
        {"configurable": {"thread_id": resolved_thread_id}},
        stream_mode="updates"
    ):
        for node_name, state_update in output.items():
            event_now = time.perf_counter()
            since_previous_ms = int((event_now - previous_event_at) * 1000)
            node_event_counts[node_name] = node_event_counts.get(node_name, 0) + 1

            # 实时更新缓冲区状态
            current_state.update(state_update)

            if metrics_enabled:
                chunk_results = current_state.get("chunk_results", [])
                chunk_count = len(chunk_results) if isinstance(chunk_results, list) else 0
                log_metric_event(
                    metrics_logger,
                    "workflow_node_update",
                    thread_id=resolved_thread_id,
                    concurrency_mode=resolved_mode,
                    node_name=node_name,
                    node_event_count=node_event_counts[node_name],
                    since_previous_event_ms=since_previous_ms,
                    chunk_count=chunk_count,
                )
            previous_event_at = event_now
            
            # 若前端注册了回调函数，且该节点是我们关注的核心智能体，则向外抛出事件信号
            if status_callback and node_name in node_msg_map:
                msg = node_msg_map[node_name]
                
                # [微智能体群特殊播报] 分片合成完成时，动态追加汇聚确认信号
                if node_name == "chunk_synthesizer_node":
                    chunk_results = current_state.get("chunk_results", [])
                    if isinstance(chunk_results, list) and chunk_results:
                        num_chunks = len(chunk_results)
                        msg = f"{msg}\n✅ [微智能体群汇聚] 已完成 {num_chunks} 个分片的并行深度分析，成果已交付全局融合层..."
                
                # 为核心草稿生成节点动态加入轮次重写标识与分片结合上下文
                if node_name == "fusion_drafter_node":
                    rev = current_state.get("revision_count", 1)
                    chunk_results = current_state.get("chunk_results", [])
                    has_chunk_data = isinstance(chunk_results, list) and len(chunk_results) > 0
                    
                    if rev > 1:
                        msg = f"🔄 [Reflective Synthesizer] 收到来自质量检查门神的强制驳回指令，正在进行第 {rev} 次深度重组修改..."
                    elif has_chunk_data:
                        chunk_count = len(chunk_results)
                        msg = f"🧩 [Synthesizer Agent 全局融合] 正在将 {chunk_count} 个分片中的音视频融合成果按时间序列编织成完整的全景总结报告..."
                    else:
                        msg = "🧩 [Synthesizer Agent] 正在进行全局融合分析..."
                
                status_callback(msg)
                
                # [状态透传高光时刻]：若探测到防线被击穿并下达了打回重写指令，在此专门向前端界面抛出夺目的红色报警日志
                if node_name == "hallucination_grader_node" and current_state.get("hallucination_score") == "yes":
                    status_callback(f"🚨 [系统熔断] 幻觉评分器刚刚挫败了一次模型捏造事实的行为！正在带参打回重建...")
                elif node_name == "usefulness_grader_node" and current_state.get("usefulness_score") == "no":
                    status_callback(f"🚨 [系统驳回] 草稿偏离了您的核心输入诉求。已生成定点修改指令打回重做...")

    # 4. 抽取对外唯一关心的最终形态产物
    summary = current_state.get("draft_summary", "")

    if metrics_enabled:
        final_state_estimate_bytes = 0
        try:
            final_state_estimate_bytes = len(json.dumps(current_state, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            final_state_estimate_bytes = 0

        total_duration_ms = int((time.perf_counter() - run_started_at) * 1000)
        log_metric_event(
            metrics_logger,
            "workflow_finished",
            thread_id=resolved_thread_id,
            concurrency_mode=resolved_mode,
            total_duration_ms=total_duration_ms,
            node_count=len(node_event_counts),
            node_event_counts=node_event_counts,
            revision_count=current_state.get("revision_count", 0),
            summary_chars=len(summary),
            final_state_estimate_bytes=final_state_estimate_bytes,
        )

    return summary


def answer_question_at_timestamp(
    thread_id: str,
    timestamp: str,
    question: str,
    window_seconds: int = 20,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    阶段2入口：基于 checkpoint 的时间旅行追问。
    输入 thread_id + 时间戳 + 追问问题，返回带证据约束的回答。
    """
    if not question or not question.strip():
        raise ValueError("question is required")
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id is required")

    resolved_thread_id = ensure_thread_id(thread_id)
    target_seconds = parse_timestamp_to_seconds(timestamp)

    if status_callback:
        status_callback(f"🕒 [Time Travel] 正在回溯 thread_id={resolved_thread_id} 的历史状态...")

    checkpointer = create_checkpointer(CHECKPOINT_BACKEND, CHECKPOINT_DB_URL)
    checkpoint = checkpointer.get({"configurable": {"thread_id": resolved_thread_id}})

    if not checkpoint:
        return (
            f"[系统提示] 未找到 thread_id={resolved_thread_id} 的历史会话状态。"
            "请先用同一个 thread_id 跑一次完整视频总结流程。"
        )

    channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(channel_values, dict):
        return "[系统提示] 检索到的会话状态格式异常，无法执行时间旅行追问。"

    transcript = str(channel_values.get("transcript", ""))
    keyframes = channel_values.get("keyframes", [])
    keyframes_base_path = str(channel_values.get("keyframes_base_path", ""))
    draft_summary = str(channel_values.get("draft_summary", ""))
    user_prompt = str(channel_values.get("user_prompt", ""))

    if not isinstance(keyframes, list):
        keyframes = []

    nearest_frame = find_nearest_keyframe(keyframes, target_seconds)
    transcript_window = extract_transcript_window(transcript, target_seconds, window_seconds=window_seconds)

    frame_time = nearest_frame.get("time", "未知") if nearest_frame else "未命中"
    frame_image_b64 = (
        resolve_frame_image_base64(nearest_frame, keyframes_base_path) if isinstance(nearest_frame, dict) else ""
    )

    if status_callback:
        status_callback(
            f"🎯 [Time Travel] 已定位目标窗口 {timestamp} ±{window_seconds}s，最近关键帧时间戳: {frame_time}"
        )

    # 无 key 时提供可解释降级结果，避免接口报错
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        return _build_time_travel_fallback_response(
            timestamp=timestamp,
            frame_time=frame_time,
            transcript_window=transcript_window,
            reason="未配置 OPENAI_API_KEY，当前返回的是证据抽取结果。",
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    system_prompt = (
        "你是一名严谨的视频证据问答助手。"
        "你只能基于提供的时间窗证据回答，禁止超出证据臆测。"
        "若证据不足，必须明确说明不足点。"
    )

    evidence_text = (
        f"[会话ID] {resolved_thread_id}\n"
        f"[目标时间戳] {timestamp}\n"
        f"[时间窗] ±{window_seconds}s\n"
        f"[最近关键帧时间戳] {frame_time}\n"
        f"[用户原始总结侧重点] {user_prompt}\n\n"
        f"[语音证据]\n{transcript_window}\n\n"
        f"[历史总结草稿摘要]\n{draft_summary[:1500]}"
    )

    user_content: List[Dict] = [{"type": "text", "text": evidence_text + f"\n\n[追问问题]\n{question}"}]
    if frame_image_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_image_b64}", "detail": "auto"},
            }
        )

    messages_payload: Any = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_payload,
            temperature=0.2,
        )
    except Exception as exc:
        if status_callback:
            status_callback(f"⚠️ [Time Travel] OpenAI 调用异常，已降级返回证据片段: {str(exc)}")
        return _build_time_travel_fallback_response(
            timestamp=timestamp,
            frame_time=frame_time,
            transcript_window=transcript_window,
            reason=f"OpenAI API 调用异常: {str(exc)}",
        )

    answer = response.choices[0].message.content
    return answer or "[系统提示] 已完成追问，但模型未返回文本内容。"