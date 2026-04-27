from typing import Any, Dict, List
from langgraph.types import Send

from core.workflow.video_summary.state import VideoSummaryState


def map_dispatch_node(state: VideoSummaryState) -> Dict[str, Any]:
    """
    分发准备节点。

    地位:
    - 位于 chunk_planner_node 之后，是并行执行前的轻量级准备层。
    - 不直接产出业务洞察，而是补齐调度和观测所需的元信息。

    任务:
    - 初始化 chunk_retry_count。
    - 标记 dispatch_strategy、chunk_count 等调试字段。
    - 透传已有的 chunk_results，供后续节点继续累积结果。

    主要输入:
    - state["chunk_plan"]: 上游规划出的分片计划。
    - state["chunk_results"]: 已存在的分片结果（恢复会话或重入场景）。

    主要输出:
    - chunk_retry_count: 每个 chunk 的重试计数基座。
    - reduce_debug_info: 分发策略和规模信息。
    - chunk_results: 原样透传。
    """
    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list):
        chunk_plan = []

    retry_count = state.get("chunk_retry_count", {})
    if not isinstance(retry_count, dict):
        retry_count = {}

    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        retry_count.setdefault(chunk_id, 0)

    reduce_debug_info = state.get("reduce_debug_info", {})
    if not isinstance(reduce_debug_info, dict):
        reduce_debug_info = {}

    reduce_debug_info.update(
        {
            "dispatch_ready": True,
            "chunk_count": len(chunk_plan),
            "dispatch_strategy": "send-api-dual-pilot",
        }
    )

    return {
        "chunk_results": state.get("chunk_results", []),
        "chunk_retry_count": retry_count,
        "reduce_debug_info": reduce_debug_info,
    }


def synthesis_barrier_node(state: VideoSummaryState) -> Dict[str, Any]:
    """
    Send API 路径下的中间汇聚节点。

    地位:
    - 位于 audio/vision worker 之后，synthesis worker 之前。
    - 负责把并行分析阶段与并行融合阶段分隔开，形成显式 barrier。

    任务:
    - 检查每个 chunk 是否同时具备 audio_insights 和 vision_insights。
    - 记录 synthesis_ready 等门控状态。
    - 不做业务推理，只做条件汇聚与状态透传。

    主要输入:
    - state["chunk_plan"]
    - state["chunk_results"]

    主要输出:
    - reduce_debug_info: 汇聚完成度与是否可进入 synthesis fan-out。
    - chunk_results: 原样透传。
    """
    reduce_debug_info = state.get("reduce_debug_info", {})
    if not isinstance(reduce_debug_info, dict):
        reduce_debug_info = {}

    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list):
        chunk_plan = []
    total_chunks = len(chunk_plan)

    chunk_results = state.get("chunk_results", [])
    if not isinstance(chunk_results, list):
        chunk_results = []

    ready_chunk_ids = {
        str(item.get("chunk_id", "")).strip()
        for item in chunk_results
        if isinstance(item, dict)
        and str(item.get("chunk_id", "")).strip()
        and str(item.get("audio_insights", "")).strip()
        and str(item.get("vision_insights", "")).strip()
    }

    reduce_debug_info.update(
        {
            "synthesis_barrier_reached": True,
            "synthesis_ready_chunks": len(ready_chunk_ids),
            "synthesis_total_chunks": total_chunks,
            "synthesis_ready": total_chunks > 0 and len(ready_chunk_ids) == total_chunks,
        }
    )

    return {
        "chunk_results": chunk_results,
        "reduce_debug_info": reduce_debug_info,
    }


def route_audio_send_tasks(state: VideoSummaryState) -> List[Send]:
    """
    为音频分析阶段生成 Send API 派发任务。
    """
    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list):
        return []

    sends: List[Send] = []
    transcript = state.get("transcript", "")
    user_prompt = state.get("user_prompt", "")
    structured_global_context = state.get("structured_global_context", {})
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        sends.append(
            Send(
                "chunk_audio_worker_node",
                {
                    "transcript": transcript,
                    "user_prompt": user_prompt,
                    "structured_global_context": structured_global_context,
                    "current_chunk": chunk,
                },
            )
        )

    return sends


def route_vision_send_tasks(state: VideoSummaryState) -> List[Send]:
    """
    为视觉分析阶段生成 Send API 派发任务。
    """
    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list):
        return []

    sends: List[Send] = []
    keyframes = state.get("keyframes", [])
    keyframes_base_path = str(state.get("keyframes_base_path", ""))
    user_prompt = state.get("user_prompt", "")
    structured_global_context = state.get("structured_global_context", {})
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        sends.append(
            Send(
                "chunk_vision_worker_node",
                {
                    "keyframes": keyframes,
                    "keyframes_base_path": keyframes_base_path,
                    "user_prompt": user_prompt,
                    "structured_global_context": structured_global_context,
                    "current_chunk": chunk,
                },
            )
        )

    return sends


def route_synthesis_send_tasks(state: VideoSummaryState) -> List[Send]:
    """
    为分片融合阶段生成 Send API 派发任务。

    仅在所有 chunk 已同时具备音频和视觉洞察时触发。
    """
    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list) or not chunk_plan:
        return []

    planned_chunk_ids: List[str] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            planned_chunk_ids.append(chunk_id)

    if not planned_chunk_ids:
        return []

    chunk_results = state.get("chunk_results", [])
    if not isinstance(chunk_results, list):
        return []

    result_map: Dict[str, Dict[str, Any]] = {
        str(item.get("chunk_id", "")).strip(): dict(item)
        for item in chunk_results
        if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
    }

    # 触发时机保底：必须等待 audio_worker 和 vision_worker 整理完全部 chunk。
    all_ready = True
    for chunk_id in planned_chunk_ids:
        item = result_map.get(chunk_id, {})
        has_audio = bool(str(item.get("audio_insights", "")).strip())
        has_vision = bool(str(item.get("vision_insights", "")).strip())
        if not (has_audio and has_vision):
            all_ready = False
            break

    if not all_ready:
        return []

    sends: List[Send] = []
    user_prompt = str(state.get("user_prompt", ""))
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        base_item = result_map.get(chunk_id, {"chunk_id": chunk_id})
        # 幂等：已生成 chunk_summary 的分片不重复派发
        if str(base_item.get("chunk_summary", "")).strip():
            continue
        if not str(base_item.get("audio_insights", "")).strip():
            continue
        if not str(base_item.get("vision_insights", "")).strip():
            continue

        sends.append(
            Send(
                "chunk_synthesizer_worker_node",
                {
                    "user_prompt": user_prompt,
                    "current_synthesis_chunk": chunk,
                    "current_synthesis_base_item": base_item,
                },
            )
        )

    return sends
