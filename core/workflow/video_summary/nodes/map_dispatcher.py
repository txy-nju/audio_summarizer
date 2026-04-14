from typing import Any, Dict, List
from langgraph.types import Send

from core.workflow.video_summary.state import VideoSummaryState


def map_dispatch_node(state: VideoSummaryState) -> Dict[str, Any]:
    """
    迭代 A：构建分发元信息，不改变现有主流程分析语义。
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
            "dispatch_strategy": "send-api-audio-pilot"
            if str(state.get("concurrency_mode", "threadpool")).strip().lower() == "send_api"
            else "threadpool-node-parallel",
        }
    )

    return {
        "chunk_results": state.get("chunk_results", []),
        "chunk_retry_count": retry_count,
        "reduce_debug_info": reduce_debug_info,
    }


def route_audio_send_tasks(state: VideoSummaryState) -> List[Send]:
    """
    Send API 试点：仅对音频分支执行图级 fan-out。
    """
    concurrency_mode = str(state.get("concurrency_mode", "threadpool")).strip().lower()
    if concurrency_mode != "send_api":
        return []

    chunk_plan = state.get("chunk_plan", [])
    if not isinstance(chunk_plan, list):
        return []

    existing_results = state.get("chunk_results", [])
    existing_map: Dict[str, Dict[str, Any]] = {
        str(item.get("chunk_id", "")).strip(): dict(item)
        for item in existing_results
        if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
    }

    sends: List[Send] = []
    transcript = state.get("transcript", "")
    user_prompt = state.get("user_prompt", "")
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
                    "current_chunk": chunk,
                    "current_chunk_base_item": existing_map.get(chunk_id, {"chunk_id": chunk_id}),
                },
            )
        )

    return sends
