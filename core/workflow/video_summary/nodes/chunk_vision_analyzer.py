import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from openai import OpenAI

from config.settings import CHUNK_MAX_TOOL_CALLS, ENABLE_CHUNK_CACHE, MAP_MAX_PARALLELISM
from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.tools.search_tools import execute_tavily_search
from core.workflow.video_summary.utils.frame_utils import resolve_frame_image_base64

_VISION_SEARCH_CACHE: Dict[str, str] = {}
_VISION_SEARCH_CACHE_LOCK = threading.Lock()


FramePayload = Dict[str, Any]
ChunkResult = Dict[str, Any]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _search_with_cache(query: str) -> str:
    if ENABLE_CHUNK_CACHE:
        with _VISION_SEARCH_CACHE_LOCK:
            cached = _VISION_SEARCH_CACHE.get(query)
        if cached is not None:
            return cached

    result = execute_tavily_search(query)
    if ENABLE_CHUNK_CACHE:
        with _VISION_SEARCH_CACHE_LOCK:
            _VISION_SEARCH_CACHE[query] = result
    return result


def _llm_vision_chunk_summary(chunk_id: str, frames: List[FramePayload], user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        times = [str(frame.get("time", "未知")) for frame in frames[:8]]
        return f"[chunk={chunk_id}] 视觉摘要（降级）：命中 {len(frames)} 帧，时间点 {times}"

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"请分析该视频分片关键帧并总结主要画面变化。\\n[user_prompt] {user_prompt}\\n[chunk_id] {chunk_id}",
        }
    ]

    for frame in frames[:8]:
        time_str = str(frame.get("time", "未知"))
        image_b64 = str(frame.get("image", ""))
        content.append({"type": "text", "text": f"时间戳: {time_str}"})
        if image_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"},
                }
            )

    client = OpenAI(api_key=api_key, base_url=base_url)
    model_name = os.getenv("OPENAI_VISION_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))
    messages_payload: Any = [
        {"role": "system", "content": "你是分片视觉分析助手，请输出关键画面、动作和图表要点。"},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages_payload,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _process_single_chunk_vision(
    chunk_id: str,
    frame_indexes: List[int],
    keyframes: List[FramePayload],
    keyframes_base_path: str,
    user_prompt: str,
    base_item: ChunkResult,
) -> tuple[str, ChunkResult]:
    started = time.perf_counter()

    selected_frames: List[FramePayload] = []
    for idx in frame_indexes:
        if not isinstance(idx, int) or idx < 0 or idx >= len(keyframes):
            continue
        frame = keyframes[idx]
        if isinstance(frame, dict):
            normalized_frame = dict(frame)
            normalized_frame["image"] = resolve_frame_image_base64(frame, keyframes_base_path)
            selected_frames.append(normalized_frame)

    if not selected_frames:
        insights = f"[chunk={chunk_id}] 无可用关键帧证据。"
        searches: List[Dict[str, str]] = []
    else:
        try:
            insights = _llm_vision_chunk_summary(chunk_id, selected_frames, user_prompt)
        except Exception as exc:
            insights = f"[chunk={chunk_id}] 视觉分析降级：{str(exc)}"

        searches = []
        for frame in selected_frames[: max(0, CHUNK_MAX_TOOL_CALLS)]:
            t = str(frame.get("time", "未知"))
            query = f"video frame context at {t}"
            searches.append({"query": query, "result": _search_with_cache(query)})

    latency_ms = int((time.perf_counter() - started) * 1000)

    merged = dict(base_item)
    evidence_refs = _as_dict(merged.get("evidence_refs"))
    token_usage = _as_dict(merged.get("token_usage"))
    latency_info = _as_dict(merged.get("latency_ms"))
    merged.update(
        {
            "chunk_id": chunk_id,
            "vision_insights": insights,
            "evidence_refs": {
                **evidence_refs,
                "keyframe_indexes": frame_indexes,
                "vision_searches": searches,
            },
            "token_usage": {
                **token_usage,
                "vision": 0,
            },
            "latency_ms": {
                **latency_info,
                "vision": latency_ms,
            },
        }
    )
    return chunk_id, merged


def chunk_vision_analyzer_node(state: VideoSummaryState) -> dict:
    """
    分片视觉分析节点。

    地位:
    - 位于 chunk_plan 之后，是多模态分析链路中的视觉分支。
    - 与 chunk_audio_analyzer_node 并行执行，为后续融合提供画面证据。

    任务:
    - 读取每个 chunk 命中的关键帧。
    - 统一解析 image/frame_file，构造可送入视觉模型的输入。
    - 调用多模态 LLM 生成 vision_insights。
    - 按需补充 vision_searches、latency_ms 等元数据。

    主要输入:
    - state["chunk_plan"]: 每个 chunk 对应的 keyframe_indexes。
    - state["keyframes"] / state["keyframes_base_path"]
    - state["user_prompt"]
    - state["chunk_results"]

    主要输出:
    - chunk_results: 为每个 chunk 补充 vision_insights 和视觉侧证据元数据。
    """
    chunk_plan = state.get("chunk_plan", [])
    keyframes = state.get("keyframes", [])
    keyframes_base_path = str(state.get("keyframes_base_path", ""))
    user_prompt = state.get("user_prompt", "")

    if not isinstance(chunk_plan, list) or not chunk_plan:
        return {"chunk_results": state.get("chunk_results", [])}
    if not isinstance(keyframes, list):
        keyframes = []

    existing = state.get("chunk_results", [])
    result_map: Dict[str, ChunkResult] = {
        str(item.get("chunk_id", "")): dict(item)
        for item in existing
        if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
    }

    valid_jobs: List[tuple[str, List[int]]] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        frame_indexes = chunk.get("keyframe_indexes", [])
        if not isinstance(frame_indexes, list):
            frame_indexes = []
        valid_jobs.append((chunk_id, frame_indexes))

    max_workers = max(1, MAP_MAX_PARALLELISM)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_chunk_vision,
                chunk_id,
                frame_indexes,
                keyframes,
                keyframes_base_path,
                user_prompt,
                result_map.get(chunk_id, {"chunk_id": chunk_id}),
            )
            for chunk_id, frame_indexes in valid_jobs
        ]

        for future in as_completed(futures):
            chunk_id, merged = future.result()
            result_map[chunk_id] = merged

    ordered_results: List[ChunkResult] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in result_map:
            ordered_results.append(result_map[chunk_id])

    return {"chunk_results": ordered_results}


def chunk_vision_worker_node(state: VideoSummaryState) -> dict:
    """
    Send API 路径下的单分片视觉分析 worker。

    地位:
    - 是 chunk_vision_analyzer_node 的单分片版本，用于图级 fan-out。

    任务:
    - 仅读取 current_chunk 命中的关键帧。
    - 生成单个 chunk 的 vision_insights。

    主要输入:
    - state["current_chunk"]
    - state["current_chunk_base_item"]
    - state["keyframes"] / state["keyframes_base_path"]
    - state["user_prompt"]

    主要输出:
    - chunk_results: 长度为 1 的列表，包含当前 chunk 的视觉分析结果。
    """
    current_chunk = state.get("current_chunk", {})
    if not isinstance(current_chunk, dict):
        return {"chunk_results": []}

    chunk_id = str(current_chunk.get("chunk_id", "")).strip()
    if not chunk_id:
        return {"chunk_results": []}

    frame_indexes = current_chunk.get("keyframe_indexes", [])
    if not isinstance(frame_indexes, list):
        frame_indexes = []

    keyframes = state.get("keyframes", [])
    if not isinstance(keyframes, list):
        keyframes = []
    keyframes_base_path = str(state.get("keyframes_base_path", ""))
    user_prompt = str(state.get("user_prompt", ""))

    base_item = state.get("current_chunk_base_item", {"chunk_id": chunk_id})
    if not isinstance(base_item, dict):
        base_item = {"chunk_id": chunk_id}

    _, merged = _process_single_chunk_vision(
        chunk_id,
        frame_indexes,
        keyframes,
        keyframes_base_path,
        user_prompt,
        base_item,
    )

    return {"chunk_results": [merged]}
