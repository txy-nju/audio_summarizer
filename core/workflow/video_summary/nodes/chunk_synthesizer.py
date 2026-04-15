import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config.settings import MAP_MAX_PARALLELISM
from core.workflow.video_summary.state import VideoSummaryState


def _llm_chunk_fusion(chunk_id: str, audio_insights: str, vision_insights: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        return (
            f"[chunk={chunk_id}] 分片融合（降级）\\n"
            f"- Audio: {audio_insights[:200]}\\n"
            f"- Vision: {vision_insights[:200]}"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是分片融合助手，请将音频洞察与视觉洞察融合为该时间片的简洁总结。",
            },
            {
                "role": "user",
                "content": (
                    f"[chunk_id]\\n{chunk_id}\\n\\n"
                    f"[user_prompt]\\n{user_prompt}\\n\\n"
                    f"[audio_insights]\\n{audio_insights}\\n\\n"
                    f"[vision_insights]\\n{vision_insights}"
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _process_single_chunk_synthesis(
    chunk_id: str,
    user_prompt: str,
    base_item: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    started = time.perf_counter()
    item = dict(base_item)
    audio_insights = str(item.get("audio_insights", ""))
    vision_insights = str(item.get("vision_insights", ""))

    try:
        chunk_summary = _llm_chunk_fusion(chunk_id, audio_insights, vision_insights, user_prompt)
    except Exception as exc:
        chunk_summary = f"[chunk={chunk_id}] 分片融合降级：{str(exc)}"

    latency_ms = int((time.perf_counter() - started) * 1000)
    latency_info = _as_dict(item.get("latency_ms"))
    item.update(
        {
            "chunk_id": chunk_id,
            "chunk_summary": chunk_summary,
            "latency_ms": {
                **latency_info,
                "synthesizer": latency_ms,
            },
        }
    )
    return chunk_id, item


def chunk_synthesizer_node(state: VideoSummaryState) -> dict:
    """
    分片融合节点。

    地位:
    - 位于音频分支和视觉分支之后，是分片级多模态汇合点。
    - 输出的 chunk_summary 会被后续 aggregator 作为更高层的证据摘要使用。

    任务:
    - 将同一 chunk 的 audio_insights 与 vision_insights 融合成 chunk_summary。
    - 在 threadpool 模式下并行处理全部 chunk。
    - 在 send_api 模式下仅负责顺序重建和结果透传，避免重复计算。

    主要输入:
    - state["chunk_plan"]
    - state["chunk_results"]
    - state["user_prompt"]
    - state["concurrency_mode"]

    主要输出:
    - chunk_results: 为每个 chunk 补充 chunk_summary 和 synthesizer 延迟信息。
    """
    concurrency_mode = str(state.get("concurrency_mode", "threadpool")).strip().lower()
    if concurrency_mode == "send_api":
        # send_api 路径下，分片融合由 chunk_synthesizer_worker_node 完成。
        # 这里仅做顺序重建与透传，避免重复线程池计算。
        chunk_plan = state.get("chunk_plan", [])
        chunk_results = state.get("chunk_results", [])
        if not isinstance(chunk_plan, list) or not isinstance(chunk_results, list):
            return {"chunk_results": chunk_results if isinstance(chunk_results, list) else []}

        result_map: Dict[str, Dict[str, Any]] = {
            str(item.get("chunk_id", "")).strip(): dict(item)
            for item in chunk_results
            if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
        }

        ordered_results: List[Dict[str, Any]] = []
        for chunk in chunk_plan:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if chunk_id and chunk_id in result_map:
                ordered_results.append(result_map[chunk_id])
        return {"chunk_results": ordered_results}

    chunk_plan = state.get("chunk_plan", [])
    chunk_results = state.get("chunk_results", [])
    user_prompt = state.get("user_prompt", "")

    if not isinstance(chunk_plan, list) or not chunk_plan:
        return {"chunk_results": chunk_results}
    if not isinstance(chunk_results, list):
        chunk_results = []

    result_map: Dict[str, Dict[str, Any]] = {
        str(item.get("chunk_id", "")): dict(item)
        for item in chunk_results
        if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
    }

    valid_chunk_ids: List[str] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        valid_chunk_ids.append(chunk_id)

    max_workers = max(1, MAP_MAX_PARALLELISM)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_chunk_synthesis,
                chunk_id,
                user_prompt,
                result_map.get(chunk_id, {"chunk_id": chunk_id}),
            )
            for chunk_id in valid_chunk_ids
        ]

        for future in as_completed(futures):
            chunk_id, item = future.result()
            result_map[chunk_id] = item

    ordered_results: List[Dict[str, Any]] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in result_map:
            ordered_results.append(result_map[chunk_id])

    return {"chunk_results": ordered_results}


def chunk_synthesizer_worker_node(state: VideoSummaryState) -> dict:
    """
    Send API 路径下的单分片融合 worker。

    地位:
    - 是 chunk_synthesizer_node 的单分片版本，用于图级 fan-out。

    任务:
    - 读取单个 chunk 已完成的音频与视觉洞察。
    - 生成当前 chunk 的 chunk_summary。

    主要输入:
    - state["current_synthesis_chunk"]
    - state["current_synthesis_base_item"]
    - state["user_prompt"]

    主要输出:
    - chunk_results: 长度为 1 的列表，包含当前 chunk 的融合结果。
    """
    current_chunk = state.get("current_synthesis_chunk", {})
    if not isinstance(current_chunk, dict):
        return {"chunk_results": []}

    chunk_id = str(current_chunk.get("chunk_id", "")).strip()
    if not chunk_id:
        return {"chunk_results": []}

    user_prompt = str(state.get("user_prompt", ""))
    base_item = state.get("current_synthesis_base_item", {"chunk_id": chunk_id})
    if not isinstance(base_item, dict):
        base_item = {"chunk_id": chunk_id}

    _, merged = _process_single_chunk_synthesis(
        chunk_id,
        user_prompt,
        base_item,
    )
    return {"chunk_results": [merged]}
