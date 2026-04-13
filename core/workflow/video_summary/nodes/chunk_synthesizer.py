import os
import time
from typing import Dict, List

from openai import OpenAI

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


def chunk_synthesizer_node(state: VideoSummaryState) -> dict:
    chunk_plan = state.get("chunk_plan", [])
    chunk_results = state.get("chunk_results", [])
    user_prompt = state.get("user_prompt", "")

    if not isinstance(chunk_plan, list) or not chunk_plan:
        return {"chunk_results": chunk_results}
    if not isinstance(chunk_results, list):
        chunk_results = []

    result_map: Dict[str, Dict] = {
        str(item.get("chunk_id", "")): dict(item)
        for item in chunk_results
        if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()
    }

    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        started = time.perf_counter()
        item = result_map.get(chunk_id, {"chunk_id": chunk_id})
        audio_insights = str(item.get("audio_insights", ""))
        vision_insights = str(item.get("vision_insights", ""))

        try:
            chunk_summary = _llm_chunk_fusion(chunk_id, audio_insights, vision_insights, user_prompt)
        except Exception as exc:
            chunk_summary = f"[chunk={chunk_id}] 分片融合降级：{str(exc)}"

        latency_ms = int((time.perf_counter() - started) * 1000)

        item.update(
            {
                "chunk_summary": chunk_summary,
                "latency_ms": {
                    **(item.get("latency_ms", {}) if isinstance(item.get("latency_ms", {}), dict) else {}),
                    "synthesizer": latency_ms,
                },
            }
        )
        result_map[chunk_id] = item

    ordered_results: List[Dict] = []
    for chunk in chunk_plan:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in result_map:
            ordered_results.append(result_map[chunk_id])

    return {"chunk_results": ordered_results}
