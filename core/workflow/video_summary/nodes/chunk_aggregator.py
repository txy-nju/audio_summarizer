from typing import Any, Dict, List

from config.settings import AGGREGATED_CHUNK_INSIGHTS_MAX_CHARS
from core.workflow.video_summary.state import VideoSummaryState


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _to_hhmmss(total_seconds: int) -> str:
    safe = max(0, int(total_seconds))
    hours = safe // 3600
    minutes = (safe % 3600) // 60
    seconds = safe % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _truncate(text: str, limit: int) -> str:
    safe_limit = max(2000, int(limit))
    if len(text) <= safe_limit:
        return text
    suffix = "\n\n[系统提示] 聚合内容超过上限，已自动截断以保护后续节点上下文窗口。"
    return text[: safe_limit - len(suffix)] + suffix


def chunk_aggregator_node(state: VideoSummaryState) -> dict:
    """
    分片聚合节点。

    地位:
    - 位于分片融合之后、全局成文之前，是从“分片级证据”过渡到“全局草稿”的桥梁。

    任务:
    - 按 chunk_plan 顺序整理 chunk_results。
    - 将 chunk_summary、audio_insights、vision_insights 聚合为单段 Markdown 文本。
    - 对过长的聚合文本执行截断，保护后续节点上下文窗口。
    - 记录 dropped_chunks 等聚合阶段元数据。

    主要输入:
    - state["chunk_plan"]
    - state["chunk_results"]
    - state["user_prompt"]

    主要输出:
    - aggregated_chunk_insights: 供 fusion_drafter_node 消费的全局证据底稿。
    - reduce_debug_info: 聚合阶段统计信息。
    """
    chunk_results = state.get("chunk_results", [])
    chunk_plan = state.get("chunk_plan", [])
    user_prompt = _safe_str(state.get("user_prompt", ""))

    if not isinstance(chunk_results, list):
        chunk_results = []
    if not isinstance(chunk_plan, list):
        chunk_plan = []

    result_map: Dict[str, Dict[str, Any]] = {}
    for item in chunk_results:
        if not isinstance(item, dict):
            continue
        chunk_id = _safe_str(item.get("chunk_id", ""))
        if not chunk_id:
            continue
        result_map[chunk_id] = item

    plan_map: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for plan_item in chunk_plan:
        if not isinstance(plan_item, dict):
            continue
        chunk_id = _safe_str(plan_item.get("chunk_id", ""))
        if not chunk_id:
            continue
        plan_map[chunk_id] = plan_item
        ordered_ids.append(chunk_id)

    for chunk_id in sorted(result_map.keys()):
        if chunk_id not in plan_map:
            ordered_ids.append(chunk_id)

    lines: List[str] = []
    lines.append("# Chunk Aggregated Insights")
    lines.append("")
    lines.append(f"- total_chunks: {len(ordered_ids)}")
    if user_prompt:
        lines.append(f"- user_focus: {user_prompt}")
    lines.append("")

    dropped_count = 0
    for chunk_id in ordered_ids:
        item = result_map.get(chunk_id, {})
        plan_item = plan_map.get(chunk_id, {})

        start_sec = int(plan_item.get("start_sec", 0)) if isinstance(plan_item.get("start_sec", 0), (int, float)) else 0
        end_sec = int(plan_item.get("end_sec", 0)) if isinstance(plan_item.get("end_sec", 0), (int, float)) else 0
        time_span = f"[{_to_hhmmss(start_sec)} - {_to_hhmmss(end_sec)}]"

        audio_insights = _safe_str(item.get("audio_insights", ""))
        vision_insights = _safe_str(item.get("vision_insights", ""))
        chunk_summary = _safe_str(item.get("chunk_summary", ""))

        if not audio_insights and not vision_insights and not chunk_summary:
            dropped_count += 1
            continue

        lines.append(f"## {chunk_id} {time_span}")
        if chunk_summary:
            lines.append("### chunk_summary")
            lines.append(chunk_summary)
        if audio_insights:
            lines.append("### audio_insights")
            lines.append(audio_insights)
        if vision_insights:
            lines.append("### vision_insights")
            lines.append(vision_insights)
        lines.append("")

    aggregated = _truncate("\n".join(lines).strip(), AGGREGATED_CHUNK_INSIGHTS_MAX_CHARS)
    return {
        "aggregated_chunk_insights": aggregated,
        "reduce_debug_info": {
            **(state.get("reduce_debug_info", {}) if isinstance(state.get("reduce_debug_info", {}), dict) else {}),
            "aggregator_total_chunks": len(ordered_ids),
            "aggregator_dropped_chunks": dropped_count,
        },
    }
