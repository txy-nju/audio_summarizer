from core.workflow.video_summary.state import VideoSummaryState


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def human_gate_node(state: VideoSummaryState) -> dict:
    """
    人类审批节点（必选）。

    语义：
    - phase-1（默认）: 未提交审批输入时，标记 pending，流程在此结束并返回可编辑聚合稿。
    - phase-2（提交审批后）: human_gate_status=approved 时放行到 fusion_drafter_node。
    """
    aggregated = _safe_str(state.get("aggregated_chunk_insights", ""))
    edited = _safe_str(state.get("human_edited_aggregated_insights", ""))
    guidance = _safe_str(state.get("human_guidance", ""))
    gate_status = _safe_str(state.get("human_gate_status", "")).lower()

    if gate_status != "approved":
        gate_status = "pending"

    if not edited:
        edited = aggregated

    reason = "human_review_required" if gate_status == "pending" else "approved"
    if gate_status == "approved" and not guidance:
        reason = "approved_with_empty_guidance"

    return {
        "human_gate_status": gate_status,
        "human_gate_reason": reason,
        "human_edited_aggregated_insights": edited,
        "human_guidance": guidance,
    }
