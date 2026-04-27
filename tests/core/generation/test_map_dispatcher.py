import unittest
from typing import cast

from core.workflow.video_summary.nodes.map_dispatcher import (
    map_dispatch_node,
    route_after_wave_synthesis,
    route_audio_send_tasks,
    route_synthesis_send_tasks,
    route_vision_send_tasks,
    synthesis_barrier_node,
    ROUTE_CONTINUE_WAVE,
    ROUTE_WAVE_DONE,
)
from core.workflow.video_summary.state import VideoSummaryState


class TestMapDispatcherNode(unittest.TestCase):
    def test_map_dispatch_populates_retry_and_debug_info(self):
        chunk_results = [{"chunk_id": "chunk-000", "chunk_summary": "ok"}]
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "start_sec": 0, "end_sec": 120},
                    {"chunk_id": "chunk-001", "start_sec": 120, "end_sec": 240},
                ],
                "chunk_results": chunk_results,
                "chunk_retry_count": {"chunk-000": 3},
                "reduce_debug_info": {"trace_id": "trace-1"},
            },
        )

        result = map_dispatch_node(state)

        # 已有重试计数应保留，新分片补齐默认值
        self.assertEqual(result["chunk_retry_count"]["chunk-000"], 3)
        self.assertEqual(result["chunk_retry_count"]["chunk-001"], 0)
        self.assertTrue(result["reduce_debug_info"]["dispatch_ready"])
        self.assertEqual(result["reduce_debug_info"]["chunk_count"], 2)
        self.assertEqual(result["reduce_debug_info"]["dispatch_strategy"], "send-api-wave-pilot")
        self.assertEqual(result["reduce_debug_info"]["trace_id"], "trace-1")
        self.assertEqual(result["active_wave_chunk_ids"], ["chunk-001"])
        self.assertEqual(result["wave_index"], 0)
        # chunk_results 应原样透传
        self.assertIs(result["chunk_results"], chunk_results)

    def test_map_dispatch_marks_send_api_strategy(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000", "start_sec": 0, "end_sec": 120}],
                "chunk_results": [],
            },
        )
        result = map_dispatch_node(state)
        self.assertEqual(result["reduce_debug_info"]["dispatch_strategy"], "send-api-wave-pilot")

    def test_map_dispatch_handles_invalid_types(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": "invalid",
                "chunk_retry_count": "invalid",
                "reduce_debug_info": "invalid",
                "chunk_results": "invalid-results",
            },
        )

        result = map_dispatch_node(state)
        self.assertEqual(result["chunk_retry_count"], {})
        self.assertEqual(result["reduce_debug_info"]["chunk_count"], 0)
        self.assertTrue(result["reduce_debug_info"]["dispatch_ready"])
        self.assertEqual(result["active_wave_chunk_ids"], [])
        self.assertEqual(result["chunk_results"], [])

    def test_route_audio_send_tasks_builds_send_payload(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "transcript_segment_indexes": [0]},
                    {"chunk_id": "chunk-001", "transcript_segment_indexes": [1]},
                ],
                "transcript": '{"segments": [{"text": "x"}]}',
                "user_prompt": "focus",
                "structured_global_context": {"entities": [{"name": "OpenAI"}]},
                "active_wave_chunk_ids": ["chunk-001"],
                "chunk_results": [{"chunk_id": "chunk-000", "audio_insights": "old"}],
            },
        )
        sends = route_audio_send_tasks(state)
        self.assertEqual(len(sends), 1)
        # Send 对象内部字段在版本间可能变化，使用属性访问而不是 __dict__
        self.assertEqual(getattr(sends[0], "node", ""), "chunk_audio_worker_node")

        arg0 = getattr(sends[0], "arg", {})
        self.assertEqual(arg0.get("current_chunk", {}).get("chunk_id"), "chunk-001")
        self.assertEqual(arg0.get("user_prompt"), "focus")
        self.assertEqual(arg0.get("structured_global_context", {}).get("entities", []), [{"name": "OpenAI"}])
        self.assertNotIn("current_chunk_base_item", arg0)

    def test_route_vision_send_tasks_builds_send_payload(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "keyframe_indexes": [0]},
                    {"chunk_id": "chunk-001", "keyframe_indexes": [1]},
                ],
                "keyframes": [{"time": "00:01", "image": "x"}, {"time": "00:02", "image": "y"}],
                "keyframes_base_path": "./frames",
                "user_prompt": "focus",
                "structured_global_context": {"timeline_anchors": [{"chunk_id": "chunk-000"}]},
                "active_wave_chunk_ids": ["chunk-001"],
                "chunk_results": [{"chunk_id": "chunk-001", "vision_insights": "old-v"}],
            },
        )
        sends = route_vision_send_tasks(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(getattr(sends[0], "node", ""), "chunk_vision_worker_node")

        arg1 = getattr(sends[0], "arg", {})
        self.assertEqual(arg1.get("current_chunk", {}).get("chunk_id"), "chunk-001")
        self.assertEqual(arg1.get("keyframes_base_path"), "./frames")
        self.assertEqual(arg1.get("structured_global_context", {}).get("timeline_anchors", []), [{"chunk_id": "chunk-000"}])
        self.assertNotIn("current_chunk_base_item", arg1)

    def test_synthesis_barrier_marks_ready_only_when_all_chunks_have_audio_and_vision(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "active_wave_chunk_ids": ["c1", "c2"],
                "chunk_results": [
                    {"chunk_id": "c1", "audio_insights": "a1", "vision_insights": "v1"},
                    {"chunk_id": "c2", "audio_insights": "a2"},
                ],
                "reduce_debug_info": {},
            },
        )
        result = synthesis_barrier_node(state)
        debug_info = result.get("reduce_debug_info", {})
        self.assertTrue(debug_info.get("synthesis_barrier_reached"))
        self.assertEqual(debug_info.get("synthesis_ready_chunks"), 1)
        self.assertEqual(debug_info.get("synthesis_total_chunks"), 2)
        self.assertFalse(debug_info.get("synthesis_ready"))

    def test_synthesis_barrier_accepts_degraded_modality_as_ready(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}],
                "active_wave_chunk_ids": ["c1"],
                "chunk_results": [
                    {
                        "chunk_id": "c1",
                        "audio_insights": "<missing_context>:audio:timeout",
                        "modality_status": {"audio": "timeout", "vision": "ok"},
                        "vision_insights": "v1",
                    }
                ],
                "reduce_debug_info": {},
            },
        )
        result = synthesis_barrier_node(state)
        debug_info = result.get("reduce_debug_info", {})
        self.assertTrue(debug_info.get("synthesis_ready"))

    def test_route_synthesis_send_tasks_waits_until_all_chunks_ready(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "user_prompt": "focus",
                "active_wave_chunk_ids": ["c1", "c2"],
                "chunk_results": [
                    {"chunk_id": "c1", "audio_insights": "a1", "vision_insights": "v1"},
                    {"chunk_id": "c2", "audio_insights": "a2"},
                ],
            },
        )
        sends = route_synthesis_send_tasks(state)
        self.assertEqual(sends, [])

    def test_route_synthesis_send_tasks_builds_payload_when_all_chunks_ready(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "user_prompt": "focus",
                "active_wave_chunk_ids": ["c1", "c2"],
                "chunk_results": [
                    {"chunk_id": "c1", "audio_insights": "a1", "vision_insights": "v1"},
                    {
                        "chunk_id": "c2",
                        "audio_insights": "a2",
                        "vision_insights": "v2",
                        "chunk_summary": "already_done",
                    },
                ],
            },
        )
        sends = route_synthesis_send_tasks(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(getattr(sends[0], "node", ""), "chunk_synthesizer_worker_node")
        arg0 = getattr(sends[0], "arg", {})
        self.assertEqual(arg0.get("current_synthesis_chunk", {}).get("chunk_id"), "c1")
        self.assertEqual(arg0.get("current_synthesis_base_item", {}).get("chunk_id"), "c1")

    def test_route_after_wave_synthesis_continue_when_pending(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "chunk_results": [{"chunk_id": "c1", "chunk_summary": "done"}],
            },
        )
        self.assertEqual(route_after_wave_synthesis(state), ROUTE_CONTINUE_WAVE)

    def test_route_after_wave_synthesis_done_when_all_synthesized(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "chunk_results": [
                    {"chunk_id": "c1", "chunk_summary": "done-1"},
                    {"chunk_id": "c2", "chunk_summary": "done-2"},
                ],
            },
        )
        self.assertEqual(route_after_wave_synthesis(state), ROUTE_WAVE_DONE)

    def test_route_after_wave_synthesis_done_when_synthesizer_terminal_without_summary(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "c1"}],
                "chunk_results": [
                    {
                        "chunk_id": "c1",
                        "modality_status": {"synthesizer": "failed"},
                        "chunk_summary": "",
                    }
                ],
            },
        )
        self.assertEqual(route_after_wave_synthesis(state), ROUTE_WAVE_DONE)


if __name__ == "__main__":
    unittest.main()
