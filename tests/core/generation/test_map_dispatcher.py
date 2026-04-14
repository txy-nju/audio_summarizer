import unittest
from typing import cast

from core.workflow.video_summary.nodes.map_dispatcher import map_dispatch_node, route_audio_send_tasks
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
        self.assertEqual(result["reduce_debug_info"]["dispatch_strategy"], "threadpool-node-parallel")
        self.assertEqual(result["reduce_debug_info"]["trace_id"], "trace-1")
        # chunk_results 应原样透传
        self.assertIs(result["chunk_results"], chunk_results)

    def test_map_dispatch_marks_send_api_strategy(self):
        state = cast(
            VideoSummaryState,
            {
                "concurrency_mode": "send_api",
                "chunk_plan": [{"chunk_id": "chunk-000", "start_sec": 0, "end_sec": 120}],
                "chunk_results": [],
            },
        )
        result = map_dispatch_node(state)
        self.assertEqual(result["reduce_debug_info"]["dispatch_strategy"], "send-api-audio-pilot")

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
        self.assertEqual(result["chunk_results"], "invalid-results")

    def test_route_audio_send_tasks_only_for_send_api(self):
        state = cast(
            VideoSummaryState,
            {
                "concurrency_mode": "threadpool",
                "chunk_plan": [{"chunk_id": "chunk-000", "transcript_segment_indexes": [0]}],
                "transcript": '{"segments": [{"text": "x"}]}',
                "user_prompt": "p",
                "chunk_results": [],
            },
        )
        sends = route_audio_send_tasks(state)
        self.assertEqual(sends, [])

    def test_route_audio_send_tasks_builds_send_payload(self):
        state = cast(
            VideoSummaryState,
            {
                "concurrency_mode": "send_api",
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "transcript_segment_indexes": [0]},
                    {"chunk_id": "chunk-001", "transcript_segment_indexes": [1]},
                ],
                "transcript": '{"segments": [{"text": "x"}]}',
                "user_prompt": "focus",
                "chunk_results": [{"chunk_id": "chunk-000", "audio_insights": "old"}],
            },
        )
        sends = route_audio_send_tasks(state)
        self.assertEqual(len(sends), 2)
        # Send 对象内部字段在版本间可能变化，使用属性访问而不是 __dict__
        self.assertEqual(getattr(sends[0], "node", ""), "chunk_audio_worker_node")
        self.assertEqual(getattr(sends[1], "node", ""), "chunk_audio_worker_node")

        arg0 = getattr(sends[0], "arg", {})
        self.assertEqual(arg0.get("current_chunk", {}).get("chunk_id"), "chunk-000")
        self.assertEqual(arg0.get("user_prompt"), "focus")
        self.assertEqual(arg0.get("current_chunk_base_item", {}).get("audio_insights"), "old")


if __name__ == "__main__":
    unittest.main()
