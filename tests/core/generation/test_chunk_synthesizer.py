import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_synthesizer import chunk_synthesizer_node, chunk_synthesizer_worker_node
from core.workflow.video_summary.state import VideoSummaryState


class TestChunkSynthesizerNode(unittest.TestCase):
    def test_reorders_existing_chunk_results_by_chunk_plan(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}, {"chunk_id": "chunk-001"}],
                "chunk_results": [
                    {"chunk_id": "chunk-001", "audio_insights": "a1", "vision_insights": "v1", "chunk_summary": "s1"},
                    {"chunk_id": "chunk-000", "audio_insights": "a0", "vision_insights": "v0", "chunk_summary": "s0"},
                ],
                "user_prompt": "关注流程",
            },
        )

        result = chunk_synthesizer_node(state)

        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 2)
        self.assertEqual(result["chunk_results"][0]["chunk_id"], "chunk-000")
        self.assertEqual(result["chunk_results"][1]["chunk_id"], "chunk-001")

    def test_handles_empty_chunk_plan(self):
        state = cast(VideoSummaryState, {"chunk_plan": [], "chunk_results": [{"chunk_id": "x"}]})
        result = chunk_synthesizer_node(state)
        self.assertEqual(result["chunk_results"], [])

    def test_ignores_chunks_not_in_plan(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}],
                "chunk_results": [
                    {
                        "chunk_id": "chunk-000",
                        "chunk_summary": "ok",
                    },
                    {"chunk_id": "chunk-999", "chunk_summary": "ignored"},
                ],
                "user_prompt": "关注流程",
            },
        )

        result = chunk_synthesizer_node(state)
        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["chunk_id"], "chunk-000")

    @patch("core.workflow.video_summary.nodes.chunk_synthesizer._process_single_chunk_synthesis")
    def test_chunk_synthesizer_worker_node_handles_single_chunk(self, mock_process):
        mock_process.return_value = (
            "chunk-009",
            {
                "chunk_id": "chunk-009",
                "chunk_summary": "synth-summary",
                "latency_ms": {"synthesizer": 10},
            },
        )

        state = cast(
            VideoSummaryState,
            {
                "current_synthesis_chunk": {"chunk_id": "chunk-009"},
                "current_synthesis_base_item": {
                    "chunk_id": "chunk-009",
                    "audio_insights": "a",
                    "vision_insights": "v",
                },
                "user_prompt": "focus",
            },
        )

        result = chunk_synthesizer_worker_node(state)
        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["chunk_summary"], "synth-summary")
        mock_process.assert_called_once()

    def test_chunk_synthesizer_worker_node_returns_empty_when_chunk_missing(self):
        state = cast(VideoSummaryState, {"current_synthesis_chunk": {}})
        result = chunk_synthesizer_worker_node(state)
        self.assertEqual(result["chunk_results"], [])

    @patch("core.workflow.video_summary.nodes.chunk_synthesizer._llm_chunk_fusion", side_effect=TimeoutError("timeout"))
    def test_chunk_synthesizer_worker_timeout_marks_terminal_degraded(self, _mock_fusion):
        state = cast(
            VideoSummaryState,
            {
                "current_synthesis_chunk": {"chunk_id": "chunk-timeout"},
                "current_synthesis_base_item": {
                    "chunk_id": "chunk-timeout",
                    "audio_insights": "a",
                    "vision_insights": "v",
                },
                "user_prompt": "focus",
            },
        )

        result = chunk_synthesizer_worker_node(state)
        item = result["chunk_results"][0]
        self.assertEqual(item.get("modality_status", {}).get("synthesizer"), "timeout")
        self.assertIn("<missing_context>", str(item.get("chunk_summary", "")))
        self.assertTrue(item.get("degraded_context", {}).get("synthesizer"))


if __name__ == "__main__":
    unittest.main()
