import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_synthesizer import chunk_synthesizer_node
from core.workflow.video_summary.state import VideoSummaryState


class TestChunkSynthesizerNode(unittest.TestCase):
    def test_generates_chunk_summary_for_all_chunks_without_api_key(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}, {"chunk_id": "chunk-001"}],
                "chunk_results": [
                    {"chunk_id": "chunk-000", "audio_insights": "a0", "vision_insights": "v0"},
                    {"chunk_id": "chunk-001", "audio_insights": "a1", "vision_insights": "v1"},
                ],
                "user_prompt": "关注流程",
            },
        )

        with patch.dict("os.environ", {}, clear=True):
            result = chunk_synthesizer_node(state)

        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 2)
        self.assertEqual(
            result["chunk_results"][0]["chunk_summary"],
            "[chunk=chunk-000] 分片融合（降级）\\n- Audio: a0\\n- Vision: v0",
        )
        self.assertEqual(
            result["chunk_results"][1]["chunk_summary"],
            "[chunk=chunk-001] 分片融合（降级）\\n- Audio: a1\\n- Vision: v1",
        )
        self.assertIn("latency_ms", result["chunk_results"][0])
        self.assertIn("synthesizer", result["chunk_results"][0]["latency_ms"])

    def test_handles_empty_chunk_plan(self):
        state = cast(VideoSummaryState, {"chunk_plan": [], "chunk_results": [{"chunk_id": "x"}]})
        result = chunk_synthesizer_node(state)
        self.assertEqual(result["chunk_results"], [{"chunk_id": "x"}])

    @patch("core.workflow.video_summary.nodes.chunk_synthesizer.OpenAI")
    def test_generates_chunk_summary_with_successful_llm_response(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="主干融合摘要"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}],
                "chunk_results": [
                    {
                        "chunk_id": "chunk-000",
                        "audio_insights": "audio data",
                        "vision_insights": "vision data",
                        "latency_ms": {"audio": 1},
                    }
                ],
                "user_prompt": "关注流程",
            },
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = chunk_synthesizer_node(state)

        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["chunk_summary"], "主干融合摘要")
        self.assertEqual(result["chunk_results"][0]["latency_ms"]["audio"], 1)
        self.assertIn("synthesizer", result["chunk_results"][0]["latency_ms"])
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
