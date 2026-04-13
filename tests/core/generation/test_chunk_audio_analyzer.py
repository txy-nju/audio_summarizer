import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_audio_analyzer import chunk_audio_analyzer_node
from core.workflow.video_summary.state import VideoSummaryState


class TestChunkAudioAnalyzerNode(unittest.TestCase):
    @patch.dict("core.workflow.video_summary.nodes.chunk_audio_analyzer._AUDIO_SEARCH_CACHE", {}, clear=True)
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.execute_tavily_search")
    def test_builds_audio_insights_for_each_chunk_without_api_key(self, mock_search):
        mock_search.side_effect = lambda query: f"search:{query}"

        state = cast(
            VideoSummaryState,
            {
                "transcript": '{"segments": [{"start": 0, "end": 8, "text": "CPU and GPU benchmark"}, {"start": 65, "end": 75, "text": "API design details"}]}',
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "transcript_segment_indexes": [0]},
                    {"chunk_id": "chunk-001", "transcript_segment_indexes": [1]},
                ],
                "chunk_results": [],
                "user_prompt": "关注技术要点",
            },
        )

        with patch.dict("os.environ", {}, clear=True):
            with patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.CHUNK_MAX_TOOL_CALLS", 2):
                result = chunk_audio_analyzer_node(state)

        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 2)
        self.assertEqual(result["chunk_results"][0]["chunk_id"], "chunk-000")
        self.assertEqual(
            result["chunk_results"][0]["audio_insights"],
            "[chunk=chunk-000] 音频摘要（降级）:\nCPU and GPU benchmark",
        )
        self.assertEqual(
            result["chunk_results"][1]["audio_insights"],
            "[chunk=chunk-001] 音频摘要（降级）:\nAPI design details",
        )
        self.assertIn("evidence_refs", result["chunk_results"][0])
        self.assertIn("latency_ms", result["chunk_results"][0])
        self.assertEqual(
            result["chunk_results"][0]["evidence_refs"]["audio_searches"],
            [
                {"query": "CPU", "result": "search:CPU"},
                {"query": "GPU", "result": "search:GPU"},
            ],
        )
        self.assertEqual(
            result["chunk_results"][1]["evidence_refs"]["audio_searches"],
            [{"query": "API", "result": "search:API"}],
        )
        self.assertEqual(mock_search.call_count, 3)

    def test_handles_empty_chunk_plan(self):
        state = cast(VideoSummaryState, {"chunk_plan": [], "chunk_results": [{"chunk_id": "x"}]})
        result = chunk_audio_analyzer_node(state)
        self.assertEqual(result["chunk_results"], [{"chunk_id": "x"}])

    @patch.dict("core.workflow.video_summary.nodes.chunk_audio_analyzer._AUDIO_SEARCH_CACHE", {}, clear=True)
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.execute_tavily_search")
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.OpenAI")
    def test_builds_audio_insights_with_successful_llm_response(self, mock_openai, mock_search):
        mock_search.return_value = "unused"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="精炼后的分片摘要"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        state = cast(
            VideoSummaryState,
            {
                "transcript": '{"segments": [{"start": 0, "end": 8, "text": "lowercase benchmark details"}]}',
                "chunk_plan": [{"chunk_id": "chunk-000", "transcript_segment_indexes": [0]}],
                "chunk_results": [{"chunk_id": "chunk-000", "evidence_refs": {"existing": True}}],
                "user_prompt": "关注技术要点",
            },
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = chunk_audio_analyzer_node(state)

        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["audio_insights"], "精炼后的分片摘要")
        self.assertTrue(result["chunk_results"][0]["evidence_refs"]["existing"])
        self.assertEqual(result["chunk_results"][0]["evidence_refs"]["transcript_segment_indexes"], [0])
        self.assertEqual(result["chunk_results"][0]["evidence_refs"]["audio_searches"], [])
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        mock_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
