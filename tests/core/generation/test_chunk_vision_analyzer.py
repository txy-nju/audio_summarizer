import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_vision_analyzer import chunk_vision_analyzer_node
from core.workflow.video_summary.state import VideoSummaryState


class TestChunkVisionAnalyzerNode(unittest.TestCase):
    @patch.dict("core.workflow.video_summary.nodes.chunk_vision_analyzer._VISION_SEARCH_CACHE", {}, clear=True)
    @patch("core.workflow.video_summary.nodes.chunk_vision_analyzer.execute_tavily_search")
    def test_builds_vision_insights_for_each_chunk_without_api_key(self, mock_search):
        mock_search.side_effect = lambda query: f"vision-search:{query}"

        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "keyframe_indexes": [0, 1]},
                    {"chunk_id": "chunk-001", "keyframe_indexes": [2]},
                ],
                "keyframes": [
                    {"time": "00:03", "image": "a"},
                    {"time": "00:12", "image": "b"},
                    {"time": "01:05", "image": "c"},
                ],
                "chunk_results": [{"chunk_id": "chunk-000"}, {"chunk_id": "chunk-001"}],
                "user_prompt": "关注界面变化",
            },
        )

        with patch.dict("os.environ", {}, clear=True):
            with patch("core.workflow.video_summary.nodes.chunk_vision_analyzer.CHUNK_MAX_TOOL_CALLS", 2):
                result = chunk_vision_analyzer_node(state)

        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 2)
        self.assertEqual(
            result["chunk_results"][0]["vision_insights"],
            "[chunk=chunk-000] 视觉摘要（降级）：命中 2 帧，时间点 ['00:03', '00:12']",
        )
        self.assertEqual(
            result["chunk_results"][1]["vision_insights"],
            "[chunk=chunk-001] 视觉摘要（降级）：命中 1 帧，时间点 ['01:05']",
        )
        self.assertEqual(result["chunk_results"][0]["evidence_refs"]["keyframe_indexes"], [0, 1])
        self.assertEqual(result["chunk_results"][1]["evidence_refs"]["keyframe_indexes"], [2])
        self.assertEqual(
            result["chunk_results"][0]["evidence_refs"]["vision_searches"],
            [
                {"query": "video frame context at 00:03", "result": "vision-search:video frame context at 00:03"},
                {"query": "video frame context at 00:12", "result": "vision-search:video frame context at 00:12"},
            ],
        )
        self.assertEqual(
            result["chunk_results"][1]["evidence_refs"]["vision_searches"],
            [{"query": "video frame context at 01:05", "result": "vision-search:video frame context at 01:05"}],
        )
        self.assertEqual(mock_search.call_count, 3)

    @patch("core.workflow.video_summary.nodes.chunk_vision_analyzer.execute_tavily_search")
    def test_handles_invalid_keyframes_type(self, mock_search):
        state = cast(VideoSummaryState, {"chunk_plan": [{"chunk_id": "c1", "keyframe_indexes": [0]}], "keyframes": "bad", "chunk_results": []})
        with patch.dict("os.environ", {}, clear=True):
            result = chunk_vision_analyzer_node(state)
        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertIn("无可用关键帧证据", result["chunk_results"][0]["vision_insights"])
        mock_search.assert_not_called()

    @patch.dict("core.workflow.video_summary.nodes.chunk_vision_analyzer._VISION_SEARCH_CACHE", {}, clear=True)
    @patch("core.workflow.video_summary.nodes.chunk_vision_analyzer.execute_tavily_search")
    @patch("core.workflow.video_summary.nodes.chunk_vision_analyzer.OpenAI")
    def test_builds_vision_insights_with_successful_llm_response(self, mock_openai, mock_search):
        mock_search.return_value = "vision-search-ok"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="视觉主干摘要"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000", "keyframe_indexes": [0]}],
                "keyframes": [{"time": "00:03", "image": "a"}],
                "chunk_results": [{"chunk_id": "chunk-000", "evidence_refs": {"existing": True}}],
                "user_prompt": "关注界面变化",
            },
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = chunk_vision_analyzer_node(state)

        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["vision_insights"], "视觉主干摘要")
        self.assertTrue(result["chunk_results"][0]["evidence_refs"]["existing"])
        self.assertEqual(result["chunk_results"][0]["evidence_refs"]["keyframe_indexes"], [0])
        self.assertEqual(len(result["chunk_results"][0]["evidence_refs"]["vision_searches"]), 1)
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        mock_search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
