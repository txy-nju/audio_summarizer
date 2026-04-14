import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_audio_analyzer import chunk_audio_analyzer_node, chunk_audio_worker_node
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

    @patch.dict("core.workflow.video_summary.nodes.chunk_audio_analyzer._AUDIO_SEARCH_CACHE", {}, clear=True)
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.as_completed")
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.ThreadPoolExecutor")
    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer._process_single_chunk_audio")
    def test_parallel_execution_reduces_runtime(self, mock_process, mock_executor_cls, mock_as_completed):
        mock_process.side_effect = lambda chunk_id, indexes, transcript_items, user_prompt, base_item: (
            chunk_id,
            {
                "chunk_id": chunk_id,
                "audio_insights": f"summary-{chunk_id}",
                "evidence_refs": {"transcript_segment_indexes": indexes, "audio_searches": []},
                "token_usage": {"audio": 0},
                "latency_ms": {"audio": 1},
            },
        )

        class _FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        submitted_futures = []
        submitted_calls = []
        executor_instance = MagicMock()

        def _submit(fn, *args, **kwargs):
            submitted_calls.append((fn, args, kwargs))
            future = _FakeFuture(fn(*args, **kwargs))
            submitted_futures.append(future)
            return future

        executor_instance.submit.side_effect = _submit
        mock_executor_cls.return_value.__enter__.return_value = executor_instance
        mock_as_completed.side_effect = lambda futures: futures

        state = cast(
            VideoSummaryState,
            {
                "transcript": (
                    '{"segments": ['
                    '{"start": 0, "end": 5, "text": "seg0"}, '
                    '{"start": 10, "end": 15, "text": "seg1"}, '
                    '{"start": 20, "end": 25, "text": "seg2"}, '
                    '{"start": 30, "end": 35, "text": "seg3"}'
                    ']}'
                ),
                "chunk_plan": [
                    {"chunk_id": "chunk-000", "transcript_segment_indexes": [0]},
                    {"chunk_id": "chunk-001", "transcript_segment_indexes": [1]},
                    {"chunk_id": "chunk-002", "transcript_segment_indexes": [2]},
                    {"chunk_id": "chunk-003", "transcript_segment_indexes": [3]},
                ],
                "chunk_results": [],
                "user_prompt": "并行验证",
            },
        )

        with patch("core.workflow.video_summary.nodes.chunk_audio_analyzer.MAP_MAX_PARALLELISM", 2):
            result = chunk_audio_analyzer_node(state)

        self.assertEqual(len(result["chunk_results"]), 4)
        mock_executor_cls.assert_called_once_with(max_workers=2)
        self.assertEqual(executor_instance.submit.call_count, 4)
        self.assertEqual(len(submitted_calls), 4)
        self.assertEqual(len(submitted_futures), 4)
        mock_as_completed.assert_called_once()
        self.assertEqual(mock_process.call_count, 4)

    @patch("core.workflow.video_summary.nodes.chunk_audio_analyzer._process_single_chunk_audio")
    def test_chunk_audio_worker_processes_single_chunk(self, mock_process):
        mock_process.return_value = (
            "chunk-000",
            {
                "chunk_id": "chunk-000",
                "audio_insights": "worker-summary",
                "evidence_refs": {"transcript_segment_indexes": [0]},
            },
        )

        state = cast(
            VideoSummaryState,
            {
                "transcript": '{"segments": [{"start": 0, "end": 8, "text": "CPU"}]}',
                "user_prompt": "focus",
                "current_chunk": {"chunk_id": "chunk-000", "transcript_segment_indexes": [0]},
                "current_chunk_base_item": {"chunk_id": "chunk-000"},
            },
        )
        result = chunk_audio_worker_node(state)
        self.assertEqual(result["chunk_results"][0]["audio_insights"], "worker-summary")
        mock_process.assert_called_once()

    def test_chunk_audio_worker_handles_invalid_current_chunk(self):
        state = cast(VideoSummaryState, {"current_chunk": "bad"})
        result = chunk_audio_worker_node(state)
        self.assertEqual(result["chunk_results"], [])


if __name__ == "__main__":
    unittest.main()
