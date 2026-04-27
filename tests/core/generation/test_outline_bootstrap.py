import unittest
from typing import cast

from core.workflow.video_summary.nodes.outline_bootstrap import outline_bootstrap_node
from core.workflow.video_summary.state import VideoSummaryState


class TestOutlineBootstrapNode(unittest.TestCase):
    def test_builds_structured_context_without_narrative_summary(self):
        state = cast(
            VideoSummaryState,
            {
                "transcript": (
                    '{"segments": ['
                    '{"start": 0, "end": 8, "text": "OpenAI launches GPT4o in Beijing"}, '
                    '{"start": 9, "end": 18, "text": "随后团队演示 LangGraph workflow"}'
                    ']}'
                ),
                "chunk_plan": [
                    {
                        "chunk_id": "chunk-000",
                        "start_sec": 0,
                        "end_sec": 60,
                        "transcript_segment_indexes": [0, 1],
                        "keyframe_indexes": [0],
                    }
                ],
            },
        )

        result = outline_bootstrap_node(state)

        structured = result.get("structured_global_context", {})
        self.assertIn("entities", structured)
        self.assertIn("timeline_anchors", structured)
        self.assertNotIn("summary", structured)
        self.assertFalse(structured.get("source_policy", {}).get("narrative_summary_allowed", True))

        entity_names = {item.get("name") for item in structured.get("entities", []) if isinstance(item, dict)}
        self.assertIn("OpenAI", entity_names)
        self.assertIn("LangGraph", entity_names)

        anchors = structured.get("timeline_anchors", [])
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].get("chunk_id"), "chunk-000")
        self.assertEqual(anchors[0].get("transcript_segment_indexes"), [0, 1])

    def test_handles_invalid_inputs_with_empty_structured_context(self):
        result = outline_bootstrap_node(cast(VideoSummaryState, {"transcript": "{bad-json}", "chunk_plan": "invalid"}))

        structured = result.get("structured_global_context", {})
        self.assertEqual(structured.get("entities"), [])
        self.assertEqual(structured.get("timeline_anchors"), [])
        self.assertEqual(structured.get("source_policy", {}).get("allowed_fields"), ["entities", "timeline_anchors"])

    def test_filters_spoken_chinese_fillers_and_keeps_keywords(self):
        state = cast(
            VideoSummaryState,
            {
                "transcript": (
                    '{"segments": ['
                    '{"start": 0, "end": 10, "text": "我觉得你看这个项目其实是一个多模态视频总结系统"}, '
                    '{"start": 11, "end": 20, "text": "然后我们用了 LangGraph 和 OpenAI API 来编排工作流"}'
                    ']}'
                ),
                "chunk_plan": [
                    {
                        "chunk_id": "chunk-000",
                        "start_sec": 0,
                        "end_sec": 30,
                        "transcript_segment_indexes": [0, 1],
                        "keyframe_indexes": [],
                    }
                ],
            },
        )

        result = outline_bootstrap_node(state)
        entity_names = {item.get("name") for item in result.get("structured_global_context", {}).get("entities", [])}

        self.assertTrue(any("多模态" in str(name) for name in entity_names))
        self.assertIn("LangGraph", entity_names)
        self.assertIn("OpenAI", entity_names)
        self.assertNotIn("我觉得", entity_names)
        self.assertNotIn("你看这个", entity_names)
        self.assertNotIn("其实", entity_names)
        self.assertNotIn("一个多", entity_names)


if __name__ == "__main__":
    unittest.main()