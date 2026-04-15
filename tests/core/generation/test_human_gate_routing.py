import unittest

from core.workflow.video_summary.edges.router import (
    route_after_human_gate,
    ROUTE_HUMAN_APPROVED,
    ROUTE_PENDING_HUMAN_REVIEW,
)


class TestHumanGateRouting(unittest.TestCase):
    def test_route_pending_by_default(self):
        self.assertEqual(route_after_human_gate({}), ROUTE_PENDING_HUMAN_REVIEW)  # type: ignore[arg-type]

    def test_route_approved(self):
        state = {"human_gate_status": "approved"}
        self.assertEqual(route_after_human_gate(state), ROUTE_HUMAN_APPROVED)  # type: ignore[arg-type]

    def test_route_pending_for_non_approved_values(self):
        state = {"human_gate_status": "pending"}
        self.assertEqual(route_after_human_gate(state), ROUTE_PENDING_HUMAN_REVIEW)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
