from __future__ import annotations

import unittest

from frontend.help_content import (
    get_about_evidence_markdown,
    get_how_to_use_markdown,
    get_normal_debug_sections,
)


class UIContentTests(unittest.TestCase):
    def test_normal_ui_only_exposes_core_debug_sections(self) -> None:
        self.assertEqual(
            get_normal_debug_sections(),
            ["Planner output (QuerySpec)", "Evidence shortlist"],
        )

    def test_instructions_include_help_section(self) -> None:
        text = get_how_to_use_markdown()
        self.assertIn("What genes are regulated by hsa-miR-210-3p?", text)
        self.assertIn("strict pathway filtering", text)
        self.assertIn("Evidence support count", text)
        self.assertIn("returns the top candidates as a chart", text)
        self.assertIn("Novel mode", text)

    def test_about_evidence_section_lists_core_sources(self) -> None:
        text = get_about_evidence_markdown()
        self.assertIn("Sequence complementarity", text)
        self.assertIn("Sequence conservation", text)
        self.assertIn("Functional binding", text)
        self.assertIn("Pathway filter", text)


if __name__ == "__main__":
    unittest.main()
