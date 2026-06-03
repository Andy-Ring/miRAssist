from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.config import get_default_k, get_synth_max_tokens


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_k_is_five(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MIRASSIST_DEFAULT_K", None)
            self.assertEqual(get_default_k(), 5)

    def test_synth_max_tokens_defaults_to_2500(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MIRASSIST_SYNTH_MAX_TOKENS", None)
            self.assertEqual(get_synth_max_tokens(), 2500)


if __name__ == "__main__":
    unittest.main()
