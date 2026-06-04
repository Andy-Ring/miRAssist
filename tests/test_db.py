from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import backend.db as db_module
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local verifier env
    db_module = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipUnless(db_module is not None, f"sqlalchemy/backend.db unavailable: {_IMPORT_ERROR}")
class DatabaseEngineTests(unittest.TestCase):
    def tearDown(self) -> None:
        db_module._ENGINE_CACHE.clear()

    def test_psycopg_engine_disables_auto_prepared_statements(self) -> None:
        url = "postgresql://user:pass@aws-1-us-west-2.pooler.supabase.com:6543/postgres"

        with patch("backend.db.create_engine", return_value=object()) as create_engine_mock:
            engine = db_module.get_database_engine(url)

        self.assertIsNotNone(engine)
        create_engine_mock.assert_called_once()
        _, kwargs = create_engine_mock.call_args
        self.assertEqual(kwargs["connect_args"], {"prepare_threshold": None})
        self.assertTrue(kwargs["future"])
        self.assertTrue(kwargs["pool_pre_ping"])

    def test_engine_cache_reuses_same_url(self) -> None:
        url = "postgresql://user:pass@host:5432/db"
        fake_engine = object()

        with patch("backend.db.create_engine", return_value=fake_engine) as create_engine_mock:
            first = db_module.get_database_engine(url)
            second = db_module.get_database_engine(url)

        self.assertIs(first, fake_engine)
        self.assertIs(second, fake_engine)
        create_engine_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
