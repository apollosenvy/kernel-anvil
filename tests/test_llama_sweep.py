"""Tests for llama_sweep helpers focused on edge-case correctness."""

import json
import sqlite3

import pytest

from kernel_anvil.llama_sweep import _gen_config, _parse_rocprof_db


class TestGenConfigUnknownQuants:
    def test_unknown_quant_emits_warning(self, capsys):
        # Known quant + an invented one. The known quant should still land in
        # configs; the unknown one is dropped, but a warning must be emitted
        # so the user knows a layer of their model didn't get tuned.
        shapes = {
            ("Q4_K", 4096, 4096): 1,
            ("INVENTED_QUANT", 4096, 4096): 1,
        }
        out = _gen_config(nwarps=4, shapes=shapes)
        captured = capsys.readouterr()

        configs = json.loads(out)["configs"]
        # Q4_K (type_idx 12) lands; INVENTED_QUANT does not.
        assert "12" in configs
        assert all("INVENTED" not in v for v in configs)

        assert "skipped untuned quant types" in captured.err
        assert "INVENTED_QUANT" in captured.err

    def test_no_warning_when_all_known(self, capsys):
        shapes = {("Q4_K", 4096, 4096): 1}
        _gen_config(nwarps=4, shapes=shapes)
        captured = capsys.readouterr()
        assert "skipped" not in captured.err


class TestParseRocprofDB:
    def _make_db(self, tmp_path, table_names: list[str]) -> str:
        """Build a tiny SQLite DB with the given table names. No data needed
        for the tests below -- we're checking that the LIKE filter correctly
        narrows to literal-underscore prefixes only."""
        db_path = tmp_path / "rocprof.db"
        db = sqlite3.connect(db_path)
        for name in table_names:
            # Identifiers may contain weird chars; quote with brackets.
            db.execute(f'CREATE TABLE "{name}" (id INTEGER)')
        db.commit()
        db.close()
        return str(db_path)

    def test_like_pattern_excludes_non_underscore_match(self, tmp_path):
        # 'rocpd_kernel_dispatchABC' would match the unescaped LIKE pattern
        # 'rocpd_kernel_dispatch_%' (because '_' is a single-char wildcard).
        # The fix escapes the underscores. Check that this table is NOT
        # selected as the dispatch table.
        db_path = self._make_db(tmp_path, [
            "rocpd_kernel_dispatchABC",  # ambiguous; SHOULD be skipped
        ])
        timings = _parse_rocprof_db(db_path)
        # No matching dispatch table -> empty result. Crucially, no crash.
        assert timings == []

    def test_handles_missing_symbol_table_gracefully(self, tmp_path):
        # The dispatch table exists with a literal-underscore prefix, but
        # the matching symbol table is absent. Pre-fix, the SELECT raised
        # OperationalError uncaught. Now it should return [].
        db_path = self._make_db(tmp_path, [
            "rocpd_kernel_dispatch_abc123",
            # Note: no rocpd_info_kernel_symbol_abc123 table.
        ])
        timings = _parse_rocprof_db(db_path)
        assert timings == []

    def test_no_matching_table_returns_empty(self, tmp_path):
        db_path = self._make_db(tmp_path, [
            "unrelated_table",
        ])
        timings = _parse_rocprof_db(db_path)
        assert timings == []
