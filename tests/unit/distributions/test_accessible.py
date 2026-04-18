"""
Unit tests for scikit_pierre.distributions.accessible.
"""
import unittest

import pytest

from scikit_pierre.distributions.accessible import distributions_funcs


ALL_KNOWN_KEYS = [
    "CWS", "WPS", "PGD", "PGD_P",
    "TWB", "TWB_P", "TGD", "TGD_P",
    "GLEB", "GLEB_P",
    "TWB_GLEB", "TWB_GLEB_P",
    "TSW", "TSW_P",
    "TSW_GLEB", "TSW_GLEB_P",
    "TSW_TWB", "TSW_TWB_P",
    "TSW_TWB_GLEB", "TSW_TWB_GLEB_P",
]


class TestDistributionsFuncs(unittest.TestCase):

    # ── all valid keys return a callable ──────────────────────────────────

    def test_cws_returns_callable(self):
        """CWS key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("CWS")))

    def test_wps_returns_callable(self):
        """WPS key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("WPS")))

    def test_pgd_returns_callable(self):
        """PGD key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("PGD")))

    def test_pgd_p_returns_callable(self):
        """PGD_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("PGD_P")))

    def test_twb_returns_callable(self):
        """TWB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TWB")))

    def test_twb_p_returns_callable(self):
        """TWB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TWB_P")))

    def test_tgd_returns_callable(self):
        """TGD key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TGD")))

    def test_tgd_p_returns_callable(self):
        """TGD_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TGD_P")))

    def test_gleb_returns_callable(self):
        """GLEB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("GLEB")))

    def test_gleb_p_returns_callable(self):
        """GLEB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("GLEB_P")))

    def test_twb_gleb_returns_callable(self):
        """TWB_GLEB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TWB_GLEB")))

    def test_twb_gleb_p_returns_callable(self):
        """TWB_GLEB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TWB_GLEB_P")))

    def test_tsw_returns_callable(self):
        """TSW key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW")))

    def test_tsw_p_returns_callable(self):
        """TSW_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_P")))

    def test_tsw_gleb_returns_callable(self):
        """TSW_GLEB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_GLEB")))

    def test_tsw_gleb_p_returns_callable(self):
        """TSW_GLEB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_GLEB_P")))

    def test_tsw_twb_returns_callable(self):
        """TSW_TWB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_TWB")))

    def test_tsw_twb_p_returns_callable(self):
        """TSW_TWB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_TWB_P")))

    def test_tsw_twb_gleb_returns_callable(self):
        """TSW_TWB_GLEB key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_TWB_GLEB")))

    def test_tsw_twb_gleb_p_returns_callable(self):
        """TSW_TWB_GLEB_P key maps to a callable."""
        self.assertTrue(callable(distributions_funcs("TSW_TWB_GLEB_P")))

    # ── unknown keys raise NameError ──────────────────────────────────────

    def test_unknown_key_raises_name_error(self):
        """An unrecognised distribution name raises NameError."""
        with self.assertRaises(NameError):
            distributions_funcs("UNKNOWN")

    def test_empty_string_raises_name_error(self):
        """An empty string raises NameError."""
        with self.assertRaises(NameError):
            distributions_funcs("")

    def test_lowercase_key_raises_name_error(self):
        """Keys are case-sensitive; lowercase 'cws' is not recognised."""
        with self.assertRaises(NameError):
            distributions_funcs("cws")

    def test_all_known_keys_count(self):
        """Exactly 20 distribution keys are registered."""
        self.assertEqual(len(ALL_KNOWN_KEYS), 20)

    def test_all_known_keys_return_distinct_callables(self):
        """Every key resolves to a distinct function object."""
        funcs = [distributions_funcs(k) for k in ALL_KNOWN_KEYS]
        self.assertEqual(len(set(id(f) for f in funcs)), len(funcs))
