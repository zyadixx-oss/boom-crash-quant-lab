import unittest

import verify as v


class VerifyTests(unittest.TestCase):
    def test_stats_keeps_missing_and_zero_separate(self):
        self.assertEqual(v.stats([None, -3., 0., 5.]), {"points": 4, "finite": 3, "null": 1, "min": -3., "median": 0., "max": 5., "positive": 1, "negative": 1, "zero": 1})
        self.assertEqual(v.stats([])["median"], None)
        self.assertEqual(v.stats([2., 4.])["median"], 3.)

    def test_edges_mask_inactive_and_repeated_scientific_pairs(self):
        a, b = {"science_id": "A"}, {"science_id": "B"}
        seen = set()
        self.assertEqual(v.edge_expected("threshold", a, a, seen), "INACTIVE_AXIS_ALIAS")
        self.assertEqual(v.edge_expected("lookback", a, b, seen), "EFFECTIVE_NEIGHBOR")
        self.assertEqual(v.edge_expected("lookback", a, b, seen), "DUPLICATE_EFFECTIVE_EDGE")
        self.assertEqual(v.edge_expected("target_atr", a, b, seen), "EFFECTIVE_NEIGHBOR")

    def test_numeric_nonfinite_rejected(self):
        for bad in ("NaN", "Infinity", "-inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                v.number(bad)
        self.assertIsNone(v.number(""))
        self.assertEqual(v.number("0"), 0.)


if __name__ == "__main__":
    unittest.main()
