import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_statistics", ROOT / "src/final/audit_statistics.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditStatisticsTests(unittest.TestCase):
    def test_paired_by_key_not_input_order(self):
        a, b = ("ds", "0"), ("ds", "1")
        methods = {audit.BASE: {a: {"f": 0.1}, b: {"f": 0.2}},
                   audit.M3: {a: {}, b: {}},
                   audit.FINAL: {b: {"f": 0.5}, a: {"f": 0.1}}}
        actual = audit.differences(methods, audit.FINAL, audit.BASE, "f")
        self.assertEqual(actual, {a: 0.0, b: 0.3})
        methods[audit.BASE].pop(a)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            audit.validate_pairs(methods)

    def test_duplicate_csv_key_rejected(self):
        row = dict(ds="ds", seed=0, train="base", proj="win", cfg="L4P1",
                   auprc=0, auroc=0, iqr_bin=0, dyn_bin=0, anom_rate=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.load_block(path, "real")

    def test_aggregates_seeds_and_standardizes_unit_effect(self):
        units = audit.aggregate({("a", "0"): 0.2, ("a", "1"): 0.4,
                                 ("b", "0"): 0.5, ("b", "1"): 0.5}, 0)
        self.assertAlmostEqual(units["a"], 0.3)
        effect = audit.estimate(units, inference=False)
        self.assertEqual(effect["n"], 2)
        self.assertAlmostEqual(effect["mean"], 0.4)
        self.assertAlmostEqual(effect["dz"], 0.4 / (0.02 ** 0.5))
        self.assertIsNone(effect["p"])
        self.assertIsNone(effect["ci95"])

    def test_drop_matches_level_endpoints_and_orients_effect(self):
        deltas = {("m", "0.01", "0"): -0.125, ("m", "0.3", "0"): 0.25}
        self.assertEqual(audit.drop_differences(deltas, 1), {("m", "0"): 0.375})

    def test_h2_pending_and_bound_matches_unadjusted_test(self):
        units = {"a": -0.025, "b": -0.03, "c": -0.02, "d": -0.035}
        pending = audit.noninferiority(units, None, inference=True)
        self.assertEqual(pending["status"], "pending_margin")
        self.assertIsNone(pending["p"])
        self.assertIsNone(pending["lower95"])
        self.assertIsNotNone(pending["ci95"])
        for margin in (0.05, 0.015):
            result = audit.noninferiority(units, margin, inference=True)
            self.assertEqual(result["p"] < 0.05, result["lower95"] > -margin)
        self.assertIsNone(audit.noninferiority(units, 0.05, inference=False)["p"])
        with self.assertRaises(ValueError):
            audit.noninferiority(units, 0, inference=True)

    def test_holm_reserves_pending_slot(self):
        rows = [{"p": p, "p_holm": None} for p in [0.01, None, 0.02]]
        audit.holm_family(rows)
        self.assertEqual([r["p_holm"] for r in rows], [0.03, None, 0.04])

    def test_archive_unit_counts_and_seed_stability(self):
        expected = {"synth": (20, 300), "contam": (20, 420), "real": (4, 40),
                    "smd": (28, 140), "nasa": (81, 405), "semi": (10, 450)}
        for block, (units, pairs) in expected.items():
            with self.subTest(block=block):
                methods = audit.load_block(ROOT / "results/final" / f"{block}_eval.csv", block,
                                           nasa_cohort=audit.HISTORICAL)
                results = audit.analyze_block(methods, block)
                self.assertEqual((results["H1"]["n"], results["H1"]["n_pairs"]), (units, pairs))
                self.assertEqual(results["H2"]["status"], "pending_margin")
                self.assertIsNone(results["H2"]["p_holm"])
                if block == "real":
                    self.assertIsNone(results["H1"]["ci95"])
                    for ds, seeds in results["H1"]["seed_deltas_by_dataset"].items():
                        self.assertEqual(len(seeds), 10)
                        self.assertAlmostEqual(results["H1"]["units"][ds], sum(seeds.values()) / 10)
                if block == "contam":
                    self.assertEqual((results["H4"]["n"], results["H4"]["n_pairs"]), (20, 60))
                if block == "semi":
                    self.assertEqual((results["H4r"]["n"], results["H4r"]["n_pairs"]), (10, 90))


if __name__ == "__main__":
    unittest.main()
