import csv
import multiprocessing as mp
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "final"), str(SRC / "replica")]
import common
from final import audit_statistics, evaluate, paper, stats
from replica import protocol
from replica import screen
from nasa_cohort import HISTORICAL, MAIN, run_files


def reference_f1(labels, predictions, average):
    return f1_score(labels, predictions, labels=[False, True], average=average, zero_division=0)


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_threshold_formulas_constant_ties_zeros_and_short_series(self):
        scores = np.array([0., 1., 2., 3., 9.])
        self.assertEqual(common.thr_iqr(scores), 3 + 1.5 * (3 - 1))
        self.assertEqual(common.thr_mad(scores), 2 + 3 * 1.4826)
        self.assertEqual(common.thr_static_frac(scores, 0.87), 0.87 * 9)
        self.assertEqual(common.thr_static_pct(scores, 50), 2)
        self.assertEqual(common.thr_pot(scores), 9)  # fewer than five exceedances
        for scores in (np.array([2.]), np.zeros(4), np.full(5, 2.),
                       np.array([0., 0., 1., 1., 1e100])):
            with self.subTest(scores=scores):
                thresholds = common.unsupervised_thresholds(scores)
                self.assertTrue(np.isfinite(list(thresholds.values())).all())
                if np.all(scores == scores[0]):
                    self.assertEqual(thresholds["iqr"], scores[0])
                    self.assertEqual(thresholds["mad"], scores[0])
                    self.assertTrue(np.all(scores >= thresholds["iqr"]))

    def test_metrics_match_sklearn_and_historical_dynamic_grid(self):
        y = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
        s = np.array([3., 1., 4., 2., 3., 9., 2., 9., 3.])
        k = len(y) // 2
        out = evaluate.metrics(y, s, .85)
        self.assertEqual(out["auprc"], average_precision_score(y, s))
        self.assertEqual(out["auroc"], roc_auc_score(y, s))
        for name, cutoff in {"static": np.percentile(s, 85), "iqr": 4 + 1.5 * (4 - 2)}.items():
            for avg, suffix in (("binary", "bin"), ("macro", "mac")):
                self.assertAlmostEqual(out[f"{name}_{suffix}"],
                                       reference_f1(y[k:], s[k:] >= cutoff, avg))
                self.assertAlmostEqual(out[f"{name}_full_{suffix}"],
                                       reference_f1(y, s >= cutoff, avg))
        candidates = np.percentile(s, evaluate.PCTS)
        for avg, suffix in (("binary", "bin"), ("macro", "mac")):
            cutoff = max(candidates, key=lambda t: reference_f1(y[:k], s[:k] >= t, avg))
            self.assertAlmostEqual(out[f"dyn_{suffix}"], reference_f1(y[k:], s[k:] >= cutoff, avg))
            if avg == "binary":
                self.assertAlmostEqual(out["dyn_bin_mac"], reference_f1(y[k:], s[k:] >= cutoff, "macro"))

    def test_unsupervised_masks_ignore_labels_and_dynamic_ignores_test_labels(self):
        s = np.array([1., 2., 3., 4., 2.7, 2.5, 2.6, 2.8])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
        with patch.object(evaluate, "f1s", wraps=evaluate.f1s) as original:
            evaluate.metrics(y, s, .85)
            before = [entry.args[1] for entry in original.call_args_list]
            original.reset_mock()
            evaluate.metrics(~y, s, .85)
            flipped = [entry.args[1] for entry in original.call_args_list]
            original.reset_mock()
            evaluate.metrics(np.r_[y[:4], ~y[4:]], s, .85)
            test_flipped = [entry.args[1] for entry in original.call_args_list]
        for a, b in zip(before[:4], flipped[:4]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(before[4:], test_flipped[4:]):
            np.testing.assert_array_equal(a, b)

    def test_historical_dynamic_grid_uses_scores_from_both_halves(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
        s = np.array([1., 2., 3., 4., 2.7, 2.5, 2.6, 2.8])
        first_test_predictions = []
        for scores in (s, np.r_[s[:5], 100., 101., 102.]):
            with patch.object(evaluate, "f1s", wraps=evaluate.f1s) as f1:
                evaluate.metrics(y, scores, .85)
            first_test_predictions.append(bool(f1.call_args_list[-3].args[1][0]))
        self.assertEqual(first_test_predictions, [True, False])

    def test_f1_fixed_classes_and_all_tied_scores(self):
        for y, p in ((np.zeros(3, bool), np.zeros(3, bool)),
                     (np.ones(3, bool), np.ones(3, bool)),
                     (np.array([False, True]), np.zeros(2, bool))):
            self.assertEqual(evaluate.f1s(y, p),
                             tuple(reference_f1(y, p, avg) for avg in ("binary", "macro")))
        y = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
        out = evaluate.metrics(y, np.zeros(len(y)), .85)
        self.assertEqual(out["iqr_rate"], 1.)
        self.assertEqual(out["auroc"], .5)
        self.assertEqual(out["auprc"], y.mean())

    def test_one_class_auc_requires_explicit_policy_before_serialization(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "both classes|dos clases"):
                evaluate.metrics(np.zeros(5, bool), np.arange(5.), .85)

    def test_legacy_oracle_documents_unattainable_tied_score_result(self):
        y, s = np.array([True, False]), np.ones(2)
        attainable = f1_score(y, s >= 1)
        self.assertEqual(protocol.oracle_f1(y, s), 1.)
        self.assertAlmostEqual(attainable, 2 / 3)

    def test_legacy_zero_scores_fail_but_final_route_is_independent(self):
        y = np.array([False, True, False, True])
        with np.errstate(invalid="raise"), self.assertRaises(FloatingPointError):
            protocol.evaluate(y, np.zeros(4), .85)
        with (
            patch.object(protocol, "evaluate", side_effect=AssertionError("legacy evaluator called")),
            patch.object(screen, "evaluate", side_effect=AssertionError("legacy evaluator called")),
        ):
            self.assertEqual(screen.parse_variant("full").grad, "full")
            self.assertEqual(evaluate.metrics(y, np.zeros(4), .85)["auprc"], .5)

    def fixture(self, name, y, **scores):
        path = self.tmp / f"{name}_s0.npz"
        np.savez(path, y=y, **scores)
        return path

    def test_file_job_aligns_variants_and_uses_stable_phase4_projection(self):
        n, L, P = 48, 4, 1
        y = (np.arange(n) % 3 == 0)
        base = np.ones(n - L - P + 1)
        base[0] = 2**54
        other = np.arange(len(base), dtype=float)
        path = self.fixture("artificial", y, base=base, adapt_r3=other)
        with patch.object(evaluate, "metrics", wraps=evaluate.metrics) as metrics:
            rows, _ = evaluate.file_job(("real", path))
        interior = slice(19, n - 19)
        self.assertEqual([(r["train"], r["proj"]) for r in rows],
                         [("base", "win"), ("base", "tgt"), ("adapt_r3", "win"), ("adapt_r3", "tgt")])
        for call, s, span in zip(metrics.call_args_list, (base, base, other, other), (5, 1, 5, 1)):
            reference = np.array([sum(s[t - 4:t - 4 + span]) for t in range(19, n - 19)])
            np.testing.assert_array_equal(call.args[0], y[interior])
            np.testing.assert_array_equal(call.args[1], reference)

    def test_file_job_rejects_missing_window_score(self):
        path = self.fixture("short_scores", np.arange(48) % 2, base=np.ones(43))  # expected 44
        with self.assertRaisesRegex(ValueError, "length|longitud"):
            evaluate.file_job(("real", path))

    def test_file_job_reports_when_edge_crop_leaves_no_evaluation_points(self):
        path = self.fixture("short_series", np.arange(38) % 2, base=np.ones(34))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "edge|recorte"):
                evaluate.file_job(("real", path))

    def test_nasa_evaluator_excludes_archived_p2_by_default(self):
        paths = self.nasa_paths()
        spawn = self.enterContext(patch.object(mp, "get_context"))
        pool = spawn.return_value.Pool.return_value.__enter__.return_value
        pool.imap_unordered.return_value = []
        with patch.object(evaluate, "RES", self.tmp), patch.object(
            sys, "argv", ["evaluate.py", "nasa", "--workers", "1"]
        ):
            evaluate.main()
        self.assertEqual(pool.imap_unordered.call_args.args[1], [("nasa", p) for p in paths[:400]])

    def test_nasa_halo_excludes_archived_p2_by_default(self):
        paths = self.nasa_paths()
        spawn = self.enterContext(patch.object(mp, "get_context"))
        pool = spawn.return_value.Pool.return_value.__enter__.return_value
        pool.imap_unordered.return_value = []
        with patch.multiple(paper, RES=self.tmp, OUT=self.tmp):
            paper.run_halo("nasa", workers=1)
        self.assertEqual(pool.imap_unordered.call_args.args[1], [("nasa", p) for p in paths[:400]])

    def nasa_paths(self):
        root = self.tmp / "nasa"
        root.mkdir()
        paths = [root / f"{ds}_s{seed}.npz" for ds in
                 [*(f"A-{i:02d}" for i in range(80)), "P-2"] for seed in range(5)]
        for path in paths:
            path.touch()
        return paths

    def test_nasa_run_inventory_rejects_missing_seed_and_supports_historical(self):
        paths = self.nasa_paths()
        self.assertEqual(run_files(paths[0].parent, HISTORICAL), paths)
        paths[0].unlink()
        for cohort in (MAIN, HISTORICAL):
            with self.subTest(cohort=cohort), self.assertRaisesRegex(ValueError, "semillas"):
                run_files(paths[0].parent, cohort)

    def test_corrected_statistics_keep_80_and_81_channel_cohorts_separate(self):
        path = self.tmp / "nasa_eval.csv"
        cols = ["ds", "seed", "cfg", "train", "proj", "auprc", "auroc", "iqr_bin", "dyn_bin", "anom_rate"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(cols)
            for i in range(80):
                for seed in range(5):
                    for train, proj in sorted(audit_statistics.METHODS):
                        writer.writerow([f"A-{i}", seed, "L4P1", train, proj, .5, .5, .5, .5, .1])
        methods = audit_statistics.load_block(path, "nasa")
        self.assertEqual(len(audit_statistics.validate_pairs(methods)), 400)
        with self.assertRaisesRegex(ValueError, "81 canales"):
            audit_statistics.load_block(path, "nasa", nasa_cohort=HISTORICAL)
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for seed in range(5):
                for train, proj in sorted(audit_statistics.METHODS):
                    writer.writerow(["P-2", seed, "L4P1", train, proj, .5, .5, .5, .5, .1])
        methods = audit_statistics.load_block(path, "nasa", nasa_cohort=HISTORICAL)
        self.assertEqual(len(audit_statistics.validate_pairs(methods)), 405)
        with self.assertRaisesRegex(ValueError, "80 canales"):
            audit_statistics.load_block(path, "nasa")

    def test_file_job_rejects_invalid_labels_and_scores_before_projection(self):
        for y, scores, message in (
            (np.ones((48, 1)), {"base": np.ones(44)}, "etiquetas.*vector"),
            (np.arange(48), {"base": np.ones(44)}, "binarias"),
            (np.arange(48) % 2, {"base": np.r_[np.nan, np.ones(43)]}, "finitos"),
            (np.arange(48) % 2, {"ts_lof": np.ones(47)}, "longitud 48"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate.file_job(("real", self.fixture("invalid", y, **scores)))

    def test_block_rejects_unpaired_configurations_before_writing(self):
        root = self.tmp / "real"
        root.mkdir()
        for seed in (0, 1):
            (root / f"A_s{seed}.npz").touch()
        results = [
            ([dict(ds="A", seed=0, cfg="L4P1", train="base", proj="win")], []),
            ([dict(ds="A", seed=1, cfg="L4P1", train="base", proj="tgt")], []),
        ]
        with (
            patch.object(mp, "get_context") as spawn,
            patch.object(evaluate, "RES", self.tmp),
            patch.object(sys, "argv", ["evaluate.py", "real", "--workers", "1"]),
        ):
            pool = spawn.return_value.Pool.return_value.__enter__.return_value
            pool.imap_unordered.return_value = results
            with self.assertRaisesRegex(ValueError, "no pareadas"):
                evaluate.main()
        self.assertFalse((self.tmp / "real_eval.csv").exists())

    def test_historical_statistics_reject_incomplete_or_undefined_pairs(self):
        frame = pd.DataFrame(dict(ds=["A", "B", "A"], seed=[0, 0, 0],
                                  method=["final", "final", "base"], auprc=[.5, .6, .7]))
        with self.assertRaisesRegex(ValueError, "no pareada"):
            stats.paired(frame, "real", "final", "base", "auprc")
        frame.loc[3] = ["B", 0, "base", np.nan]
        with self.assertRaisesRegex(ValueError, "indefinida"):
            stats.paired(frame, "real", "final", "base", "auprc")

    def test_h2_stays_pending_without_margin(self):
        row = audit_statistics.noninferiority({"A": -.02, "B": .01, "C": .03}, None, inference=True)
        self.assertEqual(row["status"], "pending_margin")
        self.assertIsNone(row["p"])
        self.assertIsNone(row["margin"])
        self.assertAlmostEqual(row["mean"], (.01 + .03 - .02) / 3)
