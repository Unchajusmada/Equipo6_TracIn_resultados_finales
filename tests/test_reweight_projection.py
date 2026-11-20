import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "final")]
from final import core, evaluate
import common
import forecaster
import lab


def project_reference(scores, n, start, stop):
    total = np.zeros(n)
    count = np.zeros(n, dtype=int)
    for i, score in enumerate(scores):
        for t in range(i + start, i + stop):
            total[t] += score
            count[t] += 1
    mean = np.divide(total, count, out=np.zeros_like(total), where=count != 0)
    return total, count, mean


class ReweightProjectionTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(17)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def test_weights_follow_protocol_and_keep_threshold_ties(self):
        # For these log-scores median=0 and unscaled MAD=1.
        scores = np.exp([-1, -1, 0, 0, 1, 4, 20])
        soft = forecaster.weights_from_influence(scores, "soft")
        np.testing.assert_allclose(soft,
                                   [1, 1, 1, 1, 1, np.exp(2 - 4 / 1.4826), 0.05])
        z = forecaster.robust_z(np.log(scores))
        legacy = np.where(z <= 2, 1., np.maximum(0.05, np.exp(-(z - 2))))
        np.testing.assert_array_equal(soft, legacy)
        np.testing.assert_array_equal(forecaster.weights_from_influence(scores, "trim"),
                                      [1, 1, 1, 1, 1, 0, 0])
        # Shift the cut to a computed z value so equality can be tested exactly.
        z_on_cut = np.log(scores[-2]) / 1.4826
        self.assertEqual(forecaster.weights_from_influence(scores, "trim", z0=z_on_cut)[-2], 1)
        self.assertEqual(forecaster.weights_from_influence(scores, "soft", z0=z_on_cut)[-2], 1)
        np.testing.assert_array_equal(core.adapt_weights(np.array([0, 1, 2, 3, 6])), np.ones(5))
        np.testing.assert_array_equal(core.adapt_weights(np.array([0, 1, 2, 3, 7])), [1, 1, 1, 1, 0])

    def test_weight_shapes_ranges_constant_scores_and_high_outliers(self):
        for scores in (np.array([0.]), np.zeros(5), np.ones(5),
                       np.array([1., 1., 1., 1., 1e300]), np.array([0., 1., 1., 1., 10.])):
            for mode in ("soft", "trim", "adapt"):
                with self.subTest(scores=scores, mode=mode), np.errstate(over="ignore"):
                    w = (core.adapt_weights(scores) if mode == "adapt"
                         else forecaster.weights_from_influence(scores, mode))
                    self.assertEqual(w.shape, scores.shape)
                    self.assertTrue(np.isfinite(w).all())
                    self.assertTrue(((w >= (0.05 if mode == "soft" else 0)) & (w <= 1)).all())
                    np.testing.assert_array_equal(w[:-1], np.ones(len(scores) - 1))
                    self.assertEqual(w[-1], (0.05 if mode == "soft" else 0) if scores[-1] > 1 else 1)

    def test_soft_finite_low_outlier_does_not_overflow_on_unselected_branch(self):
        with np.errstate(over="raise"), warnings.catch_warnings(record=True) as emitted, \
                patch.object(np, "exp", wraps=np.exp) as exp:
            warnings.simplefilter("always")
            w = forecaster.weights_from_influence(np.array([0., 1., 1., 1., 10.]), "soft")
        np.testing.assert_array_equal(w, [1, 1, 1, 1, 0.05])
        self.assertEqual(emitted, [])
        exp.assert_called_once()
        self.assertEqual(exp.call_args.args[0].shape, (1,))
        self.assertLess(exp.call_args.args[0][0], 0)

    def test_rounds_use_previous_scores_and_labels_only_change_diagnostics(self):
        x = np.zeros((10, 2))
        # An outlier moves to a different index after every mocked training run.
        scores = [np.where(np.arange(6) == i, 100., 1.) for i in range(6)]
        inputs = [None, 0, 1, 0, 3, 0, 5, 1]
        outputs = [0, 1, 2, 3, 4, 5, 1, 2]
        trainer_results = [dict(si=scores[i], loss_final=np.zeros(6), epochs=1) for i in outputs]
        runs = []
        for y in (np.zeros(10), np.ones(10), None):
            with patch.object(core, "run", side_effect=trainer_results) as run, warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                runs.append(core.all_variants(x, y, 4, 1, 17, lab.Cfg()))
            self.assertEqual(run.call_count, 8)
            for i, (call, source) in enumerate(zip(run.call_args_list, inputs)):
                self.assertIs(call.args[0], x)
                self.assertEqual(call.args[1:4], (4, 1, 17))
                if source is None:
                    self.assertNotIn("weights", call.kwargs)
                else:
                    expected = np.ones(6)
                    expected[source] = 0.05 if i in (1, 2) else 0
                    np.testing.assert_array_equal(call.kwargs["weights"], expected)
        names = ("base", "pd", "soft_r1", "soft_r2", "trim_r1", "trim_r2", "adapt_r1", "adapt_r2", "adapt_r3")
        for out in runs:
            self.assertEqual(set(k for k in out if not k.startswith("diag_")), set(names))
            for name, i in zip(names[2:], outputs[1:]):
                np.testing.assert_array_equal(out[name], scores[i])
        for name in names:
            np.testing.assert_array_equal(runs[0][name], runs[1][name])
            np.testing.assert_array_equal(runs[0][name], runs[2][name])
        self.assertGreater(runs[0]["diag_trim_r1_norm"], 0)
        self.assertGreater(runs[1]["diag_trim_r1_anom"], 0)
        self.assertTrue(np.isnan(runs[0]["diag_trim_r1_anom"]))
        self.assertTrue(np.isnan(runs[1]["diag_trim_r1_norm"]))
        self.assertNotIn("diag_trim_r1_anom", runs[2])

    def test_actual_adapt_rounds_reinitialize_model_and_adam(self):
        x = np.arange(20, dtype=np.float32).reshape(10, 2)
        cfg = lab.Cfg(sched=False, hidden=2, max_epochs=1)
        adam = torch.optim.Adam
        initializations, optimizers = [], []

        def record_optimizer(params, **kwargs):
            params = tuple(params)
            initializations.append(torch.cat([p.detach().flatten() for p in params]))
            opt = adam(params, **kwargs)
            self.assertFalse(opt.state)
            optimizers.append(opt)
            return opt

        with patch.object(torch.optim, "Adam", side_effect=record_optimizer):
            core.all_variants(x, None, 4, 1, 17, cfg, modes=("adapt",))
        self.assertEqual(len({id(opt) for opt in optimizers}), 4)
        for initial in initializations[1:]:
            torch.testing.assert_close(initial, initializations[0], rtol=0, atol=0)

    def test_zero_weight_minibatch_leaves_adam_and_parameters_unchanged(self):
        x = np.array([[0.], [1.], [-1.], [2.]], dtype=np.float32)
        weights = np.zeros(2)
        order = torch.randperm(2, generator=torch.Generator().manual_seed(17))
        weights[order[0].item()] = 1
        adam = torch.optim.Adam

        def snapshot(opt):
            params = opt.param_groups[0]["params"]
            return [(p.detach().clone(), {key: opt.state[p][key].clone()
                                          for key in ("step", "exp_avg", "exp_avg_sq")})
                    for p in params]

        for train in ("lab", "forecaster"):
            with self.subTest(train=train):
                optimizers, steps = [], []

                def record_optimizer(*args, **kwargs):
                    opt = adam(*args, **kwargs)
                    optimizers.append(opt)
                    opt.register_step_post_hook(lambda opt, *_: steps.append(snapshot(opt)))
                    return opt

                with patch.object(torch.optim, "Adam", side_effect=record_optimizer):
                    if train == "lab":
                        lab.run(x, 2, 1, 17, lab.Cfg(hidden=2, batch=1, max_epochs=1), weights=weights)
                    else:
                        forecaster.train_trace(x, 2, 1, 17, batch=1, max_epochs=1,
                                               weights=weights, model_kw={"hidden": 2})
                self.assertEqual(len(steps), 1)
                after_first = steps[0]
                self.assertTrue(any(torch.count_nonzero(state["exp_avg"]) for _, state in after_first))
                self.assertTrue(any(torch.count_nonzero(state["exp_avg_sq"]) for _, state in after_first))
                for (p, state), (end, end_state) in zip(after_first, snapshot(optimizers[0])):
                    self.assertEqual(state["step"].item(), 1)
                    torch.testing.assert_close(end, p, rtol=0, atol=0)
                    for key in state:
                        torch.testing.assert_close(end_state[key], state[key], rtol=0, atol=0)

    def test_projection_matches_loops_support_counts_and_endpoints(self):
        for horizon in (1, 2):
            L, n = 4, 11
            m = n - L - horizon + 1
            for scores in (np.arange(1., m + 1), np.ones(m), np.eye(m)[0], np.eye(m)[-1]):
                with self.subTest(horizon=horizon, scores=scores):
                    win, win_counts, _ = project_reference(scores, n, 0, L + horizon)
                    _, tgt_counts, tgt = project_reference(scores, n, L, L + horizon)
                    np.testing.assert_array_equal(common.project_sum(scores, L + horizon, n), win)
                    np.testing.assert_array_equal(common.project_targets_mean(scores, L, horizon, n), tgt)
                    self.assertEqual(win_counts.sum(), m * (L + horizon))
                    self.assertEqual(tgt_counts.sum(), m * horizon)
                    self.assertEqual(tgt_counts.max(), horizon)
                    np.testing.assert_array_equal(np.flatnonzero(tgt_counts), np.arange(L, n))
                    if horizon == 1:
                        np.testing.assert_array_equal(tgt[L:], scores)
            # A single window covers both targets when P=2, but contributes only once to each.
            single = np.array([7.])
            np.testing.assert_array_equal(common.project_sum(single, L + horizon, L + horizon),
                                          np.full(L + horizon, 7.))
            np.testing.assert_array_equal(common.project_targets_mean(single, L, horizon, L + horizon),
                                          np.r_[np.zeros(L), np.full(horizon, 7.)])

    def test_multichannel_trace_reuses_one_window_score_for_each_target(self):
        L, P, n, channels = 4, 2, 10, 3
        x = torch.arange(n * channels, dtype=torch.float32).reshape(n, channels) / 10
        model = forecaster.LSTMForecaster(channels, L, P, hidden=2)
        loss, hsq = forecaster._checkpoint_stats(model, x, n - L - P + 1, L, P, bs=2)
        trace = forecaster.Trace(L, P, channels, n, np.array([0.1]), loss.numpy()[None],
                                 hsq.numpy()[None], np.array([float(loss.mean())]))
        s = trace.self_influence()
        self.assertEqual(s.shape, (n - L - P + 1,))
        scores = trace.timestep_scores()
        np.testing.assert_allclose(scores["tracin"], project_reference(s, n, 0, L + P)[0])
        np.testing.assert_allclose(scores["tracin_tgt"], project_reference(s, n, L, L + P)[2])

    def test_projection_preserves_small_scores_after_large_score_leaves(self):
        L, n = 4, 16
        for P in (1, 2):
            with self.subTest(P=P):
                s = np.ones(n - L - P + 1)
                s[[0, 7]] = float(2**54)
                np.testing.assert_array_equal(common.project_sum(s, L + P, n),
                                              project_reference(s, n, 0, L + P)[0])
                np.testing.assert_array_equal(common.project_targets_mean(s, L, P, n),
                                              project_reference(s, n, L, L + P)[2])

    def test_evaluator_projects_same_archived_scores_and_masks_before_metrics(self):
        n, L, P = 48, 4, 2
        y = np.arange(n) % 3 == 0
        scores = np.arange(1., n - L - P + 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artificial_s0.npz"
            np.savez(path, y=y, **{"L4P2|base": scores})
            with patch.object(evaluate, "metrics", wraps=evaluate.metrics) as metric, \
                    patch.object(torch.optim, "Adam", side_effect=AssertionError("evaluation trained")):
                rows, diag = evaluate.file_job(("synth", path))
        interior = slice(19, n - 19)
        self.assertEqual([(r["cfg"], r["train"], r["proj"]) for r in rows],
                         [("L4P2", "base", "win"), ("L4P2", "base", "tgt")])
        self.assertEqual(diag, [])
        self.assertEqual(metric.call_count, 2)
        ref_win, count_win, _ = project_reference(scores, n, 0, L + P)
        _, count_tgt, ref_tgt = project_reference(scores, n, L, L + P)
        np.testing.assert_array_equal(count_win[interior], L + P)
        np.testing.assert_array_equal(count_tgt[interior], P)
        for call, expected in zip(metric.call_args_list, (ref_win, ref_tgt)):
            np.testing.assert_array_equal(call.args[0], y[interior])
            np.testing.assert_array_equal(call.args[1], expected[interior])


if __name__ == "__main__":
    unittest.main()
