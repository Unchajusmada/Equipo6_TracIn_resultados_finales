import sys
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import mse_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecaster
from final import core
import lab


L, P, HIDDEN, SEED = 2, 2, 3, 17
SERIES = np.array([[-1, 0], [0, 1], [2, -1], [1, 2], [-2, 1], [1, -1]], dtype=np.float32)
M = len(SERIES) - L - P + 1
TOL = dict(rtol=2e-6, atol=1e-12)
CFG = lab.Cfg(lr=1e-6, batch=2, max_epochs=3, hidden=HIDDEN, norm="none")


def gradients_by_window(model, series, lookback, horizon, params):
    """Reference using explicit time slices and one backward calculation per window."""
    gradients = []
    for start in range(len(series) - lookback - horizon + 1):
        split = start + lookback
        prediction, _ = model(series[start:split][None])
        target = series[split:split + horizon][None]
        gradients.append(torch.autograd.grad(mse_loss(prediction, target), params))
    return tuple(torch.stack(values) for values in zip(*gradients))


def squared_norm(gradients):
    return sum(g.flatten(1).square().sum(1) for g in gradients)


class TracInMathTests(unittest.TestCase):
    def setUp(self):
        numpy_state = np.random.get_state()
        self.addCleanup(np.random.set_state, numpy_state)
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(SEED)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def test_closed_form_head_weight_bias_and_mse(self):
        for channels, horizon, lookback, hidden in ((1, 1, 1, 2), (3, 1, 2, 4), (2, 3, 2, 3)):
            series = torch.randn(lookback + horizon + 2, channels, dtype=torch.float64)
            cases = (
                (forecaster.LSTMForecaster(channels, lookback, horizon, hidden),
                 forecaster._checkpoint_stats),
                (lab.LSTM(channels, horizon, hidden), lab._stats),
            )
            for model, stats in cases:
                with self.subTest(model=type(model).__name__, channels=channels, horizon=horizon):
                    model.double()
                    weight, bias = gradients_by_window(
                        model, series, lookback, horizon, tuple(model.head.parameters()))
                    d = horizon * channels
                    losses, states, residuals = [], [], []
                    with torch.no_grad():
                        for i in range(len(weight)):
                            pred, state = model(series[i:i + lookback][None])
                            target = series[i + lookback:i + lookback + horizon]
                            residuals.append((pred[0] - target).reshape(-1))
                            states.append(state[0])
                            losses.append(mse_loss(pred[0], target))
                    h = torch.stack(states)
                    g = (2 / d) * torch.stack(residuals)
                    precise = dict(rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(weight, g[:, :, None] * h[:, None, :], **precise)
                    torch.testing.assert_close(bias, g, **precise)
                    loss, hsq = stats(model, series, len(weight), lookback, horizon, 2)
                    torch.testing.assert_close(loss, torch.stack(losses), **precise)
                    torch.testing.assert_close(hsq, h.square().sum(1), **precise)
                    expected = squared_norm((weight, bias)).numpy()
                    trace = forecaster.Trace(lookback, horizon, channels, len(series),
                                             np.ones(1), loss.numpy()[None], hsq.numpy()[None],
                                             np.array([float(loss.mean())]))
                    np.testing.assert_allclose(trace.self_influence(), expected, **precise)

    def test_full_vmap_gradients_match_autograd_per_parameter(self):
        model = lab.LSTM(SERIES.shape[1], P, HIDDEN).double()
        series = torch.as_tensor(SERIES, dtype=torch.float64)
        params = dict(model.named_parameters())
        expected = dict(zip(params, gradients_by_window(model, series, L, P, tuple(params.values()))))
        batches = []

        def capture_vmap(*args, **kwargs):
            vectorized = torch.func.vmap(*args, **kwargs)

            def capture(*inputs):
                batch = vectorized(*inputs)
                batches.append(batch)
                return batch

            return capture

        with patch.object(lab, "vmap", side_effect=capture_vmap):
            actual_norm = lab._per_sample_sqgrad(model, series, M, L, P, bs=2)
        for name, param in params.items():
            actual = torch.cat([batch[name] for batch in batches])
            self.assertEqual(actual.shape, (M, *param.shape))
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected[name], rtol=1e-11, atol=1e-12)
        head = squared_norm([g for name, g in expected.items() if name.startswith("head.")])
        recurrent = squared_norm([g for name, g in expected.items() if not name.startswith("head.")])
        self.assertEqual(actual_norm.shape, (M,))
        self.assertTrue(torch.isfinite(actual_norm).all())
        self.assertTrue((head > 0).all() and (recurrent > 0).all())
        torch.testing.assert_close(actual_norm, head + recurrent, rtol=1e-11, atol=1e-12)

    def observe_epochs(self, cfg, *, trace=False, weights=None):
        """Observe real Adam steps and compute independent gradients at epoch boundaries."""
        stats_module = forecaster if trace else lab
        stats_name = "_checkpoint_stats" if trace else "_stats"
        original_stats = getattr(stats_module, stats_name)
        adam = torch.optim.Adam
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
        step_rates, epochs = [], []
        shuffle = torch.Generator().manual_seed(SEED)
        audit_weights = torch.ones(M) if weights is None else torch.as_tensor(weights)

        def optimizer(*args, **kwargs):
            opt = adam(*args, **kwargs)
            opt.register_step_pre_hook(lambda opt, *_: step_rates.append(opt.param_groups[0]["lr"]))
            return opt

        def short_plateau(opt, *, factor, patience):
            self.assertEqual((factor, patience), (0.9, 3))
            # A real reduction must fit within the three-epoch audit budget.
            return plateau(opt, factor=factor, patience=0)

        def checkpoint(model, series, count, lookback, horizon, bs=8192):
            params = model.parameters() if cfg.grad == "full" else model.head.parameters()
            norm = squared_norm(gradients_by_window(model, series, lookback, horizon, tuple(params)))
            batches = torch.randperm(M, generator=shuffle).split(cfg.batch)
            expected_steps = sum(int(audit_weights[batch].sum() != 0) for batch in batches)
            self.assertEqual(step_rates, [step_rates[0]] * expected_steps)
            loss, hsq = original_stats(model, series, count, lookback, horizon, bs)
            epochs.append((step_rates[0], norm.numpy(), loss.numpy()))
            step_rates.clear()
            return loss, hsq

        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.optim, "Adam", side_effect=optimizer))
            scheduler = stack.enter_context(patch.object(
                torch.optim.lr_scheduler, "ReduceLROnPlateau", side_effect=short_plateau))
            stack.enter_context(patch.object(stats_module, stats_name, side_effect=checkpoint))
            if trace:
                out = forecaster.train_trace(
                    SERIES, L, P, SEED, lr=cfg.lr, batch=cfg.batch, max_epochs=cfg.max_epochs,
                    patience=cfg.patience, weights=weights, model_kw={"hidden": cfg.hidden})
            else:
                out = lab.run(SERIES, L, P, SEED, cfg, weights=weights, keep_g2=True)
        self.assertEqual(scheduler.call_count, int(trace or cfg.sched))
        self.assertEqual(step_rates, [])
        rates, norms, losses = map(np.stack, zip(*epochs))
        expected = rates[:, None] * norms if cfg.lr_weight else norms
        if trace:
            np.testing.assert_array_equal(out.lrs, rates)
            np.testing.assert_array_equal(out.loss, losses)
            np.testing.assert_allclose(out.self_influence(last_only=True), expected[-1], **TOL)
            np.testing.assert_allclose(out.self_influence(), expected.sum(0), **TOL)
        else:
            self.assertEqual(out["epochs"], len(expected))
            self.assertEqual(out["g2"].shape, (out["epochs"], M))
            np.testing.assert_allclose(out["g2"], expected, **TOL)
            np.testing.assert_allclose(out["si"], expected.sum(0), **TOL)
            np.testing.assert_allclose(out["si"], out["g2"].sum(0), **TOL)
            np.testing.assert_array_equal(out["loss"], losses)
            np.testing.assert_array_equal(out["loss_final"], losses[-1])
        return rates, out

    def test_checkpoint_rates_and_contributions(self):
        for sched, grad, lr_weight in ((True, "full", True), (False, "last", True), (True, "last", False)):
            with self.subTest(sched=sched, grad=grad, lr_weight=lr_weight):
                cfg = replace(CFG, sched=sched, grad=grad, lr_weight=lr_weight, batch=1)
                rates, out = self.observe_epochs(cfg, weights=np.array([0, 1, 1]))
                np.testing.assert_array_equal(rates, [cfg.lr, cfg.lr, cfg.lr * (0.9 if sched else 1)])
                self.assertTrue((out["g2"][:, 0] > 0).all())
        rates, _ = self.observe_epochs(CFG, trace=True)
        np.testing.assert_array_equal(rates, [CFG.lr, CFG.lr, CFG.lr * 0.9])

    def test_early_stopping_keeps_last_executed_epoch(self):
        cfg = replace(CFG, patience=1)
        for trace in (False, True):
            with self.subTest(trace=trace):
                rates, out = self.observe_epochs(cfg, trace=trace)
                np.testing.assert_array_equal(rates, [cfg.lr, cfg.lr])
                loss = out.loss if trace else out["loss"]
                final = out.loss_final() if trace else out["loss_final"]
                self.assertFalse(np.array_equal(loss[0], loss[-1]))
                np.testing.assert_array_equal(final, loss[-1])

    def test_cpu_reproducibility_and_core_baseline(self):
        cfg = replace(CFG, grad="full")
        detailed = lab.run(SERIES, L, P, SEED, cfg, keep_g2=True)
        repeat = lab.run(SERIES, L, P, SEED, cfg, keep_g2=True)
        for key in detailed:
            np.testing.assert_array_equal(detailed[key], repeat[key])
        plain = lab.run(SERIES, L, P, SEED, cfg)
        self.assertEqual(plain.keys(), detailed.keys() - {"g2"})
        for key in plain:
            np.testing.assert_array_equal(plain[key], detailed[key])
        final = core.all_variants(SERIES, None, L, P, SEED, cfg, modes=())
        np.testing.assert_array_equal(final["base"], plain["si"])
        np.testing.assert_array_equal(final["pd"], plain["loss_final"])
        self.assertEqual(final["diag_epochs_base"], plain["epochs"])

        traces = [forecaster.train_trace(SERIES, L, P, SEED, max_epochs=2,
                                         batch=2, model_kw={"hidden": HIDDEN}) for _ in range(2)]
        for name in ("lrs", "loss", "hsq", "train_loss"):
            np.testing.assert_array_equal(getattr(traces[0], name), getattr(traces[1], name))
        np.testing.assert_array_equal(traces[0].self_influence(), traces[1].self_influence())

    def test_lab_rejects_zero_weight_monitored_set(self):
        cases = ((CFG, np.zeros(M)),
                 (replace(CFG, val_frac=0.5), np.array([1, 1, 0])))
        for cfg, weights in cases:
            with self.subTest(val_frac=cfg.val_frac), patch.object(torch.optim, "Adam") as adam:
                with self.assertRaisesRegex(ValueError, "conjunto monitoreado tiene peso total cero"):
                    lab.run(SERIES, L, P, SEED, cfg, weights=weights)
                adam.assert_not_called()

    def test_forecaster_rejects_zero_weight_monitored_set(self):
        with patch.object(torch.optim, "Adam") as adam:
            with self.assertRaisesRegex(ValueError, "conjunto monitoreado tiene peso total cero"):
                forecaster.train_trace(SERIES, L, P, SEED, weights=np.zeros(M),
                                       max_epochs=1, model_kw={"hidden": HIDDEN})
            adam.assert_not_called()


if __name__ == "__main__":
    unittest.main()
