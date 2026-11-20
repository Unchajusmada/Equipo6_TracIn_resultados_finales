import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("data_bench", ROOT / "src/data_bench.py")
db = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(db)


class NasaAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.raw = Path(self.tmp) / "raw"
        self.out = Path(self.tmp) / "processed"
        self.root = self.raw / "smap_msl"
        (self.root / "test").mkdir(parents=True)
        self.enterContext(patch.multiple(db, RAW=self.raw, OUT=self.out))

    def fixture(self, rows, lengths):
        with (self.root / "labeled_anomalies.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["chan_id", "spacecraft", "num_values", "anomaly_sequences"])
            writer.writerows(rows)
        for channel, length in lengths.items():
            np.save(self.root / "test" / f"{channel}.npy", np.zeros((length, 2), dtype=np.float32))

    def test_union_overlap_disjoint_repeated_and_inclusive_boundaries(self):
        self.fixture([("P-2", "SMAP", 12, "[[4, 6], [0, 0]]"),
                      ("P-2", "SMAP", 12, "[[2, 5], [11, 11]]"),
                      ("P-2", "SMAP", 12, "[[4, 6]]"),
                      ("M-1", "MSL", 12, "[]")], {"P-2": 12, "M-1": 12, "extra": 12})
        self.assertEqual(db.nasa_names(), ["M-1", "P-2"])
        with patch.object(db, "smd_names", side_effect=AssertionError("NASA-only build")):
            db.build(block="nasa")
        self.assertEqual(sorted(p.name for p in self.out.glob("*.npz")), ["nasa_M-1.npz", "nasa_P-2.npz"])
        with np.load(self.out / "nasa_P-2.npz") as built:
            np.testing.assert_array_equal(np.flatnonzero(built["y"]), [0, 2, 3, 4, 5, 6, 11])
            self.assertEqual(built["spacecraft"].item(), "SMAP")
            self.assertEqual(built["y"].dtype, np.int8)
        with (self.out / "resumen.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([row["name"] for row in csv.DictReader(stream)], ["M-1", "P-2"])

    def test_p2_union_vs_historical(self):
        self.fixture([("P-2", "SMAP", 8209, "[[5350, 6575]]"),
                      ("P-2", "SMAP", 8209, "[[5300, 6420]]")], {"P-2": 8209})
        db.build_nasa()
        with np.load(self.out / "nasa_P-2.npz") as built:
            corrected = built["y"]
        historical = self.out.with_name("historical")
        with patch.object(db, "OUT", historical):
            db.build_nasa(first_row=True)
        with np.load(historical / "nasa_P-2.npz") as built:
            previous = built["y"]
        self.assertEqual(int(corrected.sum()), 1276)
        self.assertEqual(int(previous.sum()), 1226)
        np.testing.assert_array_equal(np.flatnonzero(corrected - previous), np.arange(5300, 5350))

    def test_invalid_intervals_rejected_even_in_historical_mode(self):
        for value in ("[[-1, 1]]", "[[2, 1]]", "[[0, 10]]", "[[0, 1.5]]", "[[0, True]]",
                      "[[1]]", "[1, 2]", "4", "["):
            with self.subTest(value=value):
                self.fixture([("P-2", "SMAP", 10, "[[0, 1]]"),
                              ("P-2", "SMAP", 10, value)], {"P-2": 10})
                with self.assertRaisesRegex(ValueError, "P-2.*(intervalo|anomaly_sequences)"):
                    db.build_nasa(first_row=True)

    def test_inconsistent_and_invalid_metadata(self):
        for row, message in [
            (("P-2", "MSL", 10, "[]"), "incompatible"),
            (("P-2", "SMAP", 11, "[]"), "incompatible"),
            (("P-2", "OTHER", 10, "[]"), "inválido"),
            (("P-2", "SMAP", "10.5", "[]"), "inválido"),
            (("P-2", "SMAP", 0, "[]"), "inválido"),
            (("../P-2", "SMAP", 10, "[]"), "chan_id inválido"),
            (("", "SMAP", 10, "[]"), "chan_id inválido"),
        ]:
            with self.subTest(row=row):
                self.fixture([("P-2", "SMAP", 10, "[]"), row], {"P-2": 10})
                with self.assertRaisesRegex(ValueError, message):
                    db.nasa_names()

    def test_missing_array_and_length_mismatch(self):
        self.fixture([("P-2", "SMAP", 10, "[]")], {})
        with self.assertRaisesRegex(FileNotFoundError, "P-2"):
            db.nasa_names()
        self.fixture([("P-2", "SMAP", 10, "[]")], {"P-2": 11})
        with self.assertRaisesRegex(ValueError, "P-2.*10 pasos"):
            db.build_nasa()

    def test_nasa_runner_schedules_each_channel_once_per_seed(self):
        self.fixture([("P-2", "SMAP", 10, "[[0, 1]]"),
                      ("P-2", "SMAP", 10, "[[5, 6]]"),
                      ("M-1", "MSL", 10, "[[8, 9]]")], {"P-2": 10, "M-1": 10})
        core = ModuleType("core")
        core.ROOT, core.RES = ROOT / "src", Path(self.tmp)
        core.all_variants = core.save = self.fail
        spec = importlib.util.spec_from_file_location("run_bench", ROOT / "src/final/run_bench.py")
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"core": core, "data_bench": db}), patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(runner)
            jobs = runner.jobs("nasa")
        self.assertEqual(len(jobs), 10)
        self.assertEqual(dict(jobs), {f"{c}_s{s}": (f"nasa_{c}", s) for s in range(5) for c in ("M-1", "P-2")})


if __name__ == "__main__":
    unittest.main()
