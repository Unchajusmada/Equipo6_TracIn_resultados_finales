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

    def test_intervals_within_one_row_and_inclusive_boundaries(self):
        self.fixture([("A-1", "SMAP", 12, "[[4, 6], [0, 0], [2, 5], [11, 11], [4, 6]]"),
                      ("M-1", "MSL", 12, "[]")], {"A-1": 12, "M-1": 12, "extra": 12})
        self.assertEqual(db.nasa_names(), ["A-1", "M-1"])
        with patch.object(db, "smd_names", side_effect=AssertionError("NASA-only build")):
            db.build(block="nasa")
        self.assertEqual(sorted(p.name for p in self.out.glob("*.npz")), ["nasa_A-1.npz", "nasa_M-1.npz"])
        with np.load(self.out / "nasa_A-1.npz") as built:
            np.testing.assert_array_equal(np.flatnonzero(built["y"]), [0, 2, 3, 4, 5, 6, 11])
            self.assertEqual(built["spacecraft"].item(), "SMAP")
            self.assertEqual(built["y"].dtype, np.int8)
        with (self.out / "resumen.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([row["name"] for row in csv.DictReader(stream)], ["A-1", "M-1"])

    def test_same_id_and_length_do_not_identify_one_p2_sequence(self):
        self.fixture([("P-2", "SMAP", 8209, "[[5350, 6575]]"),
                      ("P-2", "SMAP", 8209, "[[5300, 6420]]")], {"P-2": 8209})
        self.assertEqual(db.build_nasa(), [])
        self.assertFalse((self.out / "nasa_P-2.npz").exists())
        db.build_nasa(first_row=True)
        with np.load(self.out / "nasa_P-2.npz") as built:
            np.testing.assert_array_equal(np.flatnonzero(built["y"]), np.arange(5350, 6576))

    def test_other_repeated_ids_require_verified_mapping(self):
        self.fixture([("A-1", "SMAP", 10, "[[0, 1]]"),
                      ("A-1", "SMAP", 10, "[[5, 6]]")], {"A-1": 10})
        for first_row in (False, True):
            with self.assertRaisesRegex(ValueError, "A-1.*ID repetido"):
                db.build_nasa(first_row=first_row)

    def test_invalid_intervals_rejected_even_in_historical_mode(self):
        for value in ("[[-1, 1]]", "[[2, 1]]", "[[0, 10]]", "[[0, 1.5]]", "[[0, True]]",
                      "[[1]]", "[1, 2]", "4", "["):
            with self.subTest(value=value):
                self.fixture([("P-2", "SMAP", 10, "[[0, 1]]"),
                              ("P-2", "SMAP", 10, value)], {"P-2": 10})
                with self.assertRaisesRegex(ValueError, "P-2.*(intervalo|anomaly_sequences)"):
                    db.build_nasa(first_row=True)

    def test_invalid_metadata(self):
        for row, message in [
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
        self.assertEqual(db.nasa_names(), [])
        with self.assertRaisesRegex(FileNotFoundError, "P-2"):
            db.nasa_names(first_row=True)
        self.fixture([("P-2", "SMAP", 10, "[]")], {"P-2": 11})
        with self.assertRaisesRegex(ValueError, "P-2.*10 pasos"):
            db.build_nasa(first_row=True)

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
            historical = runner.jobs("nasa", nasa_first_row=True)
        self.assertEqual(jobs, [(f"M-1_s{s}", ("nasa_M-1", s)) for s in range(5)])
        self.assertEqual(historical, [(f"{c}_s{s}", (f"nasa_{c}", s))
                                      for s in range(5) for c in ("M-1", "P-2")])


if __name__ == "__main__":
    unittest.main()
