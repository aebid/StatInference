#!/usr/bin/env python3
"""Member uncertainties (dc_make/member_reduction.py): the configuration, the histogram names
and the envelope / Hessian reductions, bin by bin. No ROOT needed."""

import math
import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from StatInference.dc_make.member_reduction import MemberReduction  # noqa: E402

QCD = [
    "muR0p5_muF0p5",
    "muR0p5_muF1",
    "muR1_muF0p5",
    "muR1_muF2",
    "muR2_muF1",
    "muR2_muF2",
]


class ConfigTest(unittest.TestCase):
    def test_range_and_alpha_s(self):
        m = MemberReduction.fromConfig(
            {
                "reduction": "hessian",
                "reference": 0,
                "names": {"first": 1, "last": 100},
                "alpha_s": [101, 102],
            },
            "pdf_shape_TT",
        )
        self.assertEqual(m.names, [str(k) for k in range(1, 101)])
        self.assertEqual(m.reference, "0")
        self.assertEqual(m.alpha_s, ["101", "102"])
        self.assertEqual(len(m.all_members()), 103)

    def test_histogram_names(self):
        m = MemberReduction.fromConfig({"reduction": "envelope", "names": QCD})
        self.assertEqual(
            m.histogram_name("QCDscale_shape_TT", "TT", "muR2_muF1"),
            "QCDscale_shape_TT_muR2_muF1",
        )
        sig = MemberReduction.fromConfig(
            {
                "reduction": "envelope",
                "names": QCD,
                "histogram": "QCDscale_shape_{process}_{member}",
            }
        )
        self.assertEqual(
            sig.histogram_name("QCDscale_shape_X", "X_500", "muR2_muF2"),
            "QCDscale_shape_X_500_muR2_muF2",
        )

    def test_invalid(self):
        bad = [
            {"reduction": "max", "names": QCD},
            {"reduction": "envelope"},
            {"reduction": "envelope", "names": []},
            {"reduction": "envelope", "names": QCD, "alpha_s": [1, 2]},
            {"reduction": "hessian", "names": [1, 2], "alpha_s": [3]},
            {"reduction": "hessian", "names": [1, 2], "reference": 2},
            {"reduction": "hessian", "names": {"first": 5, "last": 1}},
            {"reduction": "hessian", "names": [1], "histogram": "x_{name}"},
            {"reduction": "hessian", "names": [1], "typo": 1},
        ]
        for entry in bad:
            with self.assertRaises(ValueError, msg=str(entry)):
                MemberReduction.fromConfig(entry, "u")


class ReduceTest(unittest.TestCase):
    def test_envelope(self):
        m = MemberReduction("envelope", ["a", "b"])
        up, down = m.reduce([10.0, 5.0], {"a": [12.0, 4.0], "b": [9.0, 4.5]})
        self.assertEqual(up, [12.0, 5.0])  # never below the nominal
        self.assertEqual(down, [9.0, 4.0])

    def test_hessian(self):
        m = MemberReduction("hessian", ["1", "2"], reference="0", alpha_s=["3", "4"])
        members = {"0": [10.0], "1": [13.0], "2": [6.0], "3": [9.0], "4": [11.0]}
        up, down = m.reduce([10.5], members)
        d = math.sqrt(3**2 + 4**2 + 1**2)
        self.assertAlmostEqual(up[0], 10.5 + d)
        self.assertAlmostEqual(down[0], 10.5 - d)

    def test_hessian_down_clamped(self):
        m = MemberReduction("hessian", ["1"])
        up, down = m.reduce([1.0], {"1": [4.0]})
        self.assertEqual((up, down), ([4.0], [0.0]))

    def test_nonfinite_member_raises(self):
        m = MemberReduction("hessian", ["1"], alpha_s=["101", "102"])
        with self.assertRaisesRegex(ValueError, "member 101 is not finite"):
            m.reduce([1.0], {"1": [1.0], "101": [float("nan")], "102": [1.0]})


if __name__ == "__main__":
    unittest.main()
