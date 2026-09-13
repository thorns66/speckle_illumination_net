"""CPU invariants for analysis, not synthetic evidence in experiment results."""
import unittest
import numpy as np
from tools.report_mean100_baseline_analysis import aggregate, paired, stability_values


class AnalysisTests(unittest.TestCase):
    def test_stability_mass_invariance(self):
        a=np.zeros((10,10,3,3));a[:,2,1,1]=np.arange(1,11)
        r=stability_values(a)
        self.assertAlmostEqual(r['shape_relative_dispersion'],0)
        self.assertAlmostEqual(r['pairwise_axial_w1_um'],0)
        self.assertEqual(r['pairs'],45)
        self.assertGreater(r['mass_cv'],0)

    def test_native_depth_distance(self):
        a=np.zeros((2,10,3,3));a[0,2,1,1]=1;a[1,3,1,1]=1
        self.assertAlmostEqual(stability_values(a)['pairwise_axial_w1_um'],10)

    def test_object_equal_not_subset_pooled(self):
        rows=[{'method':'m','sample':'a','metric':1}]*10+[{'method':'m','sample':'b','metric':3}]
        objects=aggregate(rows,('method','sample'),['metric'])
        self.assertEqual(aggregate(objects,('method',),['metric'])[0]['metric'],2)

    def test_paired_direction(self):
        rows=[dict(case='x',method='mean100_final800',error=1),dict(case='x',method='other',error=3)]
        result=paired(rows,('case',),['error'])
        self.assertEqual(result[0]['error'],-2)

    def test_zero_stability_rejected(self):
        with self.assertRaises(AssertionError):stability_values(np.zeros((2,10,3,3)))


if __name__=='__main__':unittest.main()
