import unittest
import numpy as np

class SpinachProtocolTest(unittest.TestCase):
 def test_frozen_frame_selection(self):
  selected=sorted((np.random.default_rng(20260909).choice(100,10,replace=False)+1).tolist())
  self.assertEqual(selected,[9,10,27,44,54,74,77,83,85,92])
  self.assertEqual(len(set(range(1,101))-set(selected)),90)

