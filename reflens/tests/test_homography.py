import unittest
import numpy as np
from src.vision.homography import HomographyEstimator

class TestHomography(unittest.TestCase):
    def setUp(self):
        self.estimator = HomographyEstimator()

    def test_identity_mapping(self):
        # A simple translation identity mapping
        # Suppose 1 pixel = 1 meter, starting at 0,0
        img_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        ptc_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        
        success = self.estimator.fit(img_pts, ptc_pts)
        self.assertTrue(success)
        
        px, py = self.estimator.image_to_pitch((50, 30))
        self.assertAlmostEqual(px, 50.0, places=2)
        self.assertAlmostEqual(py, 30.0, places=2)
        
        ix, iy = self.estimator.pitch_to_image((50, 30))
        self.assertAlmostEqual(ix, 50.0, places=2)
        self.assertAlmostEqual(iy, 30.0, places=2)

    def test_four_point_mapping(self):
        img_pts = [(100, 100), (200, 100), (50, 200), (250, 200)]
        ptc_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        
        success = self.estimator.fit(img_pts, ptc_pts)
        self.assertTrue(success)
        
        px, py = self.estimator.image_to_pitch((100, 100))
        self.assertAlmostEqual(px, 0.0, places=2)
        self.assertAlmostEqual(py, 0.0, places=2)

    def test_reprojection_error(self):
        img_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        ptc_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        self.estimator.fit(img_pts, ptc_pts)
        
        metrics = self.estimator.validate()
        self.assertAlmostEqual(metrics["mean_reprojection_error_m"], 0.0, places=2)
        
    def test_degenerate_points(self):
        # Collinear points
        img_pts = [(0, 0), (10, 0), (20, 0), (30, 0)]
        ptc_pts = [(0, 0), (10, 0), (20, 0), (30, 0)]
        
        success = self.estimator.fit(img_pts, ptc_pts)
        self.assertFalse(success)

    def test_missing_calibration(self):
        with self.assertRaises(ValueError):
            self.estimator.image_to_pitch((10, 10))

if __name__ == '__main__':
    unittest.main()
