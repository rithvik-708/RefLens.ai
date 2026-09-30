import unittest
import numpy as np
from src.vision.homography import HomographyEstimator

class TestOffsideLineProjection(unittest.TestCase):
    def setUp(self):
        self.estimator = HomographyEstimator()
        # Create a real-ish perspective homography instead of a scaling matrix
        # E.g., points forming a trapezoid in image mapped to rectangle in pitch
        img_pts = [(200, 400), (1000, 400), (0, 700), (1200, 700)]
        ptc_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
        self.estimator.fit(img_pts, ptc_pts)

    def test_line_projection_perspective(self):
        # Generate pitch line at reference_x = 30.0 (off center, should have perspective slope)
        ref_x = 30.0
        y_pts = np.linspace(0, 68, 20)
        
        projected_x = []
        for y in y_pts:
            ix, iy = self.estimator.pitch_to_image((ref_x, float(y)))
            projected_x.append(ix)
            
        diff = max(projected_x) - min(projected_x)
        self.assertGreater(diff, 1.0, "The projected line should have perspective (not constant X in image space)")

if __name__ == '__main__':
    unittest.main()
