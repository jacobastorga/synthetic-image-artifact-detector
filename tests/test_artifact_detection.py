import unittest
import numpy as np
from src.image_metrics import ImageMetrics
from src.artifact_detector import ArtifactDetector
from src.texture_analyzer import TextureAnalyzer
from src.annotation_exporter import AnnotationExporter
from src.quality_scorer import VisualQualityScorer

class TestArtifactDetectionSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic 64x64 test images
        self.orig_img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        self.identical_img = self.orig_img.copy()
        
        # Distorted image with blur and lighting spike
        self.distorted_img = self.orig_img.copy().astype(np.float64)
        self.distorted_img[20:30, 20:30] = 255.0  # Specular hotspot
        self.distorted_img = np.clip(self.distorted_img, 0, 255).astype(np.uint8)

    def test_identical_metrics(self):
        mse = ImageMetrics.mean_squared_error(self.orig_img, self.identical_img)
        self.assertEqual(mse, 0.0)
        ssim = ImageMetrics.structural_similarity_index(self.orig_img, self.identical_img)
        self.assertAlmostEqual(ssim, 1.0, places=4)

    def test_distorted_metrics(self):
        mse = ImageMetrics.mean_squared_error(self.orig_img, self.distorted_img)
        self.assertGreater(mse, 0.0)
        ssim = ImageMetrics.structural_similarity_index(self.orig_img, self.distorted_img)
        self.assertLess(ssim, 1.0)

    def test_lighting_discontinuity_detection(self):
        detector = ArtifactDetector()
        anomalies = detector.detect_lighting_discontinuities(self.distorted_img, threshold=40.0)
        self.assertIsInstance(anomalies, list)

    def test_cvat_export_schema(self):
        sample_anomalies = [
            {"type": "lighting_gradient_anomaly", "bbox": [10, 10, 8, 8], "severity": "critical", "peak_gradient": 150.0}
        ]
        cvat_payload = AnnotationExporter.to_cvat_json(1, "test.png", 64, 64, sample_anomalies)
        self.assertEqual(cvat_payload["version"], 1.1)
        self.assertEqual(len(cvat_payload["annotations"]), 1)
        self.assertEqual(cvat_payload["annotations"][0]["label"], "lighting_gradient_anomaly")

    def test_quality_scorer_evaluation(self):
        scorer = VisualQualityScorer()
        res = scorer.evaluate_pair(self.orig_img, self.distorted_img)
        self.assertIn("composite_score", res)
        self.assertIn("grade", res)
        self.assertLessEqual(res["composite_score"], 100.0)

if __name__ == "__main__":
    unittest.main()
