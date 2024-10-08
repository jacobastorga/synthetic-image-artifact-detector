import numpy as np
from typing import Dict, Any
from .image_metrics import ImageMetrics
from .artifact_detector import ArtifactDetector
from .texture_analyzer import TextureAnalyzer

class VisualQualityScorer:
    """Calculates comprehensive visual fidelity ratings, flaw audits, and preference labels."""

    def __init__(self):
        self.metrics = ImageMetrics()
        self.detector = ArtifactDetector()
        self.texture = TextureAnalyzer()

    def evaluate_pair(self, orig_img: np.ndarray, synth_img: np.ndarray) -> Dict[str, Any]:
        """Audits a synthetic image against an original ground-truth photograph."""
        ssim = self.metrics.structural_similarity_index(orig_img, synth_img)
        psnr = self.metrics.peak_signal_noise_ratio(orig_img, synth_img)
        
        lighting_anomalies = self.detector.detect_lighting_discontinuities(synth_img)
        blur_anomalies = self.detector.detect_edge_blur(orig_img, synth_img)
        texture_data = self.texture.evaluate_texture_fidelity(orig_img, synth_img)

        # Score formulation out of 100.0
        base_score = ssim * 70.0 + min(30.0, psnr)
        
        # Penalties for detected artifacts
        flaw_penalty = len(lighting_anomalies) * 4.5 + len(blur_anomalies) * 3.5
        if texture_data["is_oversmoothed"]:
            flaw_penalty += 12.0

        final_score = max(0.0, min(100.0, base_score - flaw_penalty))

        # Determine letter grade
        if final_score >= 90.0:
            grade = "A (High Fidelity)"
        elif final_score >= 80.0:
            grade = "B (Minor Artifacts)"
        elif final_score >= 70.0:
            grade = "C (Noticeable Artifacts)"
        else:
            grade = "F (Severe Degradation)"

        return {
            "composite_score": round(final_score, 2),
            "grade": grade,
            "ssim": round(ssim, 4),
            "psnr_db": round(psnr, 2),
            "anomaly_summary": {
                "lighting_flaws": len(lighting_anomalies),
                "edge_blur_regions": len(blur_anomalies),
                "texture_oversmoothed": texture_data["is_oversmoothed"]
            },
            "anomalies": lighting_anomalies + blur_anomalies
        }
