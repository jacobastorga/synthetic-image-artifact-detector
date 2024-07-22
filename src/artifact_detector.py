import numpy as np
from typing import List, Dict, Any, Tuple

class ArtifactDetector:
    """Detects lighting gradient anomalies, boundary edge blurring, and texture artifacts."""

    @staticmethod
    def compute_sobel_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates horizontal and vertical spatial derivatives using a 3x3 Sobel kernel."""
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2).astype(np.float64)
        else:
            gray = img.astype(np.float64)

        # 3x3 Sobel operator kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        h, w = gray.shape
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        # Spatial convolution over valid region
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = gray[i-1:i+2, j-1:j+2]
                gx[i, j] = np.sum(patch * kx)
                gy[i, j] = np.sum(patch * ky)

        magnitude = np.sqrt(gx**2 + gy**2)
        return gx, gy, magnitude

    def detect_lighting_discontinuities(self, img: np.ndarray, threshold: float = 85.0) -> List[Dict[str, Any]]:
        """Identifies harsh, unphysical illumination gradients and specular clipping."""
        _, _, mag = self.compute_sobel_gradients(img)
        anomalies = []
        h, w = mag.shape
        block_size = max(8, min(h, w) // 8)

        for r in range(0, h - block_size, block_size):
            for c in range(0, w - block_size, block_size):
                block = mag[r:r+block_size, c:c+block_size]
                max_grad = float(np.max(block))
                mean_grad = float(np.mean(block))
                if max_grad > threshold and (max_grad / (mean_grad + 1e-5)) > 4.5:
                    anomalies.append({
                        "type": "lighting_gradient_anomaly",
                        "bbox": [c, r, block_size, block_size],
                        "peak_gradient": round(max_grad, 2),
                        "severity": "critical" if max_grad > 140.0 else "moderate"
                    })
        return anomalies

    def detect_edge_blur(self, orig_img: np.ndarray, synth_img: np.ndarray, threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Flags unnatural softening or loss of structural high-frequency edge definition."""
        _, _, orig_mag = self.compute_sobel_gradients(orig_img)
        _, _, synth_mag = self.compute_sobel_gradients(synth_img)

        edge_diff = orig_mag - synth_mag
        h, w = edge_diff.shape
        block_size = max(8, min(h, w) // 8)
        blur_regions = []

        for r in range(0, h - block_size, block_size):
            for c in range(0, w - block_size, block_size):
                block_loss = float(np.mean(edge_diff[r:r+block_size, c:c+block_size]))
                if block_loss > threshold:
                    blur_regions.append({
                        "type": "edge_blur_distortion",
                        "bbox": [c, r, block_size, block_size],
                        "gradient_loss": round(block_loss, 2),
                        "severity": "moderate" if block_loss < 60.0 else "critical"
                    })
        return blur_regions
