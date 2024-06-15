import numpy as np
from typing import Tuple

class ImageMetrics:
    """Core mathematical similarity and fidelity metrics for image comparison."""

    @staticmethod
    def mean_squared_error(img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes pixel-wise Mean Squared Error between two images."""
        if img1.shape != img2.shape:
            raise ValueError(f"Image shape mismatch: {img1.shape} vs {img2.shape}")
        return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

    @staticmethod
    def peak_signal_noise_ratio(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
        """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels."""
        mse = ImageMetrics.mean_squared_error(img1, img2)
        if mse == 0:
            return float('inf')
        return float(20.0 * np.log10(max_val / np.sqrt(mse)))

    @staticmethod
    def structural_similarity_index(img1: np.ndarray, img2: np.ndarray,
                                    k1: float = 0.01, k2: float = 0.03,
                                    data_range: float = 255.0) -> float:
        """Computes global Structural Similarity Index (SSIM) based on luminance, contrast, and structure."""
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        mu_x = np.mean(x)
        mu_y = np.mean(y)

        sigma_x_sq = np.var(x)
        sigma_y_sq = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

        return float(numerator / denominator)
