import numpy as np
from typing import Dict, Any

class TextureAnalyzer:
    """Analyzes texture coherence and high-frequency spectral distribution."""

    @staticmethod
    def compute_local_variance(img: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Computes local spatial variance as a proxy for surface texture energy."""
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2).astype(np.float64)
        else:
            gray = img.astype(np.float64)

        h, w = gray.shape
        pad = window_size // 2
        var_map = np.zeros_like(gray)

        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                patch = gray[i-pad:i+pad+1, j-pad:j+pad+1]
                var_map[i, j] = np.var(patch)

        return var_map

    def evaluate_texture_fidelity(self, orig_img: np.ndarray, synth_img: np.ndarray) -> Dict[str, Any]:
        """Compares texture energy distribution between original and synthetic scenes."""
        orig_var = self.compute_local_variance(orig_img)
        synth_var = self.compute_local_variance(synth_img)

        mean_orig_var = float(np.mean(orig_var))
        mean_synth_var = float(np.mean(synth_var))

        ratio = (mean_synth_var / (mean_orig_var + 1e-6))
        texture_deficit = max(0.0, 1.0 - ratio)

        return {
            "orig_texture_energy": round(mean_orig_var, 2),
            "synth_texture_energy": round(mean_synth_var, 2),
            "energy_ratio": round(ratio, 4),
            "is_oversmoothed": ratio < 0.65,
            "texture_deficit_percent": round(texture_deficit * 100.0, 2)
        }
