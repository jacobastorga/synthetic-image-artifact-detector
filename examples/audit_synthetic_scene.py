import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quality_scorer import VisualQualityScorer
from src.annotation_exporter import AnnotationExporter

def run_scene_audit():
    np.random.seed(101)
    print("Generating simulated original photograph vs. synthetic diffusion render...")
    
    # 1. Base reference pattern (original scene)
    h, w = 128, 128
    y, x = np.ogrid[:h, :w]
    orig_scene = (128 + 60 * np.sin(x / 10.0) + 40 * np.cos(y / 10.0)).astype(np.uint8)

    # 2. Synthetic scene with typical diffusion artifacts (texture smoothing + localized lighting blowout)
    synth_scene = orig_scene.copy().astype(np.float64)
    synth_scene[30:50, 30:50] = 255.0  # Unphysical specular blowout
    synth_scene[80:110, 80:110] = 120.0  # Texture obliteration / over-smoothing
    synth_scene = np.clip(synth_scene, 0, 255).astype(np.uint8)

    # 3. Perform automated visual QA audit
    scorer = VisualQualityScorer()
    eval_result = scorer.evaluate_pair(orig_scene, synth_scene)

    print("\n=== Multimodal Visual QA Audit Report ===")
    print(f"Overall Quality Rating: {eval_result['composite_score']} / 100.0")
    print(f"Quality Classification: {eval_result['grade']}")
    print(f"Structural Similarity (SSIM): {eval_result['ssim']}")
    print(f"PSNR: {eval_result['psnr_db']} dB")
    print(f"Lighting Discontinuities: {eval_result['anomaly_summary']['lighting_flaws']}")
    print(f"Boundary Blur Artifacts:  {eval_result['anomaly_summary']['edge_blur_regions']}")

    # 4. Export to CVAT 1.1 annotation format
    cvat_output = AnnotationExporter.to_cvat_json(
        image_id=402,
        image_name="diffusion_test_frame_0402.png",
        width=w,
        height=h,
        anomalies=eval_result["anomalies"]
    )
    print("\nExported CVAT Annotation Schema (Top Item):")
    if cvat_output["annotations"]:
        print(json.dumps(cvat_output["annotations"][0], indent=2))

if __name__ == "__main__":
    run_scene_audit()
