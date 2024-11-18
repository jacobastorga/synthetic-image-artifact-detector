# Synthetic Image Artifact Detector & Multimodal QA Harness

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Annotation: CVAT & Labelbox](https://img.shields.io/badge/annotation-CVAT%20%2F%20Labelbox-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An automated computer vision quality assurance pipeline and defect auditing harness designed for multimodal AI evaluation, synthetic-vs-original image comparison, and training data annotation.

---

## Overview

State-of-the-art generative image models (diffusion, autoregressive, and GAN architectures) frequently introduce subtle artifacts that compromise downstream computer vision pipelines. These include:
- **Lighting Gradient Inconsistencies:** Unphysical specular blowout, non-Lambertian shadow boundaries, and reflection vector divergence.
- **Edge Softening & Blur:** High-frequency boundary attenuation across fine structural transitions.
- **Texture Energy Deficits:** Over-smoothing across high-entropy natural materials (fabrics, foliage, skin textures).

This repository provides deterministic algorithms for spotting, scoring, and bounding these defects, exporting ready-to-label annotation schemas for **CVAT** and **Labelbox**.

---

## Defect Taxonomy

| Defect Class | Detection Algorithm | Mathematical Metric | CVAT Tag |
|---|---|---|---|
| **Lighting Discontinuity** | 3x3 Sobel Operator Convolution | $\nabla I = \sqrt{G_x^2 + G_y^2} > \tau_{\text{light}}$ | `lighting_gradient_anomaly` |
| **Boundary Edge Blur** | Spatial Frequency Disparity | $\Delta G = \|\nabla I_{\text{ref}}\| - \|\nabla I_{\text{synth}}\|$ | `edge_blur_distortion` |
| **Texture Smoothing** | Local Window Variance | $\sigma_w^2 = \frac{1}{N}\sum (I(x,y) - \mu)^2$ | `texture_oversmoothed` |
| **Structural Distortion**| SSIM Decomposition | $\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | `structural_disparity` |

---

## Architecture

```
         Original Reference Image            Synthetic AI-Generated Image
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │      ImageMetrics       │  (MSE, PSNR, SSIM)
                       └───────────┬─────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌─────────────────────┐       ┌───────────────────────┐
         │  ArtifactDetector   │       │   TextureAnalyzer     │
         │  (Sobel Gradients)  │       │ (Spatial Local Energy)│
         └──────────┬──────────┘       └──────────┬────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │   VisualQualityScorer   │  (Fidelity & Grade)
                       └───────────┬─────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         CVAT 1.1 JSON Annotations      Labelbox v2 NDJSON Objects
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/jacobastorga/synthetic-image-artifact-detector.git
cd synthetic-image-artifact-detector
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/ -v
```

### Running Scene Audit
```bash
python examples/audit_synthetic_scene.py
```

---

## Author & Research Background
**Jacob Astorga**  
Computational Scientist & Multimodal AI Evaluation Specialist  
Portfolio: [github.com/jacobastorga](https://github.com/jacobastorga)
