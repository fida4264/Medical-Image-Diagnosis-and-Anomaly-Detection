# Module 15.4 — Heatmap Visualization & Clinical Overlay Pipeline

**Medical AI Diagnosis and Anomaly Detection Platform**
**Phase 4 — Explainable AI (XAI)**
**Status:** ✅ COMPLETE · 117/117 Tests Passed (17 Module 15.4 + 18 Module 15.3 + 30 Module 15.2 + 52 Module 15.1)

---

## 1. Executive Summary

Module 15.4 delivers the production **Heatmap Visualization & Clinical Overlay Pipeline** for the Medical AI Platform. It transforms 2D normalized Grad-CAM heatmaps ($[0.0, 1.0]$) and original input images into clinically readable overlay image artifacts, binary attention masks, geometric bounding regions, quality metrics, and exported disk files (PNG/JPEG/JSON).

> 🚫 **Scope Boundary:** This module handles **pure visual rendering** and **geometric region extraction**. It does NOT produce clinical text explanations or medical diagnostic text generation (which belong to Module 15.5).

---

## 2. Architecture & Package Structure

```
medical_ai_engine/
└── explainability/
    └── visualization/
        ├── __init__.py                     ← Package exports
        ├── heatmap_renderer.py             ← HeatmapRenderer (Primary Facade)
        ├── overlay_composer.py             ← OverlayComposer (Master Pipeline Orchestrator)
        ├── color_mapper.py                 ← ColorMapper (inferno, jet, turbo, viridis, magma)
        ├── alpha_blender.py                ← AlphaBlender (Transparency blending [0.1, 0.9])
        ├── attention_mask_generator.py     ← AttentionMaskGenerator (Binary mask thresholding)
        ├── bounding_region_detector.py     ← BoundingRegionDetector (Bounding box & polygon bounds)
        ├── region_extractor.py             ← RegionExtractor (Peak, centroid, area ranking)
        ├── visualization_quality_checker.py← VisualizationQualityChecker (5 quality metrics)
        ├── image_exporter.py               ← ImageExporter (PNG/JPEG/JSON disk exporter)
        ├── overlay_metadata.py             ← OverlayMetadata dataclass
        ├── overlay_configuration.py        ← OverlayConfiguration dataclass
        └── overlay_exceptions.py          ← Typed exception hierarchy
```

---

## 3. Visualization Pipeline Flow

```
Original Image  +  Normalized 2D Heatmap [0, 1]  +  ExplainabilityContext
                                │
                                ▼
                       HeatmapRenderer (Facade)
                                │
                                ▼
                        OverlayComposer
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
  ColorMapper             AlphaBlender         AttentionMaskGenerator
  (inferno/jet/...)     (alpha=0.5 blend)        (binary threshold 0.2)
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                                ▼
                    BoundingRegionDetector & RegionExtractor
                    (bounding boxes, centroids, polygons, area)
                                │
                                ▼
                   VisualizationQualityChecker
                   (coverage, sharpness, contrast, visibility, entropy)
                                │
                                ▼
                          ImageExporter
              (overlay.png, heatmap.png, mask.png, metadata.json)
                                │
                                ▼
                   Updated ExplainabilityContext
        (overlay_image, important_regions, visualization_metrics, paths)
```

---

## 4. Key Components & Capabilities

### 4.1 Color Mapping (`ColorMapper`)

Converts a 2D float activation map into a 3D RGB image array using scientific colormaps:
- **`inferno`** (Default — recommended for medical visualization; high perceptual contrast)
- **`jet`** (Legacy blue-to-red rainbow palette)
- **`turbo`** (Improved smooth rainbow colormap)
- **`viridis`** (Perceptually uniform green-yellow palette)
- **`magma`** (High contrast dark-to-light palette)

Primary implementations use Matplotlib/OpenCV colormaps with pure NumPy synthetic look-up tables as a zero-dependency fallback.

### 4.2 Alpha Blending (`AlphaBlender`)

Blends the original medical image with the RGB colored heatmap:
$$I_{\text{overlay}} = \alpha \cdot I_{\text{heatmap}} + (1 - \alpha) \cdot I_{\text{original}}$$

- **Configurable Alpha Ratio:** Range $[0.1, 0.9]$ (Default: `0.5`).
- Handles dimension mismatches via bilinear spatial resizing.

### 4.3 Attention Mask Generation (`AttentionMaskGenerator`)

Applies intensity thresholding at a cutoff ratio (default: `threshold = 0.2`) to generate binary 2D attention masks (`255` where $\text{heatmap} \ge \text{threshold}$, `0` elsewhere) returned as PIL `L`-mode Images and uint8 NumPy arrays.

### 4.4 Geometric Bounding Regions & Region Extraction

- **`BoundingRegionDetector`**: Detects connected component bounding boxes `[x_min, y_min, x_max, y_max]` and contour polygons without disease interpretation. Optionally annotates bounding boxes onto overlay images.
- **`RegionExtractor`**: Ranks high-activation regions by peak activation, centroid `[x_c, y_c]`, pixel count, and area ratio.

### 4.5 Visualization Quality Metrics (`VisualizationQualityChecker`)

Computes 5 quantitative quality metrics for every generated overlay:
1. **Overlay Coverage Ratio:** Fraction of image covered by active heatmap ($\text{heatmap} \ge \text{threshold}$).
2. **Heatmap Sharpness:** Spatial gradient variance of the activation map.
3. **Activation Contrast Ratio:** $\frac{\max - \min}{\text{mean} + \epsilon}$.
4. **Overlay Visibility Contrast:** RMS difference between original image and blended overlay.
5. **Color Entropy:** Channel standard deviation across RGB channels.

### 4.6 Disk Artifact Export (`ImageExporter`)

Saves output artifacts to disk and populates `ExplainabilityContext`:
- `overlay.png` / `overlay.jpg`
- `heatmap.png` / `heatmap.jpg`
- `attention_mask.png` / `attention_mask.jpg`
- `overlay_metadata.json`

---

## 5. Usage Example

```python
from medical_ai_engine.explainability.visualization import HeatmapRenderer, OverlayConfiguration
from medical_ai_engine.explainability import ExplainabilityContext

# Instantiation
renderer = HeatmapRenderer(config=OverlayConfiguration(colormap="inferno", alpha=0.5))

ctx = ExplainabilityContext(request_id="req-101", session_id="sess-101")

# Render visualization
output = renderer.render_visualization(
    image=original_pil_image,
    normalized_heatmap=heatmap_2d_array,
    context=ctx,
    export_dir="artifacts/xai/",
)

# Output structure
print(output["overlay_image"])         # PIL.Image
print(output["important_regions"])     # List[dict]
print(output["visualization_metrics"]) # Dict[str, float]
print(ctx.overlay_path)                # "artifacts/xai/req-101_overlay.png"
```

---

## 6. Exception Hierarchy

```
VisualizationException
├── ColorMappingError        (invalid or unsupported colormap)
├── OverlayBlendingError     (dimension mismatch or alpha ratio failure)
├── RegionExtractionError    (contour or region detection error)
├── VisualizationQualityError(quality metric computation failure)
└── ExportError              (disk write failure)
```

---

## 7. Verification Results

```
============================= test session starts =============================
medical_ai_engine/explainability/tests/test_heatmap_visualization.py . PASSED (17 tests)
medical_ai_engine/explainability/tests/test_gradcam_engine.py ........ PASSED (18 tests)
medical_ai_engine/explainability/tests/test_cam_registry.py .......... PASSED (30 tests)
medical_ai_engine/explainability/tests/test_explainability_architecture.py PASSED (52 tests)
============================= 117 passed in 2.20s ==============================
```
