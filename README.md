# OpenCV-Starter-
A concise starter to try OpenCV on images of Archaeological reliefs and artifacts
# OpenCV Starter — Göbekli Tepe Image Enhancement

A Python image enhancement toolkit for processing photographs of archaeological
stone carvings and low-relief artifacts. Built for the Göbekli Tepe imaging project.

---

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

---

## Setup

1. Clone the repository
2. Open `enhancement_notebook_starter.ipynb`
3. Set `REPO_PATH` to your local clone location
4. Set `IMAGE_PATH` to your image file
5. Run the cells

---

## Repository Contents

| File | Description |
|---|---|
| `enhancement.py` | Core image enhancement functions |
| `image_segmentation.py` | Thresholding and segmentation functions |
| `feature_classification.py` | Edge detection, SIFT, and histogram analysis |
| `enhancement_notebook_starter.ipynb` | Starter notebook with example workflow |

---

## Functions

### enhancement.py

| Function | Description |
|---|---|
| `erode_image()` | Morphological erosion |
| `dilate_image()` | Morphological dilation |
| `histogram_equalization()` | Global contrast enhancement |
| `gradient_enhancement()` | Sobel-based edge enhancement |
| `sharpen_image()` | Fixed kernel sharpening |
| `clahe_enhancement()` | Adaptive local contrast enhancement |
| `bilateral_filter()` | Edge-preserving noise reduction |
| `unsharp_mask()` | Controllable detail sharpening |
| `blackhat_transform()` | Isolates dark recessed features |
| `tophat_transform()` | Isolates bright raised features |
| `gamma_correction()` | Adjusts midtone brightness |
| `enhance_stone_carving()` | Light bilateral denoise + unsharp mask |
| `isolate_carving_lines()` | Extracts carved lines via adaptive threshold |
| `clean_line_mask()` | Removes noise blobs from a line mask |
| `overlay_lines_on_grayscale()` | Colour overlay of line mask on base image |
| `overlay_inverted_lines_on_grayscale()` | Blended inverted line overlay |
| `enhance_deep_relief()` | Pipeline for deeply carved/shadowed subjects |
| `detect_carving_edges()` | Canny-based edge detection for carved lines |
| `enhance_low_relief()` | Full layered pipeline for shallow incisions |
| `enhance_low_relief_sharp()` | Low-blur variant for fine carved detail |

### image_segmentation.py

| Function | Description |
|---|---|
| `otsu_threshold()` | Global threshold using Otsu's method |
| `adaptive_threshold()` | Local neighbourhood thresholding |

### feature_classification.py

| Function | Description |
|---|---|
| `edge_detection()` | Canny edge detection |
| `sift_feature_detection()` | SIFT keypoints and descriptors |
| `histogram_analysis()` | Pixel intensity histogram with plot |

---

## Pipeline Selection Guide

| Image type | Recommended approach |
|---|---|
| Evenly lit, shallow incisions | `enhance_low_relief(gray, gamma=0.8)` |
| Fine lines being blurred | `enhance_low_relief_sharp(gray, gamma=0.8)` |
| Deeply carved or heavily shadowed | `enhance_deep_relief(gray)` |
| Good contrast, even lighting | `histogram_equalization()` → `sharpen_image()` |
| Overexposed highlights | Clip first, then pipeline (see below) |
| Lighter carvings on dark ground | Invert first, then pipeline (see below) |

---

## Example Usage

```python
import cv2
import numpy as np
from enhancement import enhance_low_relief

img  = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Standard pipeline — good starting point for most images
result = enhance_low_relief(gray, gamma=0.8)

# For overexposed images — clip highlights before processing
clipped = np.clip(gray, 0, 200)
clipped = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
result  = enhance_low_relief(clipped, gamma=0.8)

# For lighter-on-dark carvings — invert before processing
inverted = cv2.bitwise_not(gray)
result   = enhance_low_relief(inverted, gamma=0.8)

# Save result
cv2.imwrite('output.png', result)
```

---

## Notes

- All functions accept BGR or grayscale numpy arrays and handle conversion internally
- Parameters for all functions are documented in the function docstrings in `enhancement.py`
- The `gamma` parameter in `enhance_low_relief()` defaults to `1.0` (off) — pass a value
  below `1.0` (e.g. `0.8`) to deepen midtone contrast on most carving images
