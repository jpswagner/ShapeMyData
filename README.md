# Shape Area Chart (Pixel-Exact)

A Streamlit application that generates area-proportional charts within a custom shape (mask), ensuring pixel-exact proportions.

## Features

*   **Pixel-Exactness**: The number of pixels assigned to each category corresponds exactly to the target percentage.
*   **Custom Shapes**: Use the default map (RS) or upload your own PNG mask.
*   **Segmentation Modes**: Vertical, Horizontal, Angular, and Contiguous Wedge.
*   **High-Resolution Rendering**: Support for upscaling (High DPI) and PDF export.
*   **Robust Input**: CSV import, mask validation, and auto-detection from shapes.

## Structure

*   `app/`: Application source code.
    *   `app.py`: Main Streamlit entrypoint.
    *   `mask_utils.py`: Mask loading and processing utilities.
    *   `segmentation.py`: Pixel assignment logic.
    *   `render.py`: Visualization and export logic.
*   `assets/`: Default assets (masks).
*   `tests/`: Automated tests for math and logic verification.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run app/app.py
    ```

## Usage

1.  **Select Shape**: Use the default Rio Grande do Sul mask or upload a PNG.
    *   **Mask PNG (Alpha)**: Image where transparent pixels are ignored.
    *   **Auto-detect**: Image where shape is detected by color saturation.
2.  **Input Data**:
    *   Manually enter labels and values in the sidebar/expander.
    *   Or upload a CSV with columns: `label`, `value`, `color` (optional).
3.  **Configure**:
    *   Choose segmentation mode (Vertical, Horizontal, Angular, Contiguous Wedge).
    *   Adjust margins, fonts, and resolution.
4.  **Export**:
    *   Download as PNG or PDF.

## Development

### Running Tests
```bash
pytest
```

## Roadmap

1.  **Vector Export (SVG)**: Implement vector path tracing for true SVG export.
2.  **Performance Optimization**: Use Numba or C-extensions for very large masks (>4k).
3.  **Smart Palettes**: Add colorblind-safe and image-based palette generation.
4.  **Interactive Editing**: Click-to-edit labels and colors on the preview.
