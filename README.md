# Shape Area Chart (Pixel-Exact)

A Streamlit application that generates area-proportional charts within a custom shape (mask), ensuring pixel-exact proportions.

Avaliable at: [ShapeMyData](https://shapemydata.streamlit.app/)

## Features

*   **Pixel-Exactness**: The number of pixels assigned to each category corresponds exactly to the target percentage.
*   **Custom Shapes**:
    *   **Presets**: Select shapes from the **Brazil** and **States** datasets (vector based).
    *   **Upload**: Upload your own PNG mask.
*   **Segmentation Modes**: Vertical, Horizontal, Angular, and Contiguous Wedge.
*   **High-Resolution Rendering**: Support for upscaling (High DPI) and PDF export.
*   **Robust Input**: Mask validation and auto-detection from shapes.

## Usage

1.  **Select Shape**:
    *   **Preset: Brazil / States**: Uses vector shapes (WKT). Select "Brazil" for country or "States" for specific UF.
    *   **Upload**: Upload a PNG.
2.  **Input Data**:
    *   Manually enter labels and values in the sidebar/expander.
3.  **Configure**:
    *   Choose segmentation mode (Vertical, Horizontal, Angular, Contiguous Wedge).
    *   Adjust margins, fonts, and resolution.
4.  **Export**:
    *   Download as PNG or PDF.





**Use in Research or Publications**

If you use ShapeMyData in a scientific publication, academic project, or report, please cite it using the following format:

Wagner, J. P. S. (2026). ShapeMyData [Computer software]. Available at https://github.com/jpswagner/ShapeMyData

Citation is appreciated as it helps support the continued development of this tool.
