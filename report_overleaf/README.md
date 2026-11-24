# Report: Overleaf-ready version

Upload this folder as a zip to Overleaf (New Project > Upload Project) and compile with pdfLaTeX.

- `main.tex` is the full report. The seven diagrams are included as ready-made PDFs (`figures/diagram1.pdf` ... `diagram7.pdf`), so no TikZ runs while compiling.
- `diagram_sources/` holds the TikZ code of each diagram. To change a diagram, compile its `.tex` file on its own and replace the matching PDF in `figures/`.
- `report_final/main.tex` in the repository is the same report with the diagrams drawn inline.
