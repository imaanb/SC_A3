# Project Overview

This repository contains all materials related to the study of eigenmodes and Laplacian matrices for various geometric shapes. It includes Python scripts, computational results, visualizations, and animations.

## Directory Structure

- [`README.md`](README.md) - This file, describing the repository structure.
- [`assignment.pdf`](assignment.pdf) - The assignment or research paper detailing the problem statement and methodology.
- [`final.ipynb`](final.ipynb) - Jupyter Notebook with all the final solutions used in the research paper.

### Figures
- [`figs/`](figs) - Contains all visualization outputs, including eigenmodes, Laplacian heatmaps, and spectral plots.
  - Eigenmode solutions for different geometries (circle, square, rectangle) are provided.
  - Comparisons between dense and sparse representations.
  - Example files: [`circle_eigenmode0.png`](figs/circle_eigenmode0.png), [`spectrum of eigenfrequencies.png`](figs/spectrum%20of%20eigenfrequencies.png).

### GIFs
- [`gifs/`](gifs) - Contains animations of eigenmodes.
  - Example: [`circle_eigenmode_animation.gif`](gifs/circle_eigenmode_animation.gif).

### Results
- [`results/`](results) - Stores computed Laplacian matrices and eigenvectors.
  - Example files: [`circle_laplacian_matrix.txt`](results/circle_laplacian_matrix.txt), [`eigenvectors.txt`](results/eigenvectors.txt).

### Source Code
- [`src/`](src) - Contains Python scripts used for computation.
  - [`Eigens.py`](src/Eigens.py) - Computes eigenvalues and eigenvectors.
  - [`Laplacians.py`](src/Laplacians.py) - Constructs Laplacian matrices for different domains.
  - [`utils/`](src/utils) - Contains utility functions for data processing.

## Usage
Ensure you have the required dependencies installed and run the Jupyter notebook [`final.ipynb`](final.ipynb) to reproduce results and visualizations.

```bash
pip install -r requirements.txt
jupyter notebook final.ipynb
```

## License
This project is distributed under an open-source license. Refer to `LICENSE` for details.