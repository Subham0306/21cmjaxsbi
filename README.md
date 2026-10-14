# 21cmjaxSBI (v0.1.0)

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.34515834-blue.svg)](https://doi.org/10.6084/m9.figshare.34515834)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Backend-JAX%2FFlax-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`21cmjaxSBI` is a high-performance, **Simulation-Based Inference (SBI)** framework built in **JAX and Flax**. It implements **Neural Posterior Estimation (NPE)** with Mixture Density Networks (MDNs) to perform rapid, likelihood-free cosmological parameter constraints for the **Epoch of Reionization (EoR)**.

Presented at the **International Center for Theoretical Physics (ICTP) Summer School on Cosmology (Trieste, Italy), 2026**.

---

##  Key Features

* **JAX-Accelerated Engine:** Fully vectorized and JIT-compiled pipeline for rapid training and evaluation.
* **Mixture Density Network (MDN):** Parameterizes complex, multi-modal posteriors using flexible Gaussian mixture components with Cholesky-factorized covariance matrices.
* **Realistic SKA Noise Injection:** Incorporates thermal and total noise realizations tailored for Square Kilometre Array (SKA) sensitivity curves.
* **Rigorous Validation Suite:** Includes out-of-the-box Simulation-Based Calibration (coverage tests, 68%/95% credible interval checks) and $z$-score residual diagnostics.

---

##  Code Modules

1. **`data.py`**: Ingests Licorice / 21cmFAST light-cone power spectra, interpolates onto fixed $k$-bins, and applies log-scaling normalization.
2. **`model.py`**: Defines the Flax `NPE` architecture with shared trunk layers, mixture weights, means, and Cholesky covariance blocks.
3. **`loss.py`**: Implements stable Gaussian Negative Log-Likelihood (NLL) with `logsumexp` mixture combination.

While this repository provides modularized source files under `src/cmjaxsbi/` (covering data handling, MDN architectures, and loss functions) for clean programmatic imports, users and reviewers exploring this initial release are strongly encouraged to refer to **`notebooks/full_pipeline.ipynb`**. This comprehensive notebook serves as the primary reference guide for version `0.1.0`, walking through the entire end-to-end workflow—including simulation data preparation, dynamic SKA noise injection, neural posterior training, and diagnostic evaluation—complete with executed code blocks and visual results.

---

##  Installation

Clone the repository and install in editable mode:

```bash
git clone [https://github.com/your-username/21cmjaxsbi.git](https://github.com/Subham0306/21cmjaxsbi.git)
cd 21cmjaxsbi
pip install -e .
```


##  About the Author & Collaboration

This software framework was developed by **Subham Banerjee**, a graduate student in Astrophysics at the **University of Bonn** and guest researcher in the Radio Cosmology group at the **Max Planck Institute for Radio Astronomy (MPIfR)** in Bonn, Germany. 

The project is conducted under the academic supervision of **Prof. Jonathan R. Pritchard** (MPIfR / Imperial College London), focusing on scalable machine learning methods, simulation-based inference, and 21-cm cosmology for the Epoch of Reionization and Cosmic Dawn.

* **GitHub:** [@Subham0306](https://github.com/Subham0306)
* **Institutional Affiliations:** University of Bonn | Max Planck Institute for Radio Astronomy

---

##  Roadmap & Future Versions (`v0.2.0+`)

While `v0.1.0` establishes the foundational simulator-agnostic JAX/Flax NPE pipeline with dynamic SKA noise injection, upcoming releases will expand the methodology across higher-order summary statistics and physical models:

* **`v0.2.0` — Higher-Order Topological Statistics:** 
  Integration of 2D/3D Minkowski Functionals and Betti numbers computed directly on 21-cm spatial brightness temperature maps to extract non-Gaussian structural information beyond the power spectrum.

##  Citation

If you use `21cmjaxSBI` in your academic research, PhD application portfolios, or publications, please cite it using the following BibTeX entry:

```bibtex
@software{banerjee2026_21cmjaxsbi,
  author       = {Subham Banerjee},
  title        = {21cmjaxSBI: A Simulator-Agnostic JAX/Flax NPE Framework for 21-cm Cosmology},
  month        = oct,
  year         = 2026,
  publisher    = {Figshare},
  version      = {v0.1.0},
  doi          = {10.6084/m9.figshare.34515834},
  url          = {[https://doi.org/10.6084/m9.figshare.34515834](https://doi.org/10.6084/m9.figshare.34515834)}
}
