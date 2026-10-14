# 21cmjaxsbi (v0.1.0)

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.34515834-blue.svg)](https://doi.org/10.6084/m9.figshare.34515834)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Backend-JAX%2FFlax-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`21cmjaxsbi` is a high-performance, simulator-agnostic **Simulation-Based Inference (SBI)** framework built in **JAX and Flax**. It implements **Neural Posterior Estimation (NPE)** with Mixture Density Networks (MDNs) to perform rapid, likelihood-free cosmological parameter constraints for the **Epoch of Reionization (EoR)**.

Presented at the **ICTP Summer School on Cosmology (Trieste, Italy)**.

---

## 🚀 Key Features

* **JAX-Accelerated Engine:** Fully vectorized and JIT-compiled pipeline for rapid training and evaluation.
* **Mixture Density Network (MDN):** Parameterizes complex, multi-modal posteriors using flexible Gaussian mixture components with Cholesky-factorized covariance matrices.
* **Realistic SKA Noise Injection:** Incorporates thermal and total noise realizations tailored for Square Kilometre Array (SKA) sensitivity curves.
* **Rigorous Validation Suite:** Includes out-of-the-box Simulation-Based Calibration (coverage tests, 68%/95% credible interval checks) and $z$-score residual diagnostics.

---

## 🛠️ Code Modules

1. **`data.py`**: Ingests Licorice / 21cmFAST light-cone power spectra, interpolates onto fixed $k$-bins, and applies log-scaling normalization.
2. **`model.py`**: Defines the Flax `NPE` architecture with shared trunk layers, mixture weights, means, and Cholesky covariance blocks.
3. **`loss.py`**: Implements stable Gaussian Negative Log-Likelihood (NLL) with `logsumexp` mixture combination.

While this repository provides modularized source files under `src/cmjaxsbi/` (covering data handling, MDN architectures, and loss functions) for clean programmatic imports, users and reviewers exploring this initial release are strongly encouraged to refer to **`notebooks/full_pipeline.ipynb`**. This comprehensive notebook serves as the primary reference guide for version `0.1.0`, walking through the entire end-to-end workflow—including simulation data preparation, dynamic SKA noise injection, neural posterior training, and diagnostic evaluation—complete with executed code blocks and visual results.

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone [https://github.com/your-username/21cmjaxsbi.git](https://github.com/Subham0306/21cmjaxsbi.git)
cd 21cmjaxsbi
pip install -e .
