# Distilling Models Toolkit (distillhelpersv4)



Reusable toolkit for training, evaluation, distillation, hyperparameter optimization, and MLflow experiment logging — designed for teaching, student projects, and research.

---

## **Table of Contents**

1.	Overview
2.	Environment Options
3.	Installation
4.	Getting Started
5.	Toolkit Structure
6.	Verifying Installation
7.	Using the Toolkit
8.	Troubleshooting
9.  Recommended for Students
10. Changelog

---

## **1. Overview**

This package provides:

*	**Core helpers (src/core/)** → device selection, training, evaluation, metrics, visualization.

*	**Logging (src/logging/)** → MLflow integration (standard + distillation).

*	**Optuna (src/optuna/)** → multi-objective optimization (Accuracy + ECE), Pareto front analysis, study archives.

*	**Distillation (src/distillation/)** → teacher–student training loops with calibration support.

*	**Models (src/models/)** → MLPs, tabular transformers, wrappers.

> **Designed to support both predictive modeling classes and undergraduate thesis projects.**

---

## **2. Environment Options**

Choose the right YAML based on your hardware and use case:

| File | Hardware |	Packages Included |	Use Case |
|------|----------|-------------------|----------|
| **`distillhelpersv4_cpumin.yml`** |	CPU-only |	Minimal: PyTorch (CPU), sklearn, MLflow, Optuna, XGBoost, matplotlib, pandas, numpy, nbformat, pytest |	Teaching labs / weak machines |
| **`distillhelpersv4_cpu.yml`** |	CPU-only |	Full: PyTorch (CPU), sklearn, MLflow, Optuna, XGBoost, HuggingFace, TabNet, SHAP, Plotly |	Laptops/desktops w/out GPU; full modeling support
| **`distillhelpersv4_cuda.yml`** | NVIDIA GPU |	Full GPU stack: PyTorch + CUDA, sklearn, MLflow, Optuna, XGBoost (GPU), HuggingFace, TabNet, SHAP |	Research servers / GPU laptops |
| **`distillhelpersv4_cudamin.yml`** |	NVIDIA GPU |	Minimal GPU stack: PyTorch + CUDA, sklearn, MLflow, Optuna, XGBoost, matplotlib, pandas, numpy	| For GPU users who want essentials only |
| **`distillhelpersv4_mps.yml`** |	Apple Silicon (M1/M2/M3/M4) |	Full: PyTorch (MPS backend), sklearn, MLflow, Optuna, XGBoost, HuggingFace, TabNet, SHAP, Plotly |	Mac users with Metal backend |
| **`distillhelpersv4_mpsmin.yml`** |	Apple Silicon |	Minimal: PyTorch (MPS), sklearn, MLflow, Optuna, XGBoost, matplotlib, pandas, numpy | Mac users who want a lightweight stack |
| **`distillhelpersv4_tools.yml`** |	Any |	Tools-only: MLflow, nbformat, pytest, matplotlib, pandas, kaleido |	For reports, MLflow dashboards, notebook scanning
| **`distillhelpersv4_all.yml`** |	Any |	Everything included (CPU stack + tools + extras) |	For researchers who want all features in one env |

---

## **3. Installation**

```bash
conda env create -f environment/distillhelpersv4_cpu.yml
conda activate distillhelpersv4_cpu
```

**(Replace distillhelpersv4_cpu.yml with any other YAML file depending on your hardware.)**

---

## **4. Getting Started**

```python
import src

# Print toolkit version

src.print_version()
```

> **Always add this at the top of your notebooks to ensure runs are tagged with the toolkit version.**

---

## **5. Toolkit Structure**

```python
src/
 ├── core/          # Device, training, metrics, visualization, versioning
 ├── logging/       # MLflow loggers (standard + distillation)
 ├── optuna/        # Objectives, runner, archive
 ├── distillation/  # Teacher–student training loops
 ├── models/        # MLPs, tabular transformers, wrappers
```

---

## **6. Verifying Installation**

```python
python -c "import torch; print(torch.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
python -c "import optuna; print(optuna.__version__)"
```

---

## **7. Using the Toolkit**

*	**Training** → train_model, evaluate_model

*	**Distillation** → train_with_distillation, log_results_with_mlflow_distillation

*	**Optuna** → run_optuna_with_mlflow (multi-objective: accuracy + ECE)

*	**Visualization** → auto-generated training curves, calibration curves, confusion matrices

---

##	**8. Troubleshooting**

*	**No runs in MLflow UI** → ensure you are inside the folder containing mlruns/ when launching mlflow ui.

*	**CUDA mismatch** → check nvidia-smi and align pytorch-cuda=xx.x in your YAML.

*	**MPS unavailable** → update macOS ≥ 12.3 and PyTorch ≥ 1.12.

---

##	**9. Recommended for Students**

*	Start with minimal YAML (cpumin, mpsmin, or cudamin).

*	Install extras (TabNet, Transformers, SHAP) only when needed:

	```bash
	pip install pytorch-tabnet transformers datasets evaluate shap
	```



## **10. Changelog**

We maintain a **`CHANGELOG.md`** file to track all updates to this toolkit.

Each release follows semantic versioning (**`MAJOR.MINOR.PATCH`**).

**Template**:

```markdown
# Changelog

## [Unreleased]
### Added
- New features not yet released.

### Changed
- Updates to existing functionality.

### Fixed
- Bug fixes.

---

## [1.0.0] - 2025-09-20
### Added
- Initial release of `distillhelpersv4`
- Core helpers (training, evaluation, visualization)
- Logging (MLflow standard + distillation)
- Optuna integration (multi-objective: Accuracy + ECE)
- Distillation loops + demo notebooks
```

