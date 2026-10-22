# Medical Diagnosis Using Quantum Classification ⚛️🏥
### A Comparative Study of Classical SVM and Quantum Kernel QSVC

**Submitted by:** RANJAN KUMAR SAHOO  
**Academic Year:** 2026 - 2027  
**Dataset:** Wisconsin Diagnostic Breast Cancer (UCI Repository / `load_breast_cancer`)

---

## 🌟 Executive Overview
This academic project implements a hybrid quantum-classical machine learning pipeline to classify fine-needle aspiration breast mass samples as **Benign** or **Malignant**.

### Key Pipeline Highlights
- **Preprocessing Pipeline**: `StandardScaler()` ➔ `PCA(n_components=4)` ➔ `MinMaxScaler(feature_range=(0, π))` fitted strictly on training data to prevent leakage.
- **Classical Model Baseline**: Support Vector Machine (`SVC`) with Radial Basis Function (`RBF`) kernel ($C=2.0$, `class_weight='balanced'`).
- **Quantum Feature Map**: 4-qubit Second-Order Pauli Expansion ($ZZ$-Feature Map, $2$ repetitions, linear entanglement).
- **Quantum Kernel**: `FidelityStatevectorKernel` computing pairwise quantum state overlap $|\langle \Phi(x) | \Phi(z) \rangle|^2$.
- **Quantum Classifier**: Quantum Support Vector Classifier (`QSVC`) trained on the fidelity kernel.
- **Interactive Web UI Dashboard**: Fully-featured glassmorphism Web Application for real-time patient diagnosis simulation, metric comparison, 4-qubit circuit inspection, interactive kernel matrix heatmaps, Viva Q&A quiz, and academic report reader.

---

## 📁 Repository Structure

```
micro_project/
├── main.py                     # Execution entry point (Trains models & saves all evidence)
├── server.py                   # FastAPI / Uvicorn Backend Web Server & REST API
├── requirements.txt            # Python environment dependencies
├── README.md                   # Repository documentation
├── static/                     # Web User Interface assets
│   ├── index.html              # Modern single-page web UI dashboard
│   ├── css/
│   │   └── style.css           # Glassmorphism Quantum Medical design system styling
│   └── js/
│       └── app.js              # Frontend interactive controller & state manager
└── results/                    # Generated ML models, plots, metrics & state matrices
    ├── model_comparison.csv    # Evaluated metric scores table
    ├── test_predictions.csv    # Per-sample predictions & probabilities
    ├── classical_confusion_matrix.png
    ├── quantum_confusion_matrix.png
    ├── roc_curve_comparison.png
    ├── model_metric_comparison.png
    ├── quantum_train_kernel.npy# 455x455 training kernel state overlap matrix
    ├── quantum_test_kernel.npy # 114x455 testing kernel state overlap matrix
    ├── quantum_circuit.txt     # Decomposed 4-qubit ZZ Feature Map ASCII circuit
    ├── preprocessing_pipeline.joblib
    ├── classical_svm.joblib
    └── quantum_qsvc.joblib
```

---

## 🚀 Quick Start Guide

### 1. Run Offline Pipeline Execution
To run the full quantum machine learning pipeline and generate all artifacts:
```bash
py main.py
```

### 2. Launch the Interactive Web User Interface
To start the web server and open the UI dashboard:
```bash
py server.py
```
Then open your web browser at:
`http://127.0.0.1:8000`

---

## 📊 Evaluation Metrics Summary

| Metric | Classical RBF-SVM | Quantum QSVC | Note |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 0.965 | 0.965 | Identical baseline performance |
| **Precision (Malignant)** | 0.975 | 0.975 | High precision on positive class |
| **Sensitivity / Recall (Malignant)** | 0.929 | 0.929 | Clinically critical metric |
| **Specificity (Benign)** | 0.986 | 0.986 | Low false positive rate |
| **F1 Score (Malignant)** | 0.951 | 0.951 | Balanced harmonic mean |
| **ROC-AUC** | 0.992 | 0.986 | High discriminative ranking |

---

## ⚠️ Educational Disclaimer
This software is an educational machine-learning prototype developed for academic demonstration. It is not intended, validated, or approved for clinical diagnosis or treatment decisions.
